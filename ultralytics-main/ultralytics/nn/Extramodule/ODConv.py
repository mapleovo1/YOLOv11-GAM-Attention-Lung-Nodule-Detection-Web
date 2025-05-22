import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=True)  # 改为 bias=True
        # 移除 BatchNorm，因为在 1x1 特征图上 BatchNorm 会出问题
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)  # 直接应用 ReLU，移除 BatchNorm
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4, bias=False):  # 添加 bias 参数
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)

        # 添加 bias 支持
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_planes))
        else:
            self.register_parameter('bias', None)

        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

        # 初始化 bias
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention

        # 添加 bias
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)

        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=self.bias, stride=self.stride,
                          padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)

    def extra_repr(self):
        """返回关于模块的额外表示字符串"""
        return 'in_planes={}, out_planes={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, kernel_num={}'.format(
            self.in_planes, self.out_planes, self.kernel_size, self.stride,
            self.padding, self.dilation, self.groups, self.kernel_num
        )


# YOLO 框架兼容的包装器
class ODConv(nn.Module):
    """
    ODConv 包装器，用于在 YOLO 框架中使用 ODConv2d
    兼容 YOLO 的 Conv 层接口
    """
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self,
                 c1,  # 输入通道数
                 c2,  # 输出通道数
                 k=1,  # 卷积核大小
                 s=1,  # 步长
                 p=None,  # 填充
                 g=1,  # 分组卷积
                 d=1,  # 膨胀率
                 act=True,  # 是否使用激活函数
                 reduction=0.0625,  # 注意力通道缩减比例
                 kernel_num=4):  # 多核数量
        super().__init__()

        # 自动计算填充
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]

        # ODConv2d 卷积层
        self.conv = ODConv2d(
            in_planes=c1,
            out_planes=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=g,
            reduction=reduction,
            kernel_num=kernel_num,
            bias=False  # YOLO 中通常使用 BN，所以 conv 不用 bias
        )

        # 批归一化
        self.bn = nn.BatchNorm2d(c2)

        # 激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """前向传播"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """融合的前向传播（用于推理优化）"""
        return self.act(self.conv(x))