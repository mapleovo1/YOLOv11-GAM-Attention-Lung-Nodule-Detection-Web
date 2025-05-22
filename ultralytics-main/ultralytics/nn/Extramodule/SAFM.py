import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSAFM(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.dim = dim
        self.chunk_dim = dim // ratio

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.dwconv = nn.Conv2d(self.chunk_dim, self.chunk_dim, 3, 1, 1, groups=self.chunk_dim, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        x0, x1 = self.proj(x).split([self.chunk_dim, self.dim - self.chunk_dim], dim=1)

        x2 = F.adaptive_max_pool2d(x0, (h // 8, w // 8))
        x2 = self.dwconv(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')
        x2 = self.act(x2) * x0

        x = torch.cat([x1, x2], dim=1)
        x = self.out(self.act(x))
        return x


class CCM(nn.Module):
    def __init__(self, dim, ffn_scale, use_se=False):
        super().__init__()
        self.use_se = use_se
        hidden_dim = int(dim * ffn_scale)

        self.conv1 = nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x


class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale, use_se=False):
        super().__init__()

        self.conv1 = SimpleSAFM(dim, ratio=3)

        self.conv2 = CCM(dim, ffn_scale, use_se)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class SAFMNPP(nn.Module):
    def __init__(self, input_dim, dim, n_blocks=6, ffn_scale=1.5, use_se=False, upscaling_factor=2):
        super().__init__()
        self.scale = upscaling_factor

        self.to_feat = nn.Conv2d(input_dim, dim, 3, 1, 1, bias=False)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale, use_se) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, input_dim * upscaling_factor ** 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        res = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.to_feat(x)
        x = self.feats(x)
        return self.to_img(x) + res