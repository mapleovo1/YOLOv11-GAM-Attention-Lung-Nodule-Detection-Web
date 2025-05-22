import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # 计算Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # 缩放点积注意力
        attn = attn.softmax(dim=-1)

        # 加权聚合
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 img_size=20,
                 patch_size=5,  # 对于20x20的特征图，我们使用5x5的patch
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 drop_rate=0.1):
        super(VisionTransformer, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])

        # 输出投影
        self.norm = nn.LayerNorm(embed_dim)

        # 将特征图恢复回原始尺寸 (H, W)
        self.deconvolution = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)

        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # 输入: [B, C, H, W] -> [64, 256, 20, 20]
        batch_size, _, _, _ = x.shape

        # Patch Embedding: [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size]
        if not list(x.shape)[1:] == [256, 20, 20]:
            return x
        x = self.patch_embed(x)  # [64, 256, 4, 4]

        # 展平空间维度: [B, embed_dim, H', W'] -> [B, embed_dim, H'*W']
        x = x.flatten(2)  # [64, 256, 16]

        # 交换维度: [B, embed_dim, H'*W'] -> [B, H'*W', embed_dim]
        x = x.transpose(1, 2)  # [64, 16, 256]

        # 添加位置编码
        x = x + self.pos_embed

        # 应用Transformer块
        for block in self.blocks:
            x = block(x)

        # 应用最后的LayerNorm
        x = self.norm(x)

        # 变换维度: [B, H'*W', embed_dim] -> [B, embed_dim, H', W']
        x = x.transpose(1, 2).reshape(batch_size, -1, self.img_size // self.patch_size,
                                      self.img_size // self.patch_size)

        # 反卷积恢复维度: [B, embed_dim, H', W'] -> [B, out_channels, H, W]
        x = self.deconvolution(x)  # [64, 256, 20, 20]

        return x