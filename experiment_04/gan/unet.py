import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if time_emb_dim is not None
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale
        sim = torch.einsum("b h c i, b h c j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h c j -> b h c i", attn, v)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", x=h, y=w)
        return self.to_out(out)

class AttnBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, dim)
        self.attn = Attention(dim)

    def forward(self, x):
        x = self.group_norm(x)
        return self.attn(x) + x

class UNet(nn.Module):
    def __init__(
        self,
        dim=32,
        init_dim=None,
        out_dim=None,
        dim_mults=[1, 2, 2, 4],
        channels=3,
        base_channels=128,
        resnet_block_groups=8,
        time_emb_dim=512,
    ):
        super().__init__()
        self.channels = channels
        input_channels = channels
        init_dim = base_channels if init_dim is None else init_dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                        AttnBlock(dim_out) if dim_out >= 256 else nn.Identity(),
                        nn.AvgPool2d(2) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.mid_attn = AttnBlock(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 2)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                        AttnBlock(dim_in) if dim_in >= 256 else nn.Identity(),
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        out_dim = channels if out_dim is None else out_dim
        self.final_res_block = ResnetBlock(base_channels * 2, base_channels, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(base_channels, out_dim, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        t = self.time_mlp(time)

        h = []
        # Downsampling
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)  # 保存当前分辨率的 feature
            x = downsample(x)  # 然后下采样

        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsampling
        for block1, block2, attn, upsample in self.ups:
            # 从 h 中取出对应的 skip feature（从后往前）
            skip = h.pop()
            # 确保尺寸一致（可加 assert 调试）
            if x.shape[2:] != skip.shape[2:]:
                # 如果尺寸不一致，裁剪 skip（一般不会发生，除非 padding 问题）
                skip = skip[:, :, :x.shape[2], :x.shape[3]]
            x = torch.cat((x, skip), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        # 注意：此时 h 应该还剩一个元素（最浅层的），但我们不用它！
        # 因为 self.ups 的数量 = len(self.downs) - 1

        # Final block
        x = self.final_res_block(x, t)
        return self.final_conv(x)