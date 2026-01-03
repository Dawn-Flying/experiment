"UNet.py"
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AttentionBlock(nn.Module):

    def __init__(self, channels, num_groups=32):
        super().__init__()

        self.channels = channels

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)

        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x):

        B, C, H, W = x.shape

        q, k, v = torch.split(self.qkv(self.norm(x)), self.channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)

        dot_products = torch.bmm(q, k) * (C ** (-0.5))
        assert dot_products.shape == (B, H * W, H * W)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)

        assert out.shape == (B, H * W, C)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)

        return F.selu(self.output(out) + x)


class DownsampleResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, num_groups=32, use_attention=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

        self.attention = AttentionBlock(out_channels, num_groups=num_groups) if use_attention else nn.Identity()

    def forward(self, x, c=None):

        x = x + c if c is not None else x

        out = F.selu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.selu(out)
        out = self.attention(out)

        return out


class UpsampleResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, num_groups=32, use_attention=False):
        super().__init__()

        self.dconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2 + stride, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.dconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2 + stride, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        ) if stride != 1 or in_channels != out_channels else nn.Identity()

        self.attention = AttentionBlock(out_channels, num_groups=num_groups) if use_attention else nn.Identity()

    def forward(self, x, c=None):

        x = x + c if c is not None else x
        out = F.selu(self.norm1(self.dconv1(x)))
        out = self.norm2(self.dconv2(out))
        out += self.shortcut(x)
        out = F.selu(out)
        out = self.attention(out)

        return out


class UNet(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            block_channels=[64, 128, 256, 512, 1024],
            use_attention=[False, False, False, False, True],
            num_groups=32,
    ):
        super().__init__()

        assert len(block_channels) == len(use_attention)

        self.conv = nn.Conv2d(in_channels, block_channels[0], kernel_size=1, bias=False)

        downsample = []
        upsample = []

        for i in range(len(block_channels) - 1):

            layer = nn.ModuleDict()
            layer["t_embedder1"] = TimestepEmbedder(block_channels[i])
            layer["t_embedder2"] = TimestepEmbedder(block_channels[i])
            layer["blocks"] = nn.ModuleList([
                DownsampleResBlock(
                    block_channels[i],
                    block_channels[i],
                    stride=1,
                    num_groups=num_groups,
                    use_attention=use_attention[i]
                ),
                DownsampleResBlock(
                    block_channels[i],
                    block_channels[i+1],
                    stride=2,
                    num_groups=num_groups,
                    use_attention=use_attention[i+1]
                )
            ])

            downsample.append(layer)

        self.downsample = nn.ModuleList(downsample)

        for j in reversed(range(1, len(block_channels))):

            layer = nn.ModuleDict()
            layer["t_embedder1"] = TimestepEmbedder(block_channels[j])
            layer["t_embedder2"] = TimestepEmbedder(2 * block_channels[j-1])
            layer["blocks"] = nn.ModuleList([
                UpsampleResBlock(
                    block_channels[j],
                    block_channels[j-1],
                    stride=2,
                    num_groups=num_groups,
                    use_attention=use_attention[j]
                ),
                UpsampleResBlock(
                    2 * block_channels[j-1],
                    block_channels[j-1],
                    stride=1,
                    num_groups=num_groups,
                    use_attention=use_attention[j-1]
                )
            ])

            upsample.append(layer)

        self.upsample = nn.ModuleList(upsample)

        self.output = nn.Conv2d(block_channels[0], out_channels, kernel_size=1, bias=False)

    def forward(self, x, t):

        x = self.conv(x)

        skip_features = []

        for d_layer in self.downsample:
            t_emb1 = d_layer["t_embedder1"](t).unsqueeze(-1).unsqueeze(-1)
            x = d_layer["blocks"][0](x, t_emb1)
            skip_features.append(x)
            t_emb2 = d_layer["t_embedder2"](t).unsqueeze(-1).unsqueeze(-1)
            x = d_layer["blocks"][1](x, t_emb2)

        skip_features.reverse()

        for i, up_layer in enumerate(self.upsample):
            t_emb1 = up_layer["t_embedder1"](t).unsqueeze(-1).unsqueeze(-1)
            x = up_layer["blocks"][0](x, t_emb1)
            t_emb2 = up_layer["t_embedder2"](t).unsqueeze(-1).unsqueeze(-1)
            x = torch.cat([x, skip_features[i]], dim=1)
            x = up_layer["blocks"][1](x, t_emb2)

        x = self.output(x)

        return x

if __name__ == '__main__':

    device = "cuda"
    batch_size = 3
    model = UNet(in_channels=3, out_channels=3).to(device)

    x = torch.rand((batch_size, 3, 32, 32)).to(device)
    t = torch.randint(low=0, high=1000, size=(batch_size,)).to(device)

    y = model(x, t)
    print(y.shape)
