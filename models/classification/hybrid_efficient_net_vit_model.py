import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# --- Helper Modules ---


class Conv2d(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size=3,
        padding=1,
        stride=1,
        groups=1,
        with_bn=True,
        with_act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.with_bn = with_bn
        self.with_act = with_act
        if self.with_bn:
            self.bn = nn.BatchNorm2d(out_chan)
        if self.with_act:
            self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x


class SE(nn.Module):
    def __init__(self, in_chan, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_chan, in_chan // r, kernel_size=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(in_chan // r, in_chan, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y


class MBConv(nn.Module):
    def __init__(
        self, in_chan, out_chan, kernel_size=3, stride=1, expansion=1, padding=1
    ):
        super().__init__()
        self.skip_conn = (in_chan == out_chan) and (stride == 1)
        expanded_chan = int(in_chan * expansion)

        if expansion > 1:
            self.conv = nn.Sequential(
                Conv2d(in_chan, expanded_chan, kernel_size=1, padding=0, stride=1),
                Conv2d(
                    expanded_chan,
                    expanded_chan,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=expanded_chan,
                ),
                SE(expanded_chan),
                Conv2d(
                    expanded_chan,
                    out_chan,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    with_act=False,
                ),
            )
        else:
            self.conv = nn.Sequential(
                Conv2d(
                    in_chan,
                    in_chan,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=in_chan,
                ),
                SE(in_chan),
                Conv2d(
                    in_chan,
                    out_chan,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    with_act=False,
                ),
            )

    def forward(self, x):
        res = x
        x = self.conv(x)
        if self.skip_conn:
            x = x + res
        return x


class Fused_MBConv(nn.Module):
    def __init__(
        self, in_chan, out_chan, kernel_size=3, stride=1, expansion=2, padding=1
    ):
        super().__init__()
        self.skip_conn = (in_chan == out_chan) and (stride == 1)
        expanded_chan = int(in_chan * expansion)

        if expansion > 1:
            self.conv = nn.Sequential(
                Conv2d(
                    in_chan,
                    expanded_chan,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                SE(expanded_chan),
                Conv2d(
                    expanded_chan,
                    out_chan,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                    with_act=False,
                ),
            )
        else:
            self.conv = nn.Sequential(
                Conv2d(in_chan, out_chan, kernel_size, stride=stride, padding=padding),
                SE(out_chan),
            )

    def forward(self, x):
        res = x
        x = self.conv(x)
        if self.skip_conn:
            x = x + res
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return norm_x * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, num_query_heads=8, num_kv_heads=2):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = in_dim // num_query_heads
        kv_dim = self.head_dim * self.num_kv_heads

        self.q_norm = RMSNorm(in_dim)
        self.k_norm = RMSNorm(kv_dim)
        self.to_q = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_dim, kv_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_dim, kv_dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, _, h, w = x.shape
        q_normed = self.q_norm(x)

        q = self.to_q(q_normed)
        k = self.k_norm(self.to_k(x))
        v = self.to_v(x)

        q = rearrange(q, "b (nh d) h w -> b nh (h w) d", nh=self.num_query_heads)
        k = rearrange(k, "b (nh d) h w -> b nh (h w) d", nh=self.num_kv_heads)
        v = rearrange(v, "b (nh d) h w -> b nh (h w) d", nh=self.num_kv_heads)

        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        sim = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(sim, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return out


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(dim * 2)
        hidden_dim = int(2 * hidden_dim / 3)
        multiple_of = 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.w3 = nn.Parameter(torch.ones(1, hidden_dim, 1, 1))
        self.w2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3)


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.mhsa = MultiHeadAttention(embed_dim, num_query_heads=num_heads)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = FFN(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.mhsa(self.norm1(x))
        x = res + self.dropout(x)

        res = x
        x = self.ffn(self.norm2(x))
        x = res + self.dropout(x)
        return x


# --- Layer Sequence Helpers ---
class MBConv_Layers(nn.Module):
    def __init__(self, in_chan, jumlah, expansion):
        super().__init__()
        self.layers = nn.Sequential(
            *[MBConv(in_chan, in_chan, expansion=expansion) for _ in range(jumlah)]
        )

    def forward(self, x):
        return self.layers(x)


class Encoder_Layers(nn.Module):
    def __init__(self, embed_dim, jumlah, num_heads):
        super().__init__()
        self.layers = nn.Sequential(
            *[Encoder(embed_dim, num_heads=num_heads) for _ in range(jumlah)]
        )

    def forward(self, x):
        return self.layers(x)


# --- Full Hybrid Model ---


class HybridEfficientNetViT(nn.Module):
    def __init__(
        self,
        num_classes=4,
        embed_dim=192,
        num_heads=8,
        dropout=0.2,
    ):
        super().__init__()

        self.stage1 = nn.Sequential(
            Conv2d(3, 24, stride=2),
            Fused_MBConv(24, 32, expansion=2),
        )

        self.patch_embed1 = MBConv(
            32, embed_dim, kernel_size=16, stride=8, padding=0, expansion=1
        )

        self.layers = nn.ModuleList(
            [
                Encoder_Layers(embed_dim, jumlah=3, num_heads=num_heads),
                MBConv_Layers(embed_dim, jumlah=3, expansion=2),
                # Repatch 1
                MBConv(
                    embed_dim,
                    embed_dim * 2,
                    kernel_size=5,
                    stride=2,
                    padding=0,
                    expansion=1,
                ),
                Encoder_Layers(embed_dim * 2, jumlah=6, num_heads=num_heads),
                MBConv_Layers(embed_dim * 2, jumlah=6, expansion=2),
                # Repatch 2
                MBConv(
                    embed_dim * 2,
                    embed_dim * 4,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                    expansion=1,
                ),
                Encoder_Layers(embed_dim * 4, jumlah=2, num_heads=num_heads),
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout, inplace=True), nn.Linear(embed_dim * 4, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.patch_embed1(x)

        for layer in self.layers:
            x = layer(x)

        x = rearrange(x, "b c h w -> b c (h w)")
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.head(x)

        return x


# --- Verification ---
if __name__ == "__main__":
    model = HybridEfficientNetViT(num_classes=4)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Corrected model parameter count: {params:,}")
    print(f"Target parameter count from notebook: 15,681,448")
