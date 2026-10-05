"""Backbone blocks: the backbone's bidirectional RoPE attention + BFM's token-wise adaLN.

This is where the two lineages are joined. BFM's `TransformerBlock` and the backbone's
`DiTBlock` are both pre-LN, bidirectional, adaLN-modulated transformer blocks;
they differ in two respects that matter for the port:

  * *position encoding.* BFM uses a 2-D vision RoPE indexed by a patch grid
    (h, w). Time series have one axis, so that is replaced by the 1-D RoPE over
    timesteps from `sfmtime/modules.py`, which encodes relative lag directly in
    the attention dot product.
  * *conditioning granularity.* the backbone conditions on a per-sample vector c of
    shape (B, d). BFM conditions the velocity blocks on `c_repre`, a *per-token*
    tensor of shape (B, N, 2d) formed by concatenating the broadcast time/class
    embedding with the representation trunk's output at that token. That
    per-token conditioning is the mechanism by which Semantic Feature Guidance
    reaches the velocity field, so it cannot be collapsed to a per-sample
    vector without removing the thing being ported.

`TokenAdaLNBlock` therefore takes conditioning of shape (B, N, d_c) and emits
per-token shift/scale/gate triples. `global_adaln` reproduces BFM's
`adaln_type='lora'`: a modulation computed once outside the block stack is added
to each block's own modulation, so blocks start from a shared baseline instead of
learning one independently. Zero-initialising each block's final modulation
layer makes every block an identity map at initialisation, as in BFM and DiT.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0):
    """Sinusoidal embedding of flow time. t: (B,) -> (B, dim). As in the backbone."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def rope_cache(seq_len, head_dim, device, base=10_000.0):
    """cos/sin tables for 1-D RoPE over time, each (1, 1, T, head_dim/2)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos()[None, None], freqs.sin()[None, None]


def apply_rope(x, cos, sin):
    """x: (B, h, T, dh). Rotate channel pairs by the per-position angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    return torch.stack([xr1, xr2], dim=-1).flatten(-2)


class SelfAttention(nn.Module):
    """Bidirectional self-attention over timesteps, with optional RoPE and lag bias.

    Carried over from the backbone unchanged. `causal` is exposed but defaults to False
    for both lineages: BFM's blocks are bidirectional, and the backbone's own ablation
    found bidirectional attention to be the single most load-bearing
    architectural choice for these benchmarks.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, causal=False, use_rope=True,
                 use_lag_bias=False, seq_len=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.dh = d_model // n_heads
        self.causal = causal
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.use_lag_bias = use_lag_bias
        if use_lag_bias:
            assert seq_len is not None, "lag bias needs seq_len"
            assert not causal, "lag bias and causal masking are not combined here"
            self.lag_bias = nn.Parameter(torch.zeros(n_heads, 2 * seq_len - 1))
            idx = torch.arange(seq_len)
            self.register_buffer("lag_index", (idx[:, None] - idx[None, :]) + seq_len - 1,
                                 persistent=False)

    def forward(self, x, rope=None):
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_rope and rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attn_mask = None
        if self.use_lag_bias:
            attn_mask = self.lag_bias[:, self.lag_index].unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        return self.proj(out.transpose(1, 2).reshape(B, T, d))


class FeedForward(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mult * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


def _modulate(x, shift, scale):
    """Per-token modulation. shift/scale are (B, N, d) -- already per-token, so
    unlike the backbone's `modulate` there is no broadcast unsqueeze."""
    return x * (1 + scale) + shift


class TokenAdaLNBlock(nn.Module):
    """Pre-LN block with BFM's per-token adaLN, over the backbone's attention.

    `c` is (B, N, d_c): for the velocity blocks d_c = 2*d_model (time embedding
    concatenated with the representation feature); for the representation trunk
    d_c = d_model (the time embedding broadcast over tokens). `global_adaln` is
    either 0.0 or a (B, N, 6*d_model) tensor added to this block's modulation.
    """

    def __init__(self, d_model, n_heads, d_cond, mult=4, dropout=0.0, causal=False,
                 use_rope=True, use_lag_bias=False, seq_len=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(d_model, n_heads, dropout, causal, use_rope,
                                  use_lag_bias, seq_len)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(d_model, mult, dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_cond, 6 * d_model)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, rope=None, global_adaln=0.0):
        mod = self.adaLN_modulation(c) + global_adaln
        s1, sc1, g1, s2, sc2, g2 = mod.chunk(6, dim=-1)
        x = x + g1 * self.attn(_modulate(self.norm1(x), s1, sc1), rope=rope)
        x = x + g2 * self.ff(_modulate(self.norm2(x), s2, sc2))
        return x


class PlainBlock(nn.Module):
    """Pre-LN block with conditioning added once at the input (the backbone's default).

    Kept so that `cond_mode='add'` reproduces a plain pre-norm block exactly, which
    makes "BFM's segmentation with a plain pre-norm block" a runnable arm. Note that
    in this mode the representation trunk's output can only reach the velocity
    blocks additively, which is a weaker form of Semantic Feature Guidance than
    BFM's.
    """

    def __init__(self, d_model, n_heads, d_cond=None, mult=4, dropout=0.0, causal=False,
                 use_rope=True, use_lag_bias=False, seq_len=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout, causal, use_rope,
                                  use_lag_bias, seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, mult, dropout)
        self.cond_proj = nn.Linear(d_cond, d_model) if d_cond else None

    def forward(self, x, c, rope=None, global_adaln=0.0):
        if self.cond_proj is not None and c is not None:
            x = x + self.cond_proj(c)
        x = x + self.attn(self.norm1(x), rope=rope)
        x = x + self.ff(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """LN + per-token adaLN + projection back to (patched) feature space.

    Zero-initialised output projection, as in BFM/DiT: the model starts by
    predicting zero velocity everywhere, so early training does not fight a
    random initial field. `token_adaln=False` goes with `cond_mode='add'`.
    """

    def __init__(self, d_model, d_out, d_cond, token_adaln=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=not token_adaln, eps=1e-6)
        self.token_adaln = token_adaln
        if token_adaln:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 2 * d_model))
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.linear = nn.Linear(d_model, d_out)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        if self.token_adaln:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
            x = _modulate(self.norm(x), shift, scale)
        else:
            x = self.norm(x)
        return self.linear(x)


class LabelEmbedder(nn.Module):
    """Optional class conditioning with dropout for CFG. Unused by default."""

    def __init__(self, n_classes, d_model, dropout_prob):
        super().__init__()
        self.embedding_table = nn.Embedding(n_classes + 1, d_model)
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def forward(self, y, train):
        if train and self.dropout_prob > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
            y = torch.where(drop, torch.full_like(y, self.n_classes), y)
        return self.embedding_table(y)
