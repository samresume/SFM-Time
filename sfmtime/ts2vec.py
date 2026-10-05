"""A compact TS2Vec encoder, for Context-FID.

Context-FID (Jeha et al., ICLR 2022) is a Frechet distance computed in the
representation space of a TS2Vec encoder (Yue et al., AAAI 2022) fitted on
real data. It is the standard distributional metric in this literature, and
it is not interchangeable with a Frechet distance in some other embedding --
so rather than substitute a convenient autoencoder and label it
"Context-FID", this module implements TS2Vec itself.

The implementation follows the paper: an input projection, timestamp
masking, a dilated-convolution encoder with exponentially increasing
receptive field, and a hierarchical contrastive loss that combines an
instance-wise term (same timestamp, different series) and a temporal term
(same series, different timestamps) at every level of a max-pooling
hierarchy over time. Two views are produced by overlapping random crops.

One deviation, and it is a necessary one: TS2Vec's reference configuration
uses 10 dilated blocks (receptive field ~2^10), sized for sequences
thousands of steps long. The sequences here are 24-96 steps, so the depth is
scaled to the sequence length; stacking dilations far beyond T would add
parameters that can only ever see padding.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePadConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding=padding, dilation=dilation)
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(c_in, c_out, kernel_size, dilation)
        self.conv2 = SamePadConv(c_out, c_out, kernel_size, dilation)
        self.projector = nn.Conv1d(c_in, c_out, 1) if (c_in != c_out or final) else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class TS2VecEncoder(nn.Module):
    def __init__(self, n_features, d_repr=128, d_hidden=64, depth=None, seq_len=24):
        super().__init__()
        if depth is None:
            # receptive field ~ 2^depth; no point exceeding the sequence length
            depth = int(np.clip(np.ceil(np.log2(max(seq_len, 2))), 3, 10))
        self.input_fc = nn.Linear(n_features, d_hidden)
        channels = [d_hidden] * depth + [d_repr]
        self.net = nn.Sequential(*[
            ConvBlock(channels[i], channels[i + 1], kernel_size=3, dilation=2 ** i,
                      final=(i == len(channels) - 2))
            for i in range(len(channels) - 1)
        ])
        self.repr_dropout = nn.Dropout(0.1)

    def forward(self, x, mask=True):
        """x: (B, T, F) -> (B, T, d_repr)."""
        h = self.input_fc(x)                                  # (B, T, d_hidden)
        if mask and self.training:
            m = (torch.rand(h.shape[0], h.shape[1], 1, device=h.device) > 0.5).float()
            h = h * m
        h = h.transpose(1, 2)                                 # (B, d_hidden, T)
        h = self.repr_dropout(self.net(h))
        return h.transpose(1, 2)                              # (B, T, d_repr)

    @torch.no_grad()
    def encode(self, x):
        """Instance-level representation: max-pool the per-timestamp reprs."""
        self.eval()
        z = self.forward(x, mask=False)
        return z.max(dim=1).values


def instance_contrastive_loss(z1, z2):
    """Positives: same timestamp, same series across the two views.
    Negatives: same timestamp, other series in the batch."""
    B = z1.shape[0]
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)                 # (2B, T, D)
    z = z.transpose(0, 1)                          # (T, 2B, D)
    sim = torch.matmul(z, z.transpose(1, 2))       # (T, 2B, 2B)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(B, device=z1.device)
    return (logits[:, i, B - 1 + i].mean() + logits[:, B + i, i].mean()) / 2


def temporal_contrastive_loss(z1, z2):
    """Positives: same series, same timestamp across views.
    Negatives: same series, other timestamps."""
    T = z1.shape[1]
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)                 # (B, 2T, D)
    sim = torch.matmul(z, z.transpose(1, 2))       # (B, 2T, 2T)
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    t = torch.arange(T, device=z1.device)
    return (logits[:, t, T - 1 + t].mean() + logits[:, T + t, t].mean()) / 2


def hierarchical_contrastive_loss(z1, z2):
    loss, d = 0.0, 0
    while z1.shape[1] > 1:
        loss = loss + instance_contrastive_loss(z1, z2) + temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.shape[1] == 1:
        loss = loss + instance_contrastive_loss(z1, z2)
        d += 1
    return loss / max(d, 1)


def fit_ts2vec(X_real, d_repr=128, steps=1000, lr=1e-3, batch_size=64,
               device="cpu", seed=0, verbose=False):
    """Fit TS2Vec on REAL data only, then freeze. X_real: (N, T, F) tensor."""
    torch.manual_seed(seed)
    X = X_real.float().to(device)
    N, T, Fdim = X.shape
    enc = TS2VecEncoder(Fdim, d_repr=d_repr, seq_len=T).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr)
    enc.train()
    for step in range(steps):
        idx = torch.randint(0, N, (min(batch_size, N),), device=device)
        x = X[idx]
        # two overlapping random crops (TS2Vec's augmentation)
        if T >= 8:
            cl = np.random.randint(T // 2, T + 1)          # crop length
            o1 = np.random.randint(0, T - cl + 1)
            o2 = np.random.randint(0, T - cl + 1)
            x1, x2 = x[:, o1:o1 + cl], x[:, o2:o2 + cl]
            ov = min(o1 + cl, o2 + cl) - max(o1, o2)        # overlapping region
            if ov <= 0:
                x1 = x2 = x
                s1 = s2 = 0
                ov = T
            else:
                s1, s2 = max(o1, o2) - o1, max(o1, o2) - o2
        else:
            x1 = x2 = x; s1 = s2 = 0; ov = T
        z1 = enc(x1)[:, s1:s1 + ov]
        z2 = enc(x2)[:, s2:s2 + ov]
        loss = hierarchical_contrastive_loss(z1, z2)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (step + 1) % 250 == 0:
            print(f"    ts2vec step {step+1}/{steps}  loss={loss.item():.4f}", flush=True)
    return enc.eval().requires_grad_(False)
