"""PyTorch port of the evaluation protocol used by Diffusion-TS.

The numbers reported by Yuan & Qiao (2024) come from the TimeGAN metric code
(TensorFlow 1) for the discriminative and predictive scores, and from their
own PyTorch code for Context-FID and the correlational score. This module
reproduces that protocol so the numbers here can be placed next to published ones.
It differs from `metrics.py` in four ways that change the numbers a lot:

  * Discriminative: a GRU with only int(F/2) hidden units (1-3 units here),
    2,000 iterations, 80/20 split of each set. `metrics.py` uses a GRU with
    max(8, 4F) units, which is a much stronger classifier.
  * Predictive: the GRU sees the first F-1 features and predicts the *last*
    feature one step ahead, through a sigmoid, for 5,000 iterations.
    `metrics.py` lets the GRU see every feature including the target's own
    past, which is a much easier task.
  * Correlational: lag-0 cross-correlation per sample on the lower triangle
    (diagonal included), averaged over samples, L1 distance divided by 10,
    on a random fifth of the data per repeat.
  * Context-FID: TS2Vec with 320-d output, depth 10, batch 8, lr 1e-3, 600
    iterations, max-pooled over the full series, refit on every repeat.

Protocol-level choices are also copied: data scaled to [0, 1]; the real
reference is the training split; as many generated samples as training
samples; 5 repeats; and "+-" is the half-width of a 95% confidence interval
from the t-distribution (`display_scores`), not a standard deviation.

GRU equations differ slightly between TensorFlow and PyTorch (placement of
the reset gate), and TS2Vec's cropping augmentation is simplified in
`ts2vec.py`. Both are minor, but the port is a port, not the original code.
"""
import numpy as np
import scipy.linalg
import scipy.stats
import torch
import torch.nn as nn

from .ts2vec import TS2VecEncoder, hierarchical_contrastive_loss


def to_unit_interval(x):
    """Our data is min-max scaled to [-1, 1] per feature; the benchmark uses [0, 1]."""
    x = x.numpy() if torch.is_tensor(x) else np.asarray(x)
    return ((x + 1.0) / 2.0).astype(np.float32)


def display_scores(values, repeats=5):
    """Mean and 95% CI half-width, exactly as Diffusion-TS's metric_utils."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    half = float(scipy.stats.sem(values) * scipy.stats.t.ppf(0.975, repeats - 1))
    return mean, half


class _GRUHead(nn.Module):
    def __init__(self, n_in, hidden, per_step):
        super().__init__()
        self.rnn = nn.GRU(n_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        self.per_step = per_step

    def forward(self, x):
        h, hn = self.rnn(x)
        return self.out(h) if self.per_step else self.out(hn[-1])


def discriminative_benchmark(ori, gen, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ori, gen = np.asarray(ori, np.float32), np.asarray(gen, np.float32)
    dim = ori.shape[-1]
    hidden = max(1, int(dim / 2))

    def divide(data):
        idx = rng.permutation(len(data))
        cut = int(len(data) * 0.8)
        return data[idx[:cut]], data[idx[cut:]]

    tr_x, te_x = divide(ori)
    tr_g, te_g = divide(gen)
    model = _GRUHead(dim, hidden, per_step=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(2000):
        xb = torch.from_numpy(tr_x[rng.permutation(len(tr_x))[:128]])
        gb = torch.from_numpy(tr_g[rng.permutation(len(tr_g))[:128]])
        lr_, lg = model(xb), model(gb)
        loss = bce(lr_, torch.ones_like(lr_)) + bce(lg, torch.zeros_like(lg))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = torch.sigmoid(model(torch.from_numpy(te_x))).squeeze(-1).numpy()
        pg = torch.sigmoid(model(torch.from_numpy(te_g))).squeeze(-1).numpy()
    y_true = np.concatenate([np.ones(len(pr)), np.zeros(len(pg))])
    y_pred = np.concatenate([pr, pg]) > 0.5
    acc = float((y_pred == y_true).mean())
    return abs(0.5 - acc)


def predictive_benchmark(ori, gen, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ori, gen = np.asarray(ori, np.float32), np.asarray(gen, np.float32)
    dim = ori.shape[-1]
    if dim < 2:
        raise ValueError("benchmark predictive score needs at least two features")
    hidden = max(1, int(dim / 2))
    model = _GRUHead(dim - 1, hidden, per_step=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5000):
        b = gen[rng.permutation(len(gen))[:128]]
        x = torch.from_numpy(b[:, :-1, : dim - 1])
        y = torch.from_numpy(b[:, 1:, dim - 1:])
        loss = (torch.sigmoid(model(x)) - y).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = torch.from_numpy(ori[:, :-1, : dim - 1])
        y = ori[:, 1:, dim - 1:]
        pred = torch.sigmoid(model(x)).numpy()
    return float(np.abs(pred - y).mean(axis=(1, 2)).mean())


def _cacf_lag0(x):
    """Lag-0 cross-correlation per sample over the lower-triangular feature
    pairs (diagonal included), after standardising each feature over all
    samples and time steps. x: (N, T, F) -> (N, n_pairs)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    F = x.shape[2]
    i, j = torch.tril_indices(F, F)
    x = (x - x.mean((0, 1), keepdim=True)) / x.std((0, 1), keepdim=True)
    return (x[..., i] * x[..., j]).mean(1)


def correlational_benchmark(ori, gen, seed=0, repeats=5):
    rng = np.random.default_rng(seed)
    ori, gen = np.asarray(ori, np.float32), np.asarray(gen, np.float32)
    size = int(len(ori) / repeats)
    scores = []
    for _ in range(repeats):
        ri = rng.choice(len(ori), size, replace=False)
        fi = rng.choice(len(gen), size, replace=False)
        real = _cacf_lag0(ori[ri]).mean(0)
        fake = _cacf_lag0(gen[fi]).mean(0)
        scores.append(float((fake - real).abs().sum() / 10.0))
    return scores


def _fid(a, b):
    mu1, s1 = a.mean(0), np.cov(a, rowvar=False)
    mu2, s2 = b.mean(0), np.cov(b, rowvar=False)
    covmean = scipy.linalg.sqrtm(s1.dot(s2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2.0 * covmean))


def context_fid_benchmark(ori, gen, seed=0, device="cpu"):
    """TS2Vec settings from Diffusion-TS's Context_FID: output 320, hidden 64,
    depth 10, batch 8, lr 1e-3, 600 iterations (their data exceeds the
    100k-element threshold TS2Vec uses to pick 200 vs 600)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ori, gen = np.asarray(ori, np.float32), np.asarray(gen, np.float32)
    N, T, F = ori.shape
    n_iters = 200 if ori.size <= 100_000 else 600
    enc = TS2VecEncoder(F, d_repr=320, d_hidden=64, depth=10, seq_len=T).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
    X = torch.from_numpy(ori).to(device)
    enc.train()
    for _ in range(n_iters):
        x = X[torch.from_numpy(rng.integers(0, N, 8)).to(device)]
        cl = int(rng.integers(2, T + 1))
        o1, o2 = int(rng.integers(0, T - cl + 1)), int(rng.integers(0, T - cl + 1))
        lo, hi = max(o1, o2), min(o1 + cl, o2 + cl)
        if hi - lo < 1:
            continue
        z1 = enc(x[:, o1:o1 + cl])[:, lo - o1:hi - o1]
        z2 = enc(x[:, o2:o2 + cl])[:, lo - o2:hi - o2]
        loss = hierarchical_contrastive_loss(z1, z2)
        opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        def encode(D):
            D = torch.from_numpy(D).to(device)
            return torch.cat([enc(D[k:k + 256], mask=False).max(1).values.cpu()
                              for k in range(0, len(D), 256)]).numpy()
        a, b = encode(ori), encode(gen)
    idx = rng.permutation(len(a))
    return _fid(a[idx], b[idx[: len(b)]] if len(b) >= len(a) else b)


def evaluate_benchmark(ori, gen, repeats=5, seed=0, verbose=True):
    """ori, gen: arrays in [0, 1], len(gen) >= len(ori). Returns
    {metric: {"mean", "ci95", "values"}} following Diffusion-TS."""
    gen = gen[: len(ori)]
    out = {}
    runs = {
        "context_fid": lambda s: context_fid_benchmark(ori, gen, seed=s),
        "discriminative": lambda s: discriminative_benchmark(ori, gen, seed=s),
        "predictive": lambda s: predictive_benchmark(ori, gen, seed=s),
    }
    for name, fn in runs.items():
        vals = []
        for r in range(repeats):
            vals.append(fn(seed + r))
            if verbose:
                print(f"      {name} repeat {r+1}/{repeats}: {vals[-1]:.4f}", flush=True)
        m, h = display_scores(vals, repeats)
        out[name] = {"mean": m, "ci95": h, "values": vals}
    corr = correlational_benchmark(ori, gen, seed=seed, repeats=repeats)
    m, h = display_scores(corr, repeats)
    out["correlation"] = {"mean": m, "ci95": h, "values": corr}
    return out
