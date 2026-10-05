"""The four benchmarks: Sines and ECG (synthetic), Stocks and Energy (real).

Matched to the benchmark specification used across the time-series generation
literature so that results are comparable: Sines from the closed-form
generator, ECG as synthetic PQRST-morphology sequences, Stocks and Energy as
sliding windows over the standard CSVs. Windows are cut with stride 1 and each
feature is scaled to [-1, 1] over all windows.

`scripts/export_splits.py` writes the resulting arrays to `data/splits/`, which
is what every reported number is computed from.
"""
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


# --------------------------------------------------------------- normalisation
class MinMaxScaler:
    """Per-feature min-max to [-1, 1] -- symmetric range matches the flow prior
    N(0, I) / diffusion prior better than an unbounded z-score would."""

    def __init__(self, feature_range=(-1.0, 1.0)):
        self.lo, self.hi = feature_range

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.min_, self.max_ = flat.min(axis=0), flat.max(axis=0)
        self.span_ = np.where((self.max_ - self.min_) > 1e-8, self.max_ - self.min_, 1.0)
        return self

    def transform(self, X):
        z = (X - self.min_) / self.span_
        return z * (self.hi - self.lo) + self.lo

    def inverse_transform(self, X):
        z = (np.asarray(X) - self.lo) / (self.hi - self.lo)
        return z * self.span_ + self.min_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------- sines
def make_sines(n=10_000, seq_len=24, n_features=4, seed=0,
              freq_range=(0.1, 0.2), phase_range=(0.0, 0.1)):
    """x_i(t) = 0.5*(sin(eta*t + theta) + 1), eta~U[0.1,0.2], theta~U[0,0.1]."""
    rng = np.random.default_rng(seed)
    t = np.arange(seq_len)[None, :, None]
    eta = rng.uniform(*freq_range, size=(n, 1, n_features))
    theta = rng.uniform(*phase_range, size=(n, 1, n_features))
    return (0.5 * (np.sin(eta * t + theta) + 1.0)).astype(np.float32)


# ---------------------------------------------------------------------- ecg
def _gauss_bump(t, center, width, amp):
    return amp * np.exp(-0.5 * ((t - center) / np.maximum(width, 1e-3)) ** 2)


def make_ecg(n=8_000, seq_len=96, n_features=2, seed=0, noise=0.02, beats=(1, 2),
             beat_period=48.0):
    """Synthetic PQRST-like beats: sums of Gaussian bumps with randomised
    amplitude/width/position per beat, independent gain/timing offset per lead.

    Beat *morphology* is fixed in absolute samples (`beat_period` sets the
    nominal samples-per-beat), and a longer window therefore contains
    proportionally more beats -- which is what a longer ECG recording is.
    An earlier version scaled the template by `seq_len/96`, so a 384-sample
    window held the same 1-2 beats stretched 4x wide, i.e. a 4x slower heart
    rate rather than a longer recording; at T=96 the two agree, so previous
    results are unaffected.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(seq_len, dtype=np.float32)
    X = np.zeros((n, seq_len, n_features), dtype=np.float32)
    scale = beat_period / 48.0                      # morphology, length-independent
    template = [  # (offset, width, amp) relative to the R peak
        (-13.0 * scale, 3.2 * scale, 0.16), (-3.2 * scale, 1.1 * scale, -0.12),
        (0.0, 1.2 * scale, 1.00), (3.4 * scale, 1.6 * scale, -0.28),
        (15.0 * scale, 6.0 * scale, 0.32),
    ]
    n_periods = max(1, int(round(seq_len / beat_period)))
    for i in range(n):
        nb = int(rng.choice(beats)) * n_periods     # beats scale with duration
        if nb == 1:
            centers = [rng.uniform(0.35, 0.65) * seq_len]
        else:
            gap = seq_len / (nb + 0.5) * rng.uniform(0.9, 1.1)
            first = rng.uniform(0.10, 0.35) * gap + 0.25 * gap
            centers = [first + k * gap for k in range(nb)
                       if first + k * gap < seq_len - 0.1 * gap]
        for f in range(n_features):
            gain = rng.uniform(0.75, 1.25)
            lead_shift = rng.normal(0.0, 0.8 * scale)
            sig = np.zeros(seq_len, dtype=np.float32)
            for c in centers:
                a_jit, w_jit = rng.uniform(0.75, 1.25), rng.uniform(0.80, 1.20)
                for off, w, a in template:
                    sig += _gauss_bump(t, c + off * w_jit + lead_shift, w * w_jit,
                                      a * a_jit * gain * rng.uniform(0.9, 1.1))
            sig += rng.normal(0.0, noise, size=seq_len).astype(np.float32)
            X[i, :, f] = sig
    return X


# -------------------------------------------------------------------- real csv
def windows_from_csv(path, seq_len=24, stride=1):
    """Sliding-window a real CSV (rows=time, cols=features) into (N, T, F)."""
    import pandas as pd
    arr = pd.read_csv(path).values.astype(np.float32)
    n_windows = (len(arr) - seq_len) // stride + 1
    if n_windows < 1:
        raise ValueError(f"{path}: {len(arr)} rows too short for seq_len={seq_len}")
    return np.stack([arr[i * stride: i * stride + seq_len] for i in range(n_windows)])


# -------------------------------------------------------------------- plumbing
def make_dataset(name, seed=0, normalise=True, data_dir=None, **kw):
    """-> (X_tensor in [-1,1], scaler, meta)."""
    data_dir = data_dir or os.path.join(os.path.dirname(__file__), "..", "data")
    if name == "sines":
        kw.setdefault("n", 10_000); kw.setdefault("seq_len", 24); kw.setdefault("n_features", 4)
        X = make_sines(seed=seed, **kw)
    elif name == "ecg":
        kw.setdefault("n", 8_000); kw.setdefault("seq_len", 96); kw.setdefault("n_features", 2)
        X = make_ecg(seed=seed, **kw)
    elif name == "stocks":
        kw.setdefault("seq_len", 24)
        X = windows_from_csv(os.path.join(data_dir, "stock_data.csv"), **kw)
    elif name == "energy":
        kw.setdefault("seq_len", 24)
        X = windows_from_csv(os.path.join(data_dir, "energy_data.csv"), **kw)
    else:
        raise ValueError(f"unknown dataset {name!r}")

    scaler = MinMaxScaler().fit(X) if normalise else None
    if scaler is not None:
        X = scaler.transform(X).astype(np.float32)
    meta = {"name": name, "seq_len": X.shape[1], "n_features": X.shape[2], "n": X.shape[0]}
    return torch.from_numpy(X), scaler, meta


def split(X, fracs=(0.8, 0.1, 0.1), seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)
    n = X.shape[0]
    n_tr, n_va = int(fracs[0] * n), int(fracs[1] * n)
    return X[idx[:n_tr]], X[idx[n_tr:n_tr + n_va]], X[idx[n_tr + n_va:]]


def loader(X, batch_size=128, shuffle=True, seed=0, drop_last=True):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=shuffle,
                      generator=g if shuffle else None, drop_last=drop_last)


class InfiniteLoader:
    def __init__(self, dl):
        self.dl = dl
        self.it = iter(dl)

    def next(self):
        try:
            (x,) = next(self.it)
        except StopIteration:
            self.it = iter(self.dl)
            (x,) = next(self.it)
        return x
