"""Evaluation metrics.

Two kinds, and the difference matters when reading small gaps.

*Learned* -- `discriminative_score`, `predictive_score`, and Context-FID --
train a network on the samples and therefore carry estimator variance even when
the samples are held fixed. The discriminative score's spread across evaluator
seeds on fixed data was measured here at 0.02-0.11, which exceeds many
published gaps between methods on these benchmarks; report them over several
seeds.

*Deterministic* -- `correlation_score`, `spectral_distance`, `delta_distance`,
`acf_distance` -- are closed-form functions of the samples and carry no
evaluator variance, so differences in them are exact.

`frechet_score` is a Frechet distance in a small GRU-autoencoder embedding. It
is a legitimate distributional metric but it is *not* Context-FID and the two
are not numerically comparable; Context-FID is computed from the TS2Vec
encoder in `ts2vec.py`. `benchmark_metrics.py` implements the stricter protocol
used for the published comparison.
"""
import numpy as np
import torch
import torch.nn as nn



def _to_t(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.float().to(device)


class _GRU(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, seq_out=False):
        super().__init__()
        self.rnn = nn.GRU(n_in, n_hidden, batch_first=True)
        self.head = nn.Linear(n_hidden, n_out)
        self.seq_out = seq_out

    def forward(self, x):
        h, hn = self.rnn(x)
        return self.head(h) if self.seq_out else self.head(hn[-1])


def _train(model, batches, steps, lr, loss_fn):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        loss = loss_fn(model, batches())
        opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def discriminative_score(real, fake, hidden=None, steps=1200, lr=1e-3,
                         batch_size=128, device="cpu", seed=0, repeat=1):
    out = []
    for r in range(repeat):
        g = torch.Generator().manual_seed(seed + r)
        torch.manual_seed(seed + r)
        R, Fk = _to_t(real, device), _to_t(fake, device)
        n = min(len(R), len(Fk))
        R = R[torch.randperm(len(R), generator=g)[:n]]
        Fk = Fk[torch.randperm(len(Fk), generator=g)[:n]]
        cut = int(0.8 * n)
        Rtr, Rte, Ftr, Fte = R[:cut], R[cut:], Fk[:cut], Fk[cut:]
        h = hidden or max(8, R.shape[-1] * 4)
        model = _GRU(R.shape[-1], h, 1).to(device)

        def batches():
            i = torch.randint(0, len(Rtr), (batch_size // 2,))
            j = torch.randint(0, len(Ftr), (batch_size // 2,))
            x = torch.cat([Rtr[i], Ftr[j]])
            y = torch.cat([torch.ones(len(i)), torch.zeros(len(j))]).to(device)
            return x, y

        def loss_fn(m, b):
            x, y = b
            return nn.functional.binary_cross_entropy_with_logits(m(x).squeeze(-1), y)

        _train(model, batches, steps, lr, loss_fn)
        with torch.no_grad():
            pr = (model(Rte).squeeze(-1) > 0).float()
            pf = (model(Fte).squeeze(-1) > 0).float()
            acc = (pr.sum() + (1 - pf).sum()).item() / (len(Rte) + len(Fte))
        out.append(abs(acc - 0.5))
    return (float(np.mean(out)), float(np.std(out))) if repeat > 1 else float(out[0])


def predictive_score(real, fake, hidden=None, steps=1200, lr=1e-3,
                     batch_size=128, device="cpu", seed=0, repeat=1):
    out = []
    for r in range(repeat):
        torch.manual_seed(seed + r)
        R, Fk = _to_t(real, device), _to_t(fake, device)
        Fdim = R.shape[-1]
        h = hidden or max(8, Fdim * 4)
        model = _GRU(Fdim, h, Fdim, seq_out=True).to(device)

        def batches():
            i = torch.randint(0, len(Fk), (batch_size,))
            return Fk[i]

        def loss_fn(m, x):
            return nn.functional.l1_loss(m(x[:, :-1]), x[:, 1:])

        _train(model, batches, steps, lr, loss_fn)
        with torch.no_grad():
            mae = nn.functional.l1_loss(model(R[:, :-1]), R[:, 1:]).item()
        out.append(mae)
    return (float(np.mean(out)), float(np.std(out))) if repeat > 1 else float(out[0])


class _SeqAE(nn.Module):
    def __init__(self, n_features, d=64):
        super().__init__()
        self.enc = nn.GRU(n_features, d, batch_first=True)
        self.dec = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, n_features)

    def embed(self, x):
        _, hn = self.enc(x)
        return hn[-1]

    def forward(self, x):
        z = self.embed(x)
        h, _ = self.dec(z.unsqueeze(1).repeat(1, x.shape[1], 1))
        return self.out(h)


def fit_embedder(real, d=64, steps=1500, lr=1e-3, batch_size=128, device="cpu", seed=0):
    torch.manual_seed(seed)
    R = _to_t(real, device)
    ae = _SeqAE(R.shape[-1], d).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    for _ in range(steps):
        x = R[torch.randint(0, len(R), (batch_size,))]
        loss = nn.functional.mse_loss(ae(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
    return ae.eval().requires_grad_(False)


def _frechet(mu1, s1, mu2, s2, eps=1e-6):
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def frechet_score(real, fake, embedder, device="cpu"):
    with torch.no_grad():
        a = embedder.embed(_to_t(real, device)).cpu().numpy()
        b = embedder.embed(_to_t(fake, device)).cpu().numpy()
    return _frechet(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))


def acf(X, n_lags=None):
    X = X.numpy() if torch.is_tensor(X) else np.asarray(X)
    T = X.shape[1]
    n_lags = n_lags or min(T - 1, 12)
    Z = X - X.mean(axis=1, keepdims=True)
    var = (Z ** 2).mean(axis=1) + 1e-8
    return np.stack([(Z[:, : T - k] * Z[:, k:]).mean(axis=1) / var
                     for k in range(1, n_lags + 1)], axis=1).mean(axis=0)


def acf_distance(real, fake, n_lags=None):
    return float(np.abs(acf(real, n_lags) - acf(fake, n_lags)).mean())


def marginal_distance(real, fake, bins=50):
    R = real.numpy() if torch.is_tensor(real) else np.asarray(real)
    Fk = fake.numpy() if torch.is_tensor(fake) else np.asarray(fake)
    ds = []
    for f in range(R.shape[-1]):
        lo, hi = min(R[..., f].min(), Fk[..., f].min()), max(R[..., f].max(), Fk[..., f].max())
        hr, _ = np.histogram(R[..., f], bins=bins, range=(lo, hi))
        hf, _ = np.histogram(Fk[..., f], bins=bins, range=(lo, hi))
        ds.append(0.5 * np.abs(hr / hr.sum() - hf / hf.sum()).sum())
    return float(np.mean(ds))


def diversity(X):
    X = X if torch.is_tensor(X) else torch.from_numpy(np.asarray(X))
    Z = X.flatten(1)[:512].float()
    return float(torch.cdist(Z, Z).sum() / (len(Z) * (len(Z) - 1)))


# --------------------------------------------------------- dynamics metrics
def delta_distance(real, fake, max_n=4000):
    """1-Wasserstein distance between the distributions of step-to-step
    changes, averaged over features. This measures transition behaviour
    directly: a generator that matches the marginal distribution but moves
    too smoothly, or too erratically, scores badly here even when its
    marginals look right. Lower is better."""
    from scipy.stats import wasserstein_distance
    R = real.numpy() if torch.is_tensor(real) else np.asarray(real)
    F_ = fake.numpy() if torch.is_tensor(fake) else np.asarray(fake)
    dr, df = np.diff(R[:max_n], axis=1), np.diff(F_[:max_n], axis=1)
    return float(np.mean([wasserstein_distance(dr[..., j].ravel(), df[..., j].ravel())
                          for j in range(R.shape[-1])]))


def spectral_distance(real, fake, eps=1e-8):
    """Mean absolute difference between the average log power spectra,
    over frequencies and features. Lower is better."""
    R = real.numpy() if torch.is_tensor(real) else np.asarray(real)
    F_ = fake.numpy() if torch.is_tensor(fake) else np.asarray(fake)
    pr = np.log(np.abs(np.fft.rfft(R, axis=1)) ** 2 + eps).mean(0)
    pf = np.log(np.abs(np.fft.rfft(F_, axis=1)) ** 2 + eps).mean(0)
    return float(np.abs(pr - pf).mean())


# ------------------------------------------------------- correlation score
def _feature_corr(X):
    """Cross-feature correlation matrix over all (N*T) observations."""
    X = X.numpy() if torch.is_tensor(X) else np.asarray(X)
    Z = X.reshape(-1, X.shape[-1])
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    return (Z.T @ Z) / len(Z)


def correlation_score(real, fake):
    """L1 distance between real and generated cross-feature correlation
    matrices (Ni et al. / Diffusion-TS convention). Lower is better.

    A univariate dataset has no cross-feature structure to match, so this is
    0 by definition there -- reported as such rather than silently omitted."""
    cr, cf = _feature_corr(real), _feature_corr(fake)
    if cr.shape[0] < 2:
        return 0.0
    return float(np.abs(cr - cf).sum())


# ------------------------------------------------------------ Context-FID
def fit_context_encoder(real, device="cpu", steps=1000, seed=0, verbose=False):
    """Fit the TS2Vec encoder Context-FID is defined against, on REAL data."""
    from .ts2vec import fit_ts2vec
    return fit_ts2vec(_to_t(real, device), steps=steps, device=device, seed=seed,
                      verbose=verbose)


def context_fid(real, fake, encoder, device="cpu", batch=512):
    """Frechet distance between TS2Vec representations of real and generated
    sequences (Paul et al., 2022). Lower is better."""
    def embed(X):
        X = _to_t(X, device)
        outs = [encoder.encode(X[i:i + batch]).cpu().numpy()
                for i in range(0, len(X), batch)]
        return np.concatenate(outs, axis=0)
    a, b = embed(real), embed(fake)
    return _frechet(a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))


def evaluate_all(real, fake, embedder=None, context_encoder=None, device="cpu",
                 seed=0, repeat=1, quick=False):
    """The four headline metrics (discriminative, predictive, Context-FID,
    correlation) plus cheap structural diagnostics. Lower is better on all."""
    steps = 400 if quick else 1200
    res = {
        "discriminative": discriminative_score(real, fake, steps=steps, device=device,
                                               seed=seed, repeat=repeat),
        "predictive": predictive_score(real, fake, steps=steps, device=device,
                                       seed=seed, repeat=repeat),
        "correlation": correlation_score(real, fake),
        "delta_dist": delta_distance(real, fake),
        "spectral_dist": spectral_distance(real, fake),
        "acf_dist": acf_distance(real, fake),
        "marginal_dist": marginal_distance(real, fake),
        "diversity_fake": diversity(fake),
        "diversity_real": diversity(real),
    }
    if embedder is not None:
        res["frechet_ae"] = frechet_score(real, fake, embedder, device)
    if context_encoder is not None:
        res["context_fid"] = context_fid(real, fake, context_encoder, device)
    return res
