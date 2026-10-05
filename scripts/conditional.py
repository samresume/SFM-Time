r"""Conditional generation by masked replacement, on a trained SFM-Time model.

Nothing about the model changes and no gradient is taken: the observed entries
of a target sequence are carried to the current flow time along the model's own
straight path and written back after every Euler step. In the default
configuration that adds no network evaluations, so the NFE count is exactly the
unconditional one.

Two tasks, both defined only by the mask:

  imputation   entries are hidden at random, either independently per feature
               ("separate") or at the same timesteps across all features
               ("concurrent"), in geometric streaks of mean length `lm`
  forecasting  the final `horizon` steps are hidden for every feature

The mask construction, the reference completions and the scoring follow the
protocol of the comparison studies in this project unchanged, so the numbers are
on the same scale as published imputation and forecasting results: streaks
rather than isolated points (isolated points are trivially interpolable), and
the K-sample mean rather than a single draw as the quantity compared against the
deterministic references.

"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from sfmtime import SFMTimeConfig, ModelConfig, FlowConfig, TrainConfig, SFMTime
from sfmtime.flow import SegmentedRectifiedFlow

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = ["sines", "stocks", "energy", "ecg"]


# ---- masks ---------------------------------------------------------------
def geom_streaks(T, ratio, lm, rng):
    """A boolean run of length T that is False for ~ratio of its entries.

    Markov chain with mean masked-run length `lm`, so hidden entries arrive in
    contiguous stretches rather than as isolated points.
    """
    keep = np.ones(T, dtype=bool)
    p_m = 1.0 / lm                                  # leave a masked run
    p_u = p_m * ratio / (1.0 - ratio + 1e-12)       # enter a masked run
    masked = rng.random() < ratio
    for i in range(T):
        keep[i] = not masked
        masked = (rng.random() > p_m) if masked else (rng.random() < p_u)
    return keep


def impute_mask(shape, ratio, style, lm, rng):
    """True where observed. `shape` is (N, T, F)."""
    N, T, F = shape
    m = np.ones(shape, dtype=bool)
    for i in range(N):
        if style == "concurrent":                   # same timesteps for all features
            m[i] = geom_streaks(T, ratio, lm, rng)[:, None]
        else:                                       # each feature independent
            for f in range(F):
                m[i, :, f] = geom_streaks(T, ratio, lm, rng)
    return m


def forecast_mask(shape, horizon):
    N, T, F = shape
    m = np.ones(shape, dtype=bool)
    m[:, T - horizon:, :] = False
    return m


# ---- free reference completions -----------------------------------------
def linear_fill(x, keep):
    """Linear interpolation through the observed entries, per feature.

    Not a competitor; a scale. Without it an imputation MSE is a number with no
    reference, since this study runs our method alone.
    """
    out = x.copy()
    N, T, F = x.shape
    grid = np.arange(T)
    for i in range(N):
        for f in range(F):
            k = keep[i, :, f]
            if k.sum() == 0:
                out[i, :, f] = 0.0
            elif k.sum() == T:
                continue
            else:
                out[i, :, f] = np.interp(grid, grid[k], x[i, k, f])
    return out


def last_value_fill(x, horizon):
    out = x.copy()
    out[:, x.shape[1] - horizon:, :] = x[:, x.shape[1] - horizon - 1, :][:, None, :]
    return out


# ---- model ---------------------------------------------------------------
def load_run(ds, res, tag=None):
    """Load a run directory written by `scripts/train_sfmtime.py`."""
    d = os.path.join(res, tag or f"{ds}_sfmtime")
    run = json.load(open(os.path.join(d, "run.json")))
    c = run["config"]
    cfg = SFMTimeConfig(model=ModelConfig(**c["model"]), flow=FlowConfig(**c["flow"]),
                        train=TrainConfig(**c["train"]), name=c.get("name", ds))
    dev = (torch.device("cuda") if torch.cuda.is_available() else
           torch.device("mps") if torch.backends.mps.is_available() else
           torch.device("cpu"))
    model = SFMTime(cfg.model).to(dev)
    model.load_state_dict(torch.load(os.path.join(d, "ckpt.pt"), map_location=dev)["ema"])
    model.eval()
    flow = SegmentedRectifiedFlow(cfg.flow, device=dev)
    real = np.load(os.path.join(d, f"{ds}_sfmtime_data.npz"))["real_test"]
    return model, flow, cfg, dev, real


def complete(model, flow, cfg, dev, target, keep, steps, batch=250, seed=0):
    """Fill the hidden entries of `target`; `keep` is True where observed."""
    tgt = torch.as_tensor(target, dtype=torch.float32)
    msk = torch.as_tensor(keep)
    outs, nfe = [], 0
    g = torch.Generator(device=dev if dev.type != "mps" else "cpu")
    for s in range(0, len(tgt), batch):
        e = min(s + batch, len(tgt))
        g.manual_seed(seed + s)
        torch.manual_seed(seed + s)             # the fresh-noise writes use the
        o, nfe = flow.sample(
            model, (e - s, cfg.model.seq_len, cfg.model.n_features),
            steps_per_segment=steps, device=dev, return_nfe=True,
            cond_target=tgt[s:e], cond_mask=msk[s:e])
        outs.append(o.cpu())
    return torch.cat(outs).numpy(), nfe


def complete_k(model, flow, cfg, dev, target, keep, steps, K, seed=0):
    out = [complete(model, flow, cfg, dev, target, keep, steps, seed=seed + 1000 * k)
           for k in range(K)]
    return np.stack([o for o, _ in out]), out[0][1]


def scores(preds, real, keep):
    """Score K completions of the same input on the hidden entries.

      mse       one draw against the truth. A calibrated generative model does
                badly on this by construction, since squared error is minimised
                by the conditional mean and a sample is not it.
      mse_mean  the K-sample mean against the truth, which estimates that
                conditional mean and is what the references already are. This
                is the comparable number.
      crps      a proper scoring rule, E|X-y| - 0.5 E|X-X'|, which rewards a
                well-spread predictive distribution instead of punishing it.
    """
    hid = ~keep
    K = preds.shape[0]
    d1 = preds[0] - real
    dm = preds.mean(0) - real
    y = real[None][:, hid]
    X = preds[:, hid]
    term1 = np.abs(X - y).mean()
    if K > 1:
        Xs = np.sort(X, axis=0)
        w = (2 * np.arange(1, K + 1) - K - 1)
        term2 = (w[:, None] * Xs).sum(0).mean() * (2.0 / (K * (K - 1)))
    else:
        term2 = 0.0
    return {"mse": float((d1[hid] ** 2).mean()),
            "mae": float(np.abs(d1[hid]).mean()),
            "mse_mean": float((dm[hid] ** 2).mean()),
            "mae_mean": float(np.abs(dm[hid]).mean()),
            "crps": float(term1 - 0.5 * term2),
            "n_samples": int(K),
            "obs_max_abs": float(np.abs(d1[keep]).max()) if keep.any() else 0.0,
            "n_hidden": int(hid.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sines")
    ap.add_argument("--res", default=os.path.join(ROOT, "results", "run"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "conditional"))
    ap.add_argument("--ratios", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--styles", default="separate")
    ap.add_argument("--horizons", default="0.5")
    ap.add_argument("--steps", type=int, default=20, help="Euler steps per segment")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--lm", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tasks", default="impute,forecast")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.save:
        os.makedirs(a.save, exist_ok=True)
    tasks = a.tasks.split(",")

    rows = []
    for ds in a.datasets.split(","):
        model, flow, cfg, dev, real = load_run(ds, a.res)
        real = real[:a.n]
        T = real.shape[1]
        var = float(real.var())
        print(f"\n### {ds}  real_test={real.shape}  var={var:.4f}  "
              f"steps/seg={a.steps}", flush=True)

        if "impute" in tasks:
            for style in a.styles.split(","):
                for r in (float(x) for x in a.ratios.split(",")):
                    rng = np.random.default_rng(a.seed)
                    keep = impute_mask(real.shape, r, style, a.lm, rng)
                    t0 = time.time()
                    pred, nfe = complete_k(model, flow, cfg, dev, real * keep, keep,
                                           a.steps, a.k, seed=a.seed)
                    s = scores(pred, real, keep)
                    s |= {"dataset": ds, "task": "impute", "style": style, "ratio": r,
                          "steps": a.steps, "nfe": nfe,
                          "data_var": var,
                          "hidden_frac": float((~keep).mean()), "sec": time.time() - t0}
                    s["ref_mse"] = float((((linear_fill(real * keep, keep) - real)[~keep]) ** 2).mean())
                    rows.append(s)
                    if a.save:
                        np.savez_compressed(
                            os.path.join(a.save, f"{ds}_impute_{style}_{r:g}.npz"),
                            preds=pred.astype(np.float32),
                            real=real.astype(np.float32), keep=keep)
                    print(f"  impute {style:10s} hid={r:.1f}  mse1={s['mse']:.5f} "
                          f"mseK={s['mse_mean']:.5f} crps={s['crps']:.5f} "
                          f"(linear {s['ref_mse']:.5f})  obs={s['obs_max_abs']:.0e} "
                          f"nfe={nfe} ({s['sec']:.0f}s)", flush=True)

        if "forecast" in tasks:
            for frac in (float(x) for x in a.horizons.split(",")):
                h = max(1, int(round(frac * T)))
                keep = forecast_mask(real.shape, h)
                t0 = time.time()
                pred, nfe = complete_k(model, flow, cfg, dev, real * keep, keep,
                                       a.steps, a.k, seed=a.seed)
                s = scores(pred, real, keep)
                s |= {"dataset": ds, "task": "forecast", "style": "-", "ratio": frac,
                      "horizon": h, "steps": a.steps, "nfe": nfe,
                      "data_var": var,
                      "hidden_frac": float((~keep).mean()), "sec": time.time() - t0}
                s["ref_mse"] = float((((last_value_fill(real, h) - real)[~keep]) ** 2).mean())
                rows.append(s)
                if a.save:
                    np.savez_compressed(
                        os.path.join(a.save, f"{ds}_forecast_{frac:g}.npz"),
                        preds=pred.astype(np.float32),
                        real=real.astype(np.float32), keep=keep)
                print(f"  forecast h={h:3d} ({frac:.3f}T)  mse1={s['mse']:.5f} "
                      f"mseK={s['mse_mean']:.5f} crps={s['crps']:.5f} "
                      f"(last-value {s['ref_mse']:.5f})  obs={s['obs_max_abs']:.0e} "
                      f"nfe={nfe} ({s['sec']:.0f}s)", flush=True)

    tag = a.tag or "main"
    p = os.path.join(a.out, f"conditional_{tag}.json")
    json.dump(rows, open(p, "w"), indent=2)
    print(f"\nwrote {p}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
