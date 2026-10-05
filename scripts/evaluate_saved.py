"""Recompute all metrics from saved run artefacts, with multi-seed repeats.

The discriminative score is a freshly-initialised GRU trained to separate
real from generated, so it carries real estimator variance: measured here at
std 0.02-0.11 across evaluator seeds on *fixed* data, i.e. a single-seed
number can land anywhere in a 6-18x range. Any ranking read off one seed is
therefore mostly reading noise. This script reports mean +- std over
`--repeat` seeds, which is what the benchmark protocol in this literature
specifies and what any table in the paper should carry.

Runs entirely off `results/*_data.npz`, so it needs no retraining.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sfmtime.metrics import (discriminative_score, predictive_score, correlation_score,
                           fit_context_encoder, context_fid, delta_distance,
                           spectral_distance, acf_distance)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["sines", "stocks", "energy", "ecg"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--ts2vec-steps", type=int, default=800)
    ap.add_argument("--cfid-repeat", type=int, default=3,
                    help="encoder fits behind Context-FID. Each one refits "
                         "TS2Vec, so this is the expensive dial; it was "
                         "hard-capped at 3 before, which silently bounded the "
                         "Context-FID error bars no matter what --repeat said.")
    ap.add_argument("--suffix", default="sfmtime", help="file tag: <ds>_<suffix>_data.npz")
    ap.add_argument("--datasets", default=",".join(ORDER))
    ap.add_argument("--dir", default=os.path.join("results", "run"),
                    help="run folder holding *_data.npz (default: results/run)")
    args = ap.parse_args()

    global RESULTS
    RESULTS = args.dir if os.path.isabs(args.dir) else os.path.join(ROOT, args.dir)

    out = {}
    for name in args.datasets.split(","):
        f = os.path.join(RESULTS, f"{name}_{args.suffix}_data.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        real = torch.from_numpy(d["real_test"]).float()
        fake = torch.from_numpy(d["generated"]).float()[: len(real)]
        train = torch.from_numpy(d["real_train"]).float()
        print(f"\n=== {name}  (real {tuple(real.shape)}, fake {tuple(fake.shape)}) ===",
              flush=True)

        disc = [discriminative_score(real, fake, steps=1200, device="cpu", seed=s)
                for s in range(args.repeat)]
        pred = [predictive_score(real, fake, steps=1200, device="cpu", seed=s)
                for s in range(args.repeat)]

        # Context-FID: refit the encoder per seed too, since the encoder is
        # itself fitted and contributes variance.
        cfid = []
        for s in range(min(args.repeat, args.cfid_repeat)):
            enc = fit_context_encoder(train, device="cpu", steps=args.ts2vec_steps, seed=s)
            cfid.append(context_fid(real, fake, enc, device="cpu"))

        corr = correlation_score(real, fake)

        res = {
            "discriminative": {"mean": float(np.mean(disc)), "std": float(np.std(disc)),
                               "values": [float(x) for x in disc]},
            "predictive": {"mean": float(np.mean(pred)), "std": float(np.std(pred)),
                           "values": [float(x) for x in pred]},
            "context_fid": {"mean": float(np.mean(cfid)), "std": float(np.std(cfid)),
                            "values": [float(x) for x in cfid]},
            "correlation": {"mean": float(corr), "std": 0.0},
            # deterministic given the samples: no seed dependence
            "delta_dist": {"mean": delta_distance(real, fake), "std": 0.0},
            "spectral_dist": {"mean": spectral_distance(real, fake), "std": 0.0},
            "acf_dist": {"mean": acf_distance(real, fake), "std": 0.0},
            "n_seeds": args.repeat,
        }
        out[name] = res
        print(f"  disc  {res['discriminative']['mean']:.4f} +- {res['discriminative']['std']:.4f}")
        print(f"  pred  {res['predictive']['mean']:.4f} +- {res['predictive']['std']:.4f}")
        print(f"  cfid  {res['context_fid']['mean']:.4f} +- {res['context_fid']['std']:.4f}")
        print(f"  corr  {res['correlation']['mean']:.4f}")
        print(f"  delta {res['delta_dist']['mean']:.4f}   spec {res['spectral_dist']['mean']:.4f}"
              f"   acf {res['acf_dist']['mean']:.4f}")

    with open(os.path.join(RESULTS, f"metrics_multiseed_{args.suffix}.json"), "w") as f:
        json.dump(out, f, indent=2)

    # markdown + csv tables
    md = ["# SFM-Time results (mean +- std over seeds)", "",
          "All metrics: lower is better.", "",
          "| Dataset | Discriminative | Predictive | Context-FID | Correlation |",
          "|---|---|---|---|---|"]
    csv = ["dataset,disc_mean,disc_std,pred_mean,pred_std,cfid_mean,cfid_std,corr"]
    for name in args.datasets.split(","):
        if name not in out:
            continue
        r = out[name]
        md.append(f"| {'ECG' if name == 'ecg' else name.capitalize()} | "
                  f"{r['discriminative']['mean']:.3f} ± {r['discriminative']['std']:.3f} | "
                  f"{r['predictive']['mean']:.3f} ± {r['predictive']['std']:.3f} | "
                  f"{r['context_fid']['mean']:.3f} ± {r['context_fid']['std']:.3f} | "
                  f"{r['correlation']['mean']:.3f} |")
        csv.append(f"{name},{r['discriminative']['mean']:.6f},{r['discriminative']['std']:.6f},"
                   f"{r['predictive']['mean']:.6f},{r['predictive']['std']:.6f},"
                   f"{r['context_fid']['mean']:.6f},{r['context_fid']['std']:.6f},"
                   f"{r['correlation']['mean']:.6f}")
    open(os.path.join(RESULTS, f"metrics_table_{args.suffix}.md"), "w").write("\n".join(md) + "\n")
    open(os.path.join(RESULTS, f"metrics_table_{args.suffix}.csv"), "w").write("\n".join(csv) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
