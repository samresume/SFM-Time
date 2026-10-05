"""Train SFM-Time on one benchmark and write samples a scorer can read.

The defaults are the reported configuration: K=4 segments of depth 2, a trunk
of depth 4, width 96 (128 on Energy), 6,000 optimizer steps and NFE 80. The run
directory holds the checkpoint, the configuration, the health check and an
`.npz` whose `generated` and `real_test` arrays are what `evaluate_saved.py`
scores.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from sfmtime import SFMTimeConfig, ModelConfig, FlowConfig, TrainConfig, SFMTime
from sfmtime.train import SFMTimeTrainer
from sfmtime.data import make_dataset, split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ_LEN = {"sines": 64, "energy": 96, "stocks": 128, "ecg": 192}
D_MODEL = {"sines": 96, "energy": 128, "stocks": 96, "ecg": 96}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sines", choices=list(SEQ_LEN))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--segment-depth", type=int, default=2)
    ap.add_argument("--rep-depth", type=int, default=4)
    ap.add_argument("--nfe", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-gen", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--core", default="flow", choices=["flow", "ddpm"],
                    help="'ddpm' swaps the flow core for the diffusion ablation")
    ap.add_argument("--no-trunk", action="store_true")
    ap.add_argument("--per-sample-cond", action="store_true")
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "run"))
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    ds = a.dataset
    T, Fdim = SEQ_LEN[ds], None
    d_model = a.d_model or D_MODEL[ds]
    assert a.nfe % a.segments == 0, "NFE must divide evenly across segments"

    X, scaler, meta = make_dataset(ds, seed=a.seed, seq_len=T)
    Xtr, Xva, Xte = split(X, seed=a.seed)
    Fdim = int(meta["n_features"])

    model_cfg = ModelConfig(
        seq_len=T, n_features=Fdim, d_model=d_model,
        segments=a.segments, segment_depth=a.segment_depth,
        rep_depth=0 if a.no_trunk else a.rep_depth,
        cond_mode="add" if a.per_sample_cond else "adaln_token",
    )
    cfg = SFMTimeConfig(
        model=model_cfg,
        flow=FlowConfig(segments=a.segments, steps_per_segment=a.nfe // a.segments,
                        core=a.core),
        train=TrainConfig(steps=a.steps, batch_size=a.batch_size, lr=a.lr, seed=a.seed),
        name=a.tag or f"sfmtime_{ds}_d{d_model}_nfe{a.nfe}",
    )

    dev = pick_device()
    torch.manual_seed(a.seed)
    trainer = SFMTimeTrainer(cfg, Xtr, X_val=Xva, device=dev)
    info = trainer.summary()
    print(f"{ds}: T={T} F={Fdim} d={d_model} K={a.segments} NFE={a.nfe} on {trainer.device}")
    print(f"  {info['params_total']:,} parameters held, "
          f"{info['params_active_per_step']:,} evaluated per step "
          f"({info['blocks_active_per_step']} of {info['blocks_total']} blocks)")

    t0 = time.time()
    trainer.train(a.steps)
    train_s = time.time() - t0
    health = trainer.health_check(verbose=False)

    with torch.no_grad():                       # warm-up, excluded from timing
        trainer.generate(min(128, a.n_gen))
    t1 = time.time()
    gen = trainer.generate(a.n_gen)
    ms = 1000 * (time.time() - t1) / a.n_gen

    tag = a.tag or f"{ds}_sfmtime"
    d = os.path.join(a.dir, tag)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{ds}_sfmtime_data.npz"),
                        generated=gen, real_train=np.asarray(Xtr),
                        real_val=np.asarray(Xva), real_test=np.asarray(Xte))
    torch.save({"ema": trainer.ema.shadow.state_dict()}, os.path.join(d, "ckpt.pt"))
    rec = {"dataset": ds, "tag": tag, "config": cfg.to_dict(), "summary": info,
           "health_check": health, "nfe": trainer.last_nfe,
           "seq_len": T, "n_features": Fdim,
           "train_seconds": round(train_s, 1), "train_minutes": round(train_s / 60, 2),
           "sample_ms_per_sequence": round(ms, 3), "n_generated": a.n_gen,
           "history": trainer.history}
    with open(os.path.join(d, "run.json"), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"\ntrained in {train_s/60:.1f} min; sampling {ms:.1f} ms/sequence at NFE {trainer.last_nfe}")
    print("wrote", d)


if __name__ == "__main__":
    main()
