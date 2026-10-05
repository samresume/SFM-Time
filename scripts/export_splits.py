"""Export the canonical train/validation/test splits.

Any method compared against the ones reported here must be trained on exactly
the same training split and scored against exactly the same test split, or a
difference in scores is confounded with a difference in data. This writes the
arrays `train_sfmtime.py` itself constructs, through the same calls in the same
order, so there is one definition of the data and every method reads it from
disk.

A competing method should write its samples as `<dataset>_<method>_data.npz`
with the same keys (real_train, real_val, real_test, generated), after which
`evaluate_saved.py --suffix <method>` scores it unchanged.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sfmtime.data import make_dataset, split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = {"sines": 64, "energy": 96, "stocks": 128, "ecg": 192}
OUT = os.path.join(ROOT, "data", "splits")

os.makedirs(OUT, exist_ok=True)
manifest = {}
for ds, T in PLAN.items():
    X, _, meta = make_dataset(ds, seed=0, seq_len=T)
    Xtr, Xva, Xte = split(X, seed=0)
    f = os.path.join(OUT, f"{ds}_T{T}.npz")
    np.savez_compressed(f, train=Xtr.numpy(), val=Xva.numpy(), test=Xte.numpy())
    manifest[ds] = {"seq_len": T, "n_features": int(meta["n_features"]),
                    "train": list(Xtr.shape), "val": list(Xva.shape),
                    "test": list(Xte.shape),
                    "file": f"data/splits/{ds}_T{T}.npz", "range": [-1.0, 1.0]}
    print(f"{ds:7s} train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  test {tuple(Xte.shape)}")
with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=1)
print("wrote", os.path.join(OUT, "manifest.json"))
