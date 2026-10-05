"""Trainer for SFM-Time: stage 1 (velocity + optional alignment), stage 2 (FRN).

Deliberately close to the backbone's trainer in shape -- AdamW, EMA, gradient clipping,
a warmup ramp -- so that a BFM-vs-the sequence baseline comparison differs in the model and
the schedule, not in the optimiser. BFM's own recipe (lr 1e-4, betas (0.9,
0.999), no weight decay, grad clip 1.0) is already the default in `TrainConfig`.
"""
import copy
import math
import time

import numpy as np
import torch

from .config import SFMTimeConfig
from .model import SFMTime
from .flow import SegmentedRectifiedFlow


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EMA:
    """Exponential moving average of the weights, with the usual warmup ramp so
    early steps are not dominated by the initialisation."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


class SFMTimeTrainer:
    def __init__(self, cfg: SFMTimeConfig, X_train, X_val=None, device=None):
        self.cfg = cfg
        self.device = pick_device(device or cfg.train.device)
        self.X = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
        self.X_val = (torch.as_tensor(np.asarray(X_val), dtype=torch.float32)
                      if X_val is not None else None)

        self.model = SFMTime(cfg.model).to(self.device)
        if cfg.flow.core == "ddpm":
            from .diffusion import SegmentedDiffusion
            self.flow = SegmentedDiffusion(cfg.flow, device=self.device,
                                           n_steps=cfg.flow.ddpm_steps).to(self.device)
        else:
            self.flow = SegmentedRectifiedFlow(cfg.flow, device=self.device)
        self.ema = EMA(self.model, cfg.train.ema_decay)

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr, betas=cfg.train.adam_betas,
            weight_decay=cfg.train.weight_decay)
        self.gen = torch.Generator(device="cpu").manual_seed(cfg.train.seed)
        self.history = []

        if cfg.train.batch_size < cfg.model.segments:
            raise ValueError(
                f"batch_size ({cfg.train.batch_size}) < segments "
                f"({cfg.model.segments}): some segments would get no samples "
                "in a step. Use a batch size that is a multiple of segments.")

    # ------------------------------------------------------------------ utils
    def _batch(self, bs):
        idx = torch.randint(0, self.X.shape[0], (bs,), generator=self.gen)
        return self.X[idx].to(self.device)

    def _lr_at(self, step):
        w = self.cfg.train.warmup_steps
        return self.cfg.train.lr * min(1.0, (step + 1) / max(w, 1))

    def summary(self):
        m = self.model
        return {
            "params_total": m.n_params(),
            "params_active_per_step": m.n_params_active(),
            "blocks_total": self.cfg.model.rep_depth
                            + self.cfg.model.segments * self.cfg.model.segment_depth,
            "blocks_active_per_step": m.n_blocks_active(),
            "nfe": self.cfg.model.segments * self.cfg.flow.steps_per_segment,
            "device": str(self.device),
        }

    # ------------------------------------------------------------------ train
    def train(self, steps=None, log=True):
        steps = steps or self.cfg.train.steps
        tc = self.cfg.train
        self.model.train()
        t0 = time.time()
        for step in range(steps):
            for g in self.opt.param_groups:
                g["lr"] = self._lr_at(step)
            x = self._batch(tc.batch_size)
            out = self.flow.loss(self.model, x, align_weight=tc.align_weight)
            self.opt.zero_grad(set_to_none=True)
            out["total"].backward()
            if tc.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)
            self.opt.step()
            self.ema.update(self.model, step)

            if log and (step % tc.log_every == 0 or step == steps - 1):
                rec = {"step": step, "loss": float(out["total"].detach()),
                       "denoise": float(out["denoise"]), "align": float(out["align"]),
                       "elapsed_s": round(time.time() - t0, 1)}
                self.history.append(rec)
                print(f"  step {step:>6}  loss {rec['loss']:.5f}  "
                      f"denoise {rec['denoise']:.5f}  ({rec['elapsed_s']}s)", flush=True)
        self.model.eval()
        return self

    def train_frn(self, steps=2000, lr=1e-4, log=True):
        """Stage 2: fit the Feature Residual Approximation with the rest frozen."""
        self.model.enable_frn()
        self.model.train()
        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)
        t0 = time.time()
        for step in range(steps):
            x = self._batch(self.cfg.train.batch_size)
            out = self.flow.frn_loss(self.model, x)
            opt.zero_grad(set_to_none=True)
            out["total"].backward()
            opt.step()
            if log and (step % self.cfg.train.log_every == 0 or step == steps - 1):
                print(f"  [frn] step {step:>6}  loss {float(out['total']):.6f}  "
                      f"({round(time.time()-t0,1)}s)", flush=True)
        self.model.eval()
        # keep the EMA copy structurally identical so `generate(ema=True)` works
        self.ema = EMA(self.model, self.cfg.train.ema_decay)
        return self

    # --------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, n, steps_per_segment=None, ema=True, batch=256, frn=False):
        model = self.ema.shadow if ema else self.model
        model.eval()
        T, F = self.cfg.model.seq_len, self.cfg.model.n_features
        out, total_nfe = [], None
        done = 0
        while done < n:
            b = min(batch, n - done)
            fn = self.flow.sample_frn if frn else self.flow.sample
            x, nfe = fn(model, (b, T, F), steps_per_segment=steps_per_segment,
                        device=self.device, return_nfe=True)
            out.append(x.cpu())
            total_nfe = nfe
            done += b
        self.last_nfe = total_nfe
        return torch.cat(out, dim=0).numpy()

    # ------------------------------------------------------------ diagnostics
    def health_check(self, verbose=True):
        """Three checks that catch the silent failures specific to this design.

        1. every segment's velocity stack actually receives gradient (a batch
           smaller than `segments`, or a bad split, silently starves one);
        2. the per-segment losses are not wildly unbalanced, which would mean
           one interval is doing all the work and the partition is mis-placed;
        3. samples are finite and not constant.
        """
        ok = {}
        self.model.train()
        x = self._batch(max(self.cfg.train.batch_size, self.cfg.model.segments))
        out = self.flow.loss(self.model, x, align_weight=self.cfg.train.align_weight)
        self.model.zero_grad(set_to_none=True)
        out["total"].backward()
        reached = []
        for seg, stack in enumerate(self.model.velocity_blocks):
            g = sum(float(p.grad.abs().sum()) for p in stack.parameters()
                    if p.grad is not None)
            reached.append(g > 0)
        self.model.zero_grad(set_to_none=True)
        ok["all_segments_get_gradient"] = bool(all(reached))
        ok["segments_without_gradient"] = [i for i, r in enumerate(reached) if not r]

        # per-segment loss balance
        self.model.eval()
        with torch.no_grad():
            sizes = self.flow.split_sizes(x.shape[0])
            t = self.flow.sample_segment_t(sizes, self.device)
            noise = self.flow.sample_prior(x.shape, self.device)
            x_t = self.flow.corrupt(x, t, noise)
            v, _ = self.model.forward_train(x_t, t, sizes)
            err = ((v - (noise - x)) ** 2).mean(dim=(1, 2))
            per_seg = [float(c.mean()) for c in torch.split(err, sizes) if c.numel()]
        ok["per_segment_mse"] = [round(p, 5) for p in per_seg]
        ok["segment_imbalance_ratio"] = round(max(per_seg) / max(min(per_seg), 1e-12), 2)

        s = self.generate(16, ema=False)
        ok["samples_finite"] = bool(np.isfinite(s).all())
        ok["samples_nonconstant"] = bool(float(np.std(s)) > 1e-6)
        ok["sample_std"] = round(float(np.std(s)), 4)

        if verbose:
            for k, v_ in ok.items():
                print(f"  {k}: {v_}")
        return ok
