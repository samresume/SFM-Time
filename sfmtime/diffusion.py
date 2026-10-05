"""Segmented Gaussian diffusion: the ablation arm that swaps only the core.

Everything about \\method{} is kept -- the K velocity stacks, the shared trunk,
the per-token conditioning, the stratified minibatch split -- and only the
generative process is replaced. Segments now partition the diffusion timestep
axis rather than flow time, which is the direct analogue: the model still sees a
normalised t in [0,1] with t=1 the noisy end, and the same `sigmas` grid still
decides which stack owns it.

The core follows the usual DDPM formulation on a cosine schedule with
eps-prediction and DDIM sampling, so that a difference against the flow arm is
attributable to the objective and not to a different architecture.

One property worth naming, because it is the practical difference between the
two cores. Recovering the clean sequence here requires

    x0 = (x_tau - sqrt(1 - abar) * eps) / sqrt(abar),

whose division blows up as abar -> 0 at the noisy end. The flow core's estimate,
x0 = x_t - t * v, has no division at any t. Nothing in this file depends on that
estimate during training, so the ablation is unaffected, but it is why the
sampler clamps.
"""
import math

import torch

from .config import FlowConfig


def cosine_betas(T, s=0.008):
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return (1 - ab[1:] / ab[:-1]).clamp(1e-8, 0.999).float()


class SegmentedDiffusion:
    """Same surface as SegmentedRectifiedFlow: loss(), sample(), split_sizes()."""

    def __init__(self, cfg: FlowConfig, device="cpu", n_steps=500, clamp=1.0):
        self.cfg = cfg
        self.device = device
        self.n_steps = n_steps
        self.clamp = clamp
        self.sigmas = torch.linspace(1.0, 0.0, cfg.segments + 1)
        b = cosine_betas(n_steps)
        self.betas = b
        self.abar = torch.cumprod(1.0 - b, dim=0)

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.abar = self.abar.to(device)
        return self

    # ---- mapping between the shared [0,1] axis and integer steps ----------
    def _tau(self, t):
        return (t * (self.n_steps - 1)).round().long().clamp(0, self.n_steps - 1)

    def _coef(self, t, like):
        tau = self._tau(t)
        a = self.abar.to(like.device)[tau]
        shape = (-1,) + (1,) * (like.dim() - 1)
        return a.sqrt().view(shape), (1 - a).sqrt().view(shape)

    def sample_prior(self, shape, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)

    def corrupt(self, x0, t, noise):
        a, b = self._coef(t, x0)
        return a * x0 + b * noise

    def predict_clean(self, x_tau, t, eps, clamp=None):
        a, b = self._coef(t, x_tau)
        x0 = (x_tau - b * eps) / a.clamp_min(1e-4)
        return x0.clamp(-clamp, clamp) if clamp is not None else x0

    # ---- the segment schedule, identical to the flow core ----------------
    def split_sizes(self, batch_size):
        K = self.cfg.segments
        base, rem = divmod(batch_size, K)
        return [base + 1 if i < rem else base for i in range(K)]

    def sample_segment_t(self, split_sizes, device, generator=None):
        parts = []
        for seg, n in enumerate(split_sizes):
            if n == 0:
                continue
            hi, lo = self.sigmas[seg].item(), self.sigmas[seg + 1].item()
            u = torch.rand(n, device=device, generator=generator)
            parts.append(lo + (hi - lo) * u)
        return torch.cat(parts, dim=0)

    def segment_of(self, t):
        K = self.cfg.segments
        return min(max(int((1.0 - float(t)) * K), 0), K - 1)

    # ------------------------------------------------------------------ loss
    def loss(self, model, x_data, y=None, z_target=None, align_weight=0.0,
             generator=None):
        """eps-prediction, stratified across segments exactly as the flow core."""
        B = x_data.shape[0]
        device = x_data.device
        sizes = self.split_sizes(B)
        t = self.sample_segment_t(sizes, device, generator)

        noise = self.sample_prior(x_data.shape, device, generator)
        x_tau = self.corrupt(x_data, t, noise)
        eps, _ = model.forward_train(x_tau, t, sizes, y=y)
        denoise = ((eps - noise) ** 2).mean()
        return {"total": denoise, "denoise": denoise.detach(),
                "align": torch.zeros((), device=device)}

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, model, shape, steps_per_segment=None, device=None, y=None,
               generator=None, return_nfe=False):
        """DDIM, walking the segments in order. NFE = segments * steps_per_segment."""
        device = device or self.device
        n = steps_per_segment or self.cfg.steps_per_segment
        x = self.sample_prior(shape, device, generator)
        nfe = 0
        grid_all = []
        for seg in range(self.cfg.segments):
            hi, lo = self.sigmas[seg].item(), self.sigmas[seg + 1].item()
            g = torch.linspace(hi, lo, n + 1)
            grid_all += [(seg, float(g[i]), float(g[i + 1])) for i in range(n)]

        for seg, t_cur, t_next in grid_all:
            t_vec = torch.full((shape[0],), t_cur, device=device)
            eps, _ = model.forward_sample(x, t_vec, seg, y=y)
            nfe += 1
            x0 = self.predict_clean(x, t_vec, eps, clamp=self.clamp)
            if t_next <= 0:
                x = x0
            else:
                a_prev = self.abar.to(device)[self._tau(
                    torch.full((1,), t_next, device=device))][0]
                x = a_prev.sqrt() * x0 + (1 - a_prev).clamp_min(0).sqrt() * eps
        return (x, nfe) if return_nfe else x
