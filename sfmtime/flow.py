"""Segment-stratified rectified flow: BFM's scheduler on the backbone's interpolation.

The two lineages already agree on the path, which is why no sign reconciliation
was needed in the port. BFM's `blockwise_flow_matching` uses

    alpha_t = 1 - t,  sigma_t = t,  x_t = (1-t)*x + t*eps,  v = eps - x

and integrates from t=1 (noise) to t=0 (data); the backbone's `RectifiedFlow` uses the
identical straight-line interpolation and the identical target. The only thing
BFM adds here is *where t comes from*: instead of t ~ U(0,1) for the whole
minibatch, the batch is split evenly across `segments` intervals and each slice
draws t uniformly inside its own interval, so every velocity stack is updated at
every step from B/segments samples.

Segment indexing follows BFM: `sigmas = linspace(1, 0, segments+1)`, so segment
0 covers the noise end [1, 1-1/K] and segment K-1 covers the data end [1/K, 0].
Sampling walks segments in that order.

the backbone's additions to the flow itself -- colored prior, minibatch OT coupling,
min-SNR weighting, temporal preconditioner -- are deliberately absent: this is
BFM's loss as published, with a white Gaussian prior, independent pairing and an
unweighted squared error. Run those variants in the the backbone repo.
"""
import torch
import torch.nn.functional as Fn

from .config import FlowConfig


class SegmentedRectifiedFlow:
    """Holds no parameters -- just the flow mechanics and the segment schedule."""

    def __init__(self, cfg: FlowConfig, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.sigmas = torch.linspace(1.0, 0.0, cfg.segments + 1)

    def to(self, device):
        self.device = device
        return self

    # -------------------------------------------------------------- mechanics
    def sample_prior(self, shape, device, generator=None):
        return torch.randn(shape, device=device, generator=generator)

    def corrupt(self, x_data, t, noise):
        tt = t.view(-1, *([1] * (x_data.dim() - 1)))
        return (1 - tt) * x_data + tt * noise

    def predict_clean(self, x_t, t, v, clamp=None):
        """The division-free clean-sequence estimate, x0 = x_t - t*v.

        Not used by the samplers below -- BFM integrates the velocity directly --
        but it is what any conditional (imputation / forecasting) sampler built
        on top of this would need, so it stays.
        """
        tt = t.view(-1, *([1] * (x_t.dim() - 1)))
        x0 = x_t - tt * v
        return x0.clamp(-clamp, clamp) if clamp is not None else x0

    def split_sizes(self, batch_size):
        """BFM's even split of the batch across segments."""
        K = self.cfg.segments
        base, rem = divmod(batch_size, K)
        return [base + 1 if i < rem else base for i in range(K)]

    def sample_segment_t(self, split_sizes, device, generator=None):
        """t stratified by segment, concatenated in segment order to match the
        contiguous batch slices the model's `split_sizes` addresses."""
        parts = []
        for seg, n in enumerate(split_sizes):
            if n == 0:
                continue
            hi, lo = self.sigmas[seg].item(), self.sigmas[seg + 1].item()
            u = torch.rand(n, device=device, generator=generator)
            parts.append(lo + (hi - lo) * u)
        return torch.cat(parts, dim=0)

    def segment_of(self, t):
        """Index of the segment containing scalar time t."""
        K = self.cfg.segments
        idx = int((1.0 - float(t)) * K)
        return min(max(idx, 0), K - 1)

    # ------------------------------------------------------------------- loss
    def loss(self, model, x_data, y=None, z_target=None, align_weight=0.0,
             generator=None):
        """One BFM training step's losses.

        `z_target` is the external-encoder feature for the alignment term
        (BFM's DINOv2 features, weight 0.05). Left None -- and `align_weight`
        left 0 -- the trunk is trained through the velocity loss alone. See
        config.py for why that is the default here and why TS2Vec must not be
        used for it.
        """
        B = x_data.shape[0]
        device = x_data.device
        sizes = self.split_sizes(B)
        t = self.sample_segment_t(sizes, device, generator)

        noise = self.sample_prior(x_data.shape, device, generator)
        x_t = self.corrupt(x_data, t, noise)
        target = noise - x_data

        v, z = model.forward_train(x_t, t, sizes, y=y)
        denoise = ((v - target) ** 2).mean()

        align = torch.zeros((), device=device)
        if align_weight and z is not None and z_target is not None:
            align = -(Fn.normalize(z_target, dim=-1) *
                      Fn.normalize(z, dim=-1)).sum(dim=-1).mean()

        return {"total": denoise + align_weight * align,
                "denoise": denoise.detach(), "align": align.detach()}

    def frn_loss(self, model, x_data, y=None, generator=None):
        """Stage-2 loss: how well the residual approximation reproduces the
        frozen trunk's feature at t, given its feature at the segment start."""
        B = x_data.shape[0]
        device = x_data.device
        sizes = self.split_sizes(B)
        t = self.sample_segment_t(sizes, device, generator)

        t_start = []
        for seg, n in enumerate(sizes):
            if n == 0:
                continue
            t_start.append(torch.full((n,), self.sigmas[seg].item(), device=device))
        t_start = torch.cat(t_start, dim=0)

        # fraction of the way through the segment, as in BFM's sampler
        widths = torch.zeros_like(t)
        off = 0
        for seg, n in enumerate(sizes):
            if n == 0:
                continue
            hi, lo = self.sigmas[seg].item(), self.sigmas[seg + 1].item()
            widths[off:off + n] = hi - lo
            off += n
        coeff = (t_start - t) / widths.clamp_min(1e-8)

        noise = self.sample_prior(x_data.shape, device, generator)
        x_t = self.corrupt(x_data, t, noise)
        x_start = self.corrupt(x_data, t_start, noise)

        approx, target = model.forward_frn_train(x_t, t, x_start, t_start, coeff, y=y)
        return {"total": ((approx - target) ** 2).mean()}

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, model, shape, steps_per_segment=None, device=None, y=None,
               generator=None, return_nfe=False, cond_target=None, cond_mask=None,
               clamp=None):
        """Euler integration segment by segment. NFE = segments * steps_per_segment.

        With `cond_mask` marking the observed entries of `cond_target`, sampling
        is conditioned by masked replacement, imposed on the model's own
        clean-sequence estimate rather than on the integrated state. Given the
        velocity v at (x, t), the division-free estimate and the noise it
        implies are

            x0_hat = x_t - t v,      eps_hat = x0_hat + v,

        which reproduce the state exactly, since (1-t) x0_hat + t eps_hat = x_t.
        Setting the observed entries of x0_hat to y and re-forming the state at
        the next time t' along the same interpolation gives

            x_obs  <- (1-t') y + t' eps_hat,
            x_hid  <- x + (t'-t) v.

        Wherever the mask is 0 this is the plain Euler step, so the hidden
        entries are integrated exactly as in unconditional sampling.

        The textbook form instead overwrites the state itself,

            x_t <- m * [(1-t) y + t eps] + (1-m) * x_t,

        with eps drawn independently of the sample being generated. That splices
        a model-independent trajectory into the state the next evaluation reads,
        and the hidden region has no way to reconcile with a trajectory it did
        not produce. Imposing the constraint on the clean estimate removes the
        splice at no cost: the observed entries then carry the model's own noise
        realisation. The difference is large -- on Sines imputation it is close
        to two orders of magnitude -- which is why only this form is offered.

        The replacement is applied after every step rather than only at segment
        boundaries: between boundaries the hidden entries are being integrated,
        and letting the observed ones drift would feed the next evaluation a
        state that does not satisfy the observations. At t=0 the carried target
        is y itself, so the returned sample agrees with every observation
        exactly. This conditions the sampler; it is not an exact sampler from
        the conditional distribution.

        Either way it is elementwise arithmetic, so conditioning adds no network
        evaluations and the NFE count is exactly the unconditional one. Note
        also that because the path is the straight line, replacing the state is
        equivalent to projecting the velocity at the observed entries onto the
        known value eps - y, the carried state being the exact integral of that
        constant velocity; there is no separate velocity-projection variant.
        """
        device = device or self.device
        n = steps_per_segment or self.cfg.steps_per_segment
        x = self.sample_prior(shape, device, generator)
        nfe = 0
        cond = cond_target is not None and cond_mask is not None
        if cond:
            cond_target = cond_target.to(device)
            m = cond_mask.to(device).to(x.dtype)
            # Nothing to initialise at t=1: the carried target there is
            # (1-1)*y + 1*eps = eps, so the observed entries already start on
            # their path. Writing y instead would hand the model clean values
            # while telling it t=1, a state it never saw in training.

        for seg in range(self.cfg.segments):
            hi, lo = self.sigmas[seg].item(), self.sigmas[seg + 1].item()
            grid = torch.linspace(hi, lo, n + 1, device=device)
            for i in range(n):
                t_cur, t_next = float(grid[i]), float(grid[i + 1])
                t_vec = torch.full((shape[0],), t_cur, device=device)
                v, _ = model.forward_sample(x, t_vec, seg, y=y)
                nfe += 1
                x_hid = x + (t_next - t_cur) * v
                if not cond:
                    x = x_hid
                    continue
                x0 = self.predict_clean(x, t_vec, v, clamp=clamp)
                implied = x0 + v
                x_obs = (1 - t_next) * cond_target + t_next * implied
                x = m * x_obs + (1 - m) * x_hid
        return (x, nfe) if return_nfe else x

    @torch.no_grad()
    def sample_frn(self, model, shape, steps_per_segment=None, device=None, y=None,
                   generator=None, return_nfe=False):
        """BFM's accelerated sampler: the trunk runs only at each segment's first
        step; later steps in that segment reuse its feature plus the cheap FRN
        correction, scaled by how far through the segment t has travelled."""
        assert model.frn_blocks is not None, "call model.enable_frn() and train stage 2"
        device = device or self.device
        n = steps_per_segment or self.cfg.steps_per_segment
        x = self.sample_prior(shape, device, generator)
        nfe = 0

        for seg in range(self.cfg.segments):
            hi, lo = self.sigmas[seg].item(), self.sigmas[seg + 1].item()
            grid = torch.linspace(hi, lo, n + 1, device=device)
            rep = None
            for i in range(n):
                t_cur, t_next = grid[i], grid[i + 1]
                t_vec = torch.full((shape[0],), t_cur.item(), device=device)
                coeff = torch.full((shape[0],), (hi - t_cur.item()) / max(hi - lo, 1e-8),
                                   device=device)
                v, rep = model.forward_sample(x, t_vec, seg, y=y,
                                              representation_feature=rep, coeff=coeff)
                nfe += 1
                x = x + (t_next - t_cur) * v
        return (x, nfe) if return_nfe else x
