"""Configuration for SFM-Time: Blockwise Flow Matching for multivariate time series.

A port of Blockwise Flow Matching (Park, Lee, Joo & Kim, NeurIPS 2025;
arXiv:2510.21167, github.com/mlvlab/Blockwise-Flow-Matching) from ImageNet
latents to (B, T, F) multivariate time series, with the image DiT backbone
replaced by the bidirectional RoPE Transformer used by a bidirectional RoPE Transformer for sequences.

What is kept from BFM, unchanged in substance:

  * the trajectory is partitioned into `segments` intervals of flow time, and
    each interval gets its own stack of `segment_depth` velocity blocks;
  * a *shared* representation trunk of `rep_depth` blocks is evaluated at every
    step, and its per-token output conditions the velocity blocks (the
    structural half of BFM's Semantic Feature Guidance);
  * the per-block adaLN modulation is the sum of a globally computed
    modulation and a per-block one (BFM's `adaln_type='lora'` scheme);
  * training draws one minibatch, splits it evenly across segments, and samples
    t uniformly *within* each segment's interval, so every segment is updated
    at every step;
  * sampling walks the segments in order, taking `steps_per_segment` Euler
    steps inside each, so NFE = segments * steps_per_segment;
  * Feature Residual Approximation (FRA/FRN) is available as a second stage.

What necessarily changed, and why:

  * (B, C, H, W) VAE latents with 2-D patchify become (B, T, F) sequences with
    an optional patch along time only. There is no spatial axis to patch.
  * 2-D vision RoPE becomes the 1-D RoPE over timesteps from the backbone's
    `modules.py`, and attention is bidirectional self-attention. This
    is the substitution requested: BFM's blocks are already bidirectional, so
    what changes is the position encoding and the attention implementation,
    not the information flow.
  * class conditioning and classifier-free guidance are dropped by default
    (`n_classes=None`). The four benchmarks are unconditional generation.

BFM's representation *alignment* (Semantic Feature Guidance) is implemented but
**off by default** (`align_weight=0.0`, `proj_dim=0`). BFM aligns the trunk to
DINOv2 features. There is no agreed pretrained encoder for MVTS, and the obvious
candidate in this codebase -- TS2Vec -- is the *evaluation* encoder: Context-FID
is a Frechet distance in TS2Vec space (`benchmark_metrics.py`), so aligning
to it would mean training against the metric. Set `proj_dim` to the encoder's
width, `align_weight` to BFM's 0.05, and pass `z_target` to `flow.loss` once that
encoder is chosen deliberately.

What is deliberately **not implemented**:

  * other flow-matching refinements -- colored prior, minibatch OT coupling, min-SNR
    weighting, temporal preconditioner, and self-conditioning. Those live in
    other work and are out of scope here.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    # data shape
    seq_len: int = 64
    n_features: int = 4
    patch_size: int = 1              # patch along time; 1 = one token per step

    # width
    d_model: int = 128
    n_heads: int = 4
    d_ff_mult: int = 4
    dropout: float = 0.0

    # BFM depth structure
    segments: int = 6                # number of flow-time intervals
    segment_depth: int = 5           # velocity blocks per interval
    rep_depth: int = 20              # shared representation-trunk blocks

    # conditioning
    cond_mode: str = "adaln_token"   # 'adaln_token' (BFM) | 'add' (a plain pre-norm block)
    use_global_adaln: bool = True    # BFM's adaln_type='lora': global + per-block
    n_classes: Optional[int] = None  # None = unconditional
    class_dropout_prob: float = 0.1

    # backbone (the bidirectional transformer)
    use_rope: bool = True
    use_lag_bias: bool = False
    causal: bool = False             # both the image formulation and this one are bidirectional

    # alignment head (BFM's Semantic Feature Guidance projection)
    proj_dim: int = 0                # 0 disables the projection head entirely
    proj_hidden: int = 512

    # FRA / FRN second stage
    frn_depth: int = 4

    def head_dim(self):
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        return self.d_model // self.n_heads

    def n_tokens(self):
        assert self.seq_len % self.patch_size == 0, "patch_size must divide seq_len"
        return self.seq_len // self.patch_size


@dataclass
class FlowConfig:
    segments: int = 6                # must match ModelConfig.segments
    steps_per_segment: int = 8       # BFM's eval default; NFE = segments * this
    core: str = "flow"               # 'flow' | 'ddpm' (the ablation arm)
    ddpm_steps: int = 500            # diffusion timesteps when core='ddpm'


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 96             # should be >= segments, ideally a multiple
    lr: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    warmup_steps: int = 200
    align_weight: float = 0.0        # BFM uses 0.05 with DINOv2; see module docstring
    log_every: int = 200
    seed: int = 0
    device: Optional[str] = None     # None -> mps/cuda/cpu autodetect


@dataclass
class SFMTimeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    name: str = "sfmtime"

    def __post_init__(self):
        # one source of truth for the segment count
        self.flow.segments = self.model.segments

    def to_dict(self):
        return {"name": self.name, "model": asdict(self.model),
                "flow": asdict(self.flow), "train": asdict(self.train)}

    # ---- presets -----------------------------------------------------------
    @classmethod
    def faithful(cls, seq_len, n_features, d_model=128, n_heads=4):
        """BFM's published depth structure (6 x 5 velocity + 20 trunk).

        Note the cost: 20 + 6*5 = 50 blocks total, 25 evaluated per step. At
        these widths that is ~10x a comparable sequence baseline's parameter count, so this arm is
        NOT capacity-matched to it -- use `param_matched` for that comparison.
        """
        return cls(
            model=ModelConfig(seq_len=seq_len, n_features=n_features,
                              d_model=d_model, n_heads=n_heads,
                              segments=6, segment_depth=5, rep_depth=20),
            flow=FlowConfig(steps_per_segment=8),
            name="sfmtime_faithful",
        )

    @classmethod
    def small(cls, seq_len, n_features, d_model=128, n_heads=4):
        """A depth structure sized for the MVTS benchmarks rather than ImageNet.

        Keeps BFM's mechanism (per-interval velocity blocks + shared trunk) at a
        depth where each velocity stack is still a functioning transformer.
        """
        return cls(
            model=ModelConfig(seq_len=seq_len, n_features=n_features,
                              d_model=d_model, n_heads=n_heads,
                              segments=4, segment_depth=2, rep_depth=4),
            flow=FlowConfig(steps_per_segment=5),
            name="sfmtime_small",
        )

    @classmethod
    def param_matched(cls, seq_len, n_features, target_params,
                      segments=4, segment_depth=2, rep_depth=4,
                      n_heads=4, d_min=32, d_max=512):
        """Pick `d_model` so total params land closest to `target_params`.

        A comparison against another model is only interpretable at matched
        capacity: BFM's whole design trades one large network for K smaller
        specialised ones, and a K-fold parameter increase would confound that.
        Pass the baseline's parameter count for the dataset as the target.
        """
        from .model import SFMTime
        best = None
        for d in range(d_min, d_max + 1, n_heads):
            if d % n_heads:
                continue
            cfg = cls(model=ModelConfig(seq_len=seq_len, n_features=n_features,
                                        d_model=d, n_heads=n_heads,
                                        segments=segments,
                                        segment_depth=segment_depth,
                                        rep_depth=rep_depth),
                      flow=FlowConfig(steps_per_segment=5),
                      name="sfmtime_matched")
            n = SFMTime(cfg.model).n_params()
            gap = abs(n - target_params)
            if best is None or gap < best[0]:
                best = (gap, d, n, cfg)
        _, d, n, cfg = best
        cfg.matched_params = n
        return cfg
