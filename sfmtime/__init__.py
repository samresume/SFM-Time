"""SFM-Time -- Segmented Flow Matching for time-series generation.

Flow time is partitioned into K contiguous segments and each segment gets its
own small stack of velocity blocks, while one shared representation trunk
conditions every stack per time step. Blocks are bidirectional Transformers
with rotary position embeddings over timesteps.

The segmentation and the shared-trunk conditioning follow Blockwise Flow
Matching (Park et al., NeurIPS 2025, arXiv:2510.21167), which was developed for
image latents; what is adapted here is the backbone, the token layout and the
sampler, so that the field operates on (B, T, F) sequences. See `config.py` for
what is kept and what changed.
"""
from .config import SFMTimeConfig, ModelConfig, FlowConfig, TrainConfig
from .model import SFMTime
from .flow import SegmentedRectifiedFlow

__all__ = ["SFMTimeConfig", "ModelConfig", "FlowConfig", "TrainConfig",
           "SFMTime", "SegmentedRectifiedFlow", "SFMTimeTrainer"]


def __getattr__(name):
    if name == "SFMTimeTrainer":
        from .train import SFMTimeTrainer
        return SFMTimeTrainer
    raise AttributeError(name)
