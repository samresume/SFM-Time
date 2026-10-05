"""SFM-Time: segmented velocity blocks + a shared representation trunk, for (B, T, F).

Structure, following BFM:

    x_t ──► rep_embedder ──► [rep_depth shared blocks] ──► h_rep ──┬──► proj head ──► z
                                                                   │   (alignment, optional)
    x_t ──► x_embedder ────────────────────────────────────────────┴──► c_repre
                                                                          │
                          [segment_depth blocks for the segment holding t] │
    x_t ──► x_embedder ──────────────────────────────────────────────────►┴──► final ──► v

The trunk is evaluated once per step and shared across all segments; only the
velocity stack for the segment containing t is evaluated. Per step that is
`rep_depth + segment_depth` blocks out of `rep_depth + segments*segment_depth`
held in memory -- the source of BFM's inference saving, and the reason
`n_params_active()` is reported separately from `n_params()`.

During training a single minibatch is split across segments (`split_sizes`, one
count per segment, summing to B and ordered by segment). The trunk runs over the
whole batch; each contiguous slice then goes through its own velocity stack and
the outputs are concatenated back. So every segment receives a gradient at every
step from B/segments samples.
"""
import torch
import torch.nn as nn

from .config import ModelConfig
from .modules import (
    TokenAdaLNBlock, PlainBlock, FinalLayer, LabelEmbedder,
    timestep_embedding, rope_cache,
)


class SFMTime(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        p = cfg.patch_size
        d_in = cfg.n_features * p
        self.n_tok = cfg.n_tokens()
        self.use_trunk = cfg.rep_depth > 0

        # ---- embedders -----------------------------------------------------
        self.x_embedder = nn.Linear(d_in, d)
        self.rep_embedder = nn.Linear(d_in, d) if self.use_trunk else None

        self.t_embedder = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.y_embedder = (LabelEmbedder(cfg.n_classes, d, cfg.class_dropout_prob)
                           if cfg.n_classes else None)

        # ---- block factory -------------------------------------------------
        Block = TokenAdaLNBlock if cfg.cond_mode == "adaln_token" else PlainBlock
        token_adaln = cfg.cond_mode == "adaln_token"

        def mk(d_cond):
            return Block(d, cfg.n_heads, d_cond, cfg.d_ff_mult, cfg.dropout,
                         cfg.causal, cfg.use_rope, cfg.use_lag_bias, self.n_tok)

        # conditioning width: time embedding, plus the trunk feature when present
        d_cond_vel = 2 * d if self.use_trunk else d
        self.d_cond_vel = d_cond_vel

        # ---- shared representation trunk (Semantic Feature Guidance) -------
        if self.use_trunk:
            self.representation_blocks = nn.ModuleList([mk(d) for _ in range(cfg.rep_depth)])
            if cfg.proj_dim:
                self.linear_projection = nn.Sequential(
                    nn.Linear(d, cfg.proj_hidden), nn.SiLU(),
                    nn.Linear(cfg.proj_hidden, cfg.proj_hidden), nn.SiLU(),
                    nn.Linear(cfg.proj_hidden, cfg.proj_dim),
                )
            else:
                self.linear_projection = None
        else:
            self.representation_blocks = None
            self.linear_projection = None

        # ---- per-segment velocity stacks -----------------------------------
        self.velocity_blocks = nn.ModuleList([
            nn.ModuleList([mk(d_cond_vel) for _ in range(cfg.segment_depth)])
            for _ in range(cfg.segments)
        ])

        # ---- BFM's 'lora' adaLN: a global modulation added to every block ---
        if cfg.use_global_adaln and token_adaln:
            self.global_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
            self.global_adaLN_modulation2 = nn.Sequential(nn.SiLU(), nn.Linear(d_cond_vel, 6 * d))
            for m in (self.global_adaLN_modulation, self.global_adaLN_modulation2):
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
        else:
            self.global_adaLN_modulation = None
            self.global_adaLN_modulation2 = None

        self.final_layer = FinalLayer(d, d_in, d_cond_vel, token_adaln=token_adaln)

        # ---- FRA / FRN second stage ----------------------------------------
        self.frn_blocks = None

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(basic)
        # re-zero what must stay zero after the sweep above
        for stack in self.velocity_blocks:
            for blk in stack:
                if hasattr(blk, "adaLN_modulation"):
                    nn.init.zeros_(blk.adaLN_modulation[-1].weight)
                    nn.init.zeros_(blk.adaLN_modulation[-1].bias)
        if self.representation_blocks is not None:
            for blk in self.representation_blocks:
                if hasattr(blk, "adaLN_modulation"):
                    nn.init.zeros_(blk.adaLN_modulation[-1].weight)
                    nn.init.zeros_(blk.adaLN_modulation[-1].bias)
        for m in (self.global_adaLN_modulation, self.global_adaLN_modulation2):
            if m is not None:
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
        if self.final_layer.token_adaln:
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def enable_frn(self, frn_depth=None, freeze_backbone=True):
        """Attach the Feature Residual Approximation blocks for stage 2.

        BFM freezes everything but these blocks: the trunk's output at a segment's
        first step is reused for the rest of the segment, corrected by a cheap
        `frn_depth`-block residual scaled by how far through the segment t is. The
        velocity field must stay fixed for that approximation to be measured
        against anything.
        """
        cfg = self.cfg
        depth = frn_depth or cfg.frn_depth
        Block = TokenAdaLNBlock if cfg.cond_mode == "adaln_token" else PlainBlock
        self.frn_blocks = nn.ModuleList([
            Block(cfg.d_model, cfg.n_heads, cfg.d_model, cfg.d_ff_mult, cfg.dropout,
                  cfg.causal, cfg.use_rope, cfg.use_lag_bias, self.n_tok)
            for _ in range(depth)
        ])
        self.frn_blocks.to(next(self.parameters()).device)
        if freeze_backbone:
            for p_ in self.parameters():
                p_.requires_grad = False
            for p_ in self.frn_blocks.parameters():
                p_.requires_grad = True
        return self

    # -------------------------------------------------------------- reshaping
    def patchify(self, x):
        """(B, T, F) -> (B, T/p, F*p). p=1 is a no-op reshape."""
        B, T, Fdim = x.shape
        p = self.cfg.patch_size
        return x.reshape(B, T // p, p * Fdim)

    def unpatchify(self, x):
        """(B, T/p, F*p) -> (B, T, F)."""
        B, N, _ = x.shape
        p = self.cfg.patch_size
        return x.reshape(B, N * p, self.cfg.n_features)

    def _rope(self, device):
        if not self.cfg.use_rope:
            return None
        return rope_cache(self.n_tok, self.cfg.head_dim(), device)

    # ------------------------------------------------------------ conditioning
    def _cond(self, t, y, training):
        """Per-sample conditioning vector c = t_emb (+ y_emb)."""
        c = self.t_embedder(timestep_embedding(t, self.cfg.d_model))
        if self.y_embedder is not None and y is not None:
            c = c + self.y_embedder(y, training)
        return c

    def _trunk(self, x_tok, c_tok, rope, blocks, global_adaln):
        h = x_tok
        for blk in blocks:
            h = blk(h, c_tok, rope=rope, global_adaln=global_adaln)
        return h

    def _embed(self, x):
        xp = self.patchify(x)
        return xp, self.x_embedder(xp)

    def _representation(self, xp, c_tok, rope, c):
        """Run the shared trunk and build the per-token velocity conditioning."""
        if not self.use_trunk:
            return None, c_tok, None
        ga = (self.global_adaLN_modulation(c).unsqueeze(1)
              if self.global_adaLN_modulation is not None else 0.0)
        h_rep = self._trunk(self.rep_embedder(xp), c_tok, rope,
                            self.representation_blocks, ga)
        c_repre = torch.cat([c_tok, h_rep], dim=-1)
        z = self.linear_projection(h_rep) if self.linear_projection is not None else None
        return h_rep, c_repre, z

    def _global2(self, c_repre):
        if self.global_adaLN_modulation2 is None:
            return 0.0
        return self.global_adaLN_modulation2(c_repre)

    # ------------------------------------------------------------------ train
    def forward_train(self, x_t, t, split_sizes, y=None):
        """x_t: (B, T, F); t: (B,); split_sizes: per-segment counts summing to B.

        Returns (v, z) where v is (B, T, F) and z is the projected trunk feature
        for the optional alignment loss (None when the projection head is off).
        """
        rope = self._rope(x_t.device)
        c = self._cond(t, y, self.training)
        c_tok = c.unsqueeze(1).expand(-1, self.n_tok, -1)

        xp, h = self._embed(x_t)
        _, c_repre, z = self._representation(xp, c_tok, rope, c)
        ga2 = self._global2(c_repre)

        h_sp = torch.split(h, split_sizes, dim=0)
        c_sp = torch.split(c_repre, split_sizes, dim=0)
        ga2_sp = (torch.split(ga2, split_sizes, dim=0)
                  if torch.is_tensor(ga2) else [0.0] * len(split_sizes))

        outs = []
        for seg, n in enumerate(split_sizes):
            if n == 0:
                continue
            hs, cs, gs = h_sp[seg], c_sp[seg], ga2_sp[seg]
            for blk in self.velocity_blocks[seg]:
                hs = blk(hs, cs, rope=rope, global_adaln=gs)
            outs.append(self.final_layer(hs, cs))
        v = self.unpatchify(torch.cat(outs, dim=0))
        return v, z

    # ----------------------------------------------------------------- sample
    @torch.no_grad()
    def forward_sample(self, x_t, t, segment_idx, y=None,
                       representation_feature=None, coeff=None):
        """One velocity evaluation inside `segment_idx`.

        If `representation_feature` is given, the trunk is *not* re-run: the
        stored feature from the segment's first step is corrected by the FRN
        blocks scaled by `coeff` (BFM's Feature Residual Approximation). The
        returned feature is always the one to carry forward within the segment.
        """
        rope = self._rope(x_t.device)
        c = self._cond(t, y, False)
        c_tok = c.unsqueeze(1).expand(-1, self.n_tok, -1)
        xp, h = self._embed(x_t)

        if not self.use_trunk:
            c_repre, carry = c_tok, None
        elif representation_feature is None or self.frn_blocks is None:
            h_rep, c_repre, _ = self._representation(xp, c_tok, rope, c)
            carry = h_rep
        else:
            approx = representation_feature
            resid = self.rep_embedder(xp)
            for blk in self.frn_blocks:
                resid = blk(resid, c_tok, rope=rope, global_adaln=0.0)
            cf = coeff if torch.is_tensor(coeff) else torch.as_tensor(
                coeff, device=x_t.device, dtype=x_t.dtype)
            cf = cf.reshape(-1, 1, 1).to(x_t.dtype) if cf.dim() else cf
            h_rep = approx + cf * resid
            c_repre = torch.cat([c_tok, h_rep], dim=-1)
            carry = approx

        ga2 = self._global2(c_repre)
        for blk in self.velocity_blocks[segment_idx]:
            h = blk(h, c_repre, rope=rope, global_adaln=ga2)
        return self.unpatchify(self.final_layer(h, c_repre)), carry

    # -------------------------------------------------------------- FRN stage
    def forward_frn_train(self, x_t, t, x_start, t_start, coeff, y=None):
        """Stage-2 target/prediction pair for the residual approximation.

        Returns (approx, target): the approximated trunk feature at t built from
        the frozen trunk's feature at the segment start, and the frozen trunk's
        true feature at t. The stage-2 loss is the distance between them, so the
        velocity field never moves.
        """
        assert self.frn_blocks is not None, "call enable_frn() first"
        rope = self._rope(x_t.device)
        with torch.no_grad():
            c = self._cond(t, y, False)
            c_tok = c.unsqueeze(1).expand(-1, self.n_tok, -1)
            c_s = self._cond(t_start, y, False)
            c_s_tok = c_s.unsqueeze(1).expand(-1, self.n_tok, -1)

            xp = self.patchify(x_t)
            xp_s = self.patchify(x_start)

            ga_s = (self.global_adaLN_modulation(c_s).unsqueeze(1)
                    if self.global_adaLN_modulation is not None else 0.0)
            ga = (self.global_adaLN_modulation(c).unsqueeze(1)
                  if self.global_adaLN_modulation is not None else 0.0)

            h_start = self._trunk(self.rep_embedder(xp_s), c_s_tok, rope,
                                  self.representation_blocks, ga_s)
            target = self._trunk(self.rep_embedder(xp), c_tok, rope,
                                 self.representation_blocks, ga)
            resid_in = self.rep_embedder(xp).detach()

        resid = resid_in
        for blk in self.frn_blocks:
            resid = blk(resid, c_tok, rope=rope, global_adaln=0.0)
        cf = coeff.reshape(-1, 1, 1).to(x_t.dtype)
        return h_start.detach() + cf * resid, target.detach()

    # ------------------------------------------------------------------ counts
    def n_params(self):
        """All parameters held, including velocity stacks for unvisited segments."""
        return sum(p.numel() for p in self.parameters())

    def n_params_active(self):
        """Parameters evaluated in a single velocity call: embedders + trunk +
        one segment stack + final layer. This is the number to compare against a
        monolithic model's parameter count when arguing about per-step cost."""
        n = sum(p.numel() for p in self.x_embedder.parameters())
        n += sum(p.numel() for p in self.t_embedder.parameters())
        if self.use_trunk:
            n += sum(p.numel() for p in self.rep_embedder.parameters())
            n += sum(p.numel() for p in self.representation_blocks.parameters())
        if self.global_adaLN_modulation is not None:
            n += sum(p.numel() for p in self.global_adaLN_modulation.parameters())
            n += sum(p.numel() for p in self.global_adaLN_modulation2.parameters())
        n += sum(p.numel() for p in self.velocity_blocks[0].parameters())
        n += sum(p.numel() for p in self.final_layer.parameters())
        return n

    def n_blocks_active(self):
        return self.cfg.rep_depth + self.cfg.segment_depth
