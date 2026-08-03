"""Tiled exact scoring: the training forward's per-position CE over a tail
span, at bounded memory, in two phases.  Phase A (bottom-up, all T tokens):
windowed encoders stream through rolling KV caches (write index != RoPE
offset; ``roll_left`` keeps baked phases); the NoPE CAP latent cache is the
only growing state; fixed tails ("rings") of encoder states are retained.
Phase B (top-down, T-independent): a streaming decoder's receptive field is
n*(w-1) stream positions, so decoding a segment with warm = ceil(n*(w-1)/(R+C))
warm-up groups (discarded) reproduces the full-stream outputs exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..model.hierarchy import ScalaForCausalLM

__all__ = ["TiledScorer"]


@dataclass
class _LevelPlan:
    seg: int          # segment length, in this level's decoder groups
    warm: int         # leading groups computed but discarded
    g0: int           # absolute index of the segment's first group
    ring: int         # retained tail of THIS level's encoder states, in units


class TiledScorer:
    """Exact tail scoring at bounded memory.  One instance per (model, batch,
    dtype); ``score_span`` is stateless across calls (it re-feeds)."""

    def __init__(self, model: ScalaForCausalLM, device, dtype,
                 tile_tokens: int = 65_536, enc_block_units: int = 1_024):
        cfg = model.cfg
        if cfg.global_skip:
            raise NotImplementedError("TiledScorer v1 does not support global_skip")
        if cfg.mtp_depth:
            pass  # MTP heads are unused in teacher-forced scoring
        for i, lvl in enumerate(model.levels, start=1):
            if not lvl.stream:
                raise NotImplementedError("all decoders must be decoder_stream")
            if lvl.lookback:
                raise NotImplementedError("decoder_lookback is not supported")
            if i < cfg.n_levels and not lvl.encoder_window:
                raise NotImplementedError("non-top encoders must be windowed")
        self.model = model
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.L = cfg.n_levels
        cp = cfg.chunk_product
        self.tile_tokens = max(cp, tile_tokens // cp * cp)
        self.enc_block_units = enc_block_units

    # ------------------------------------------------------------------ #
    def _plan(self, T: int, span: int,
              warm_override: dict[int, int] | None = None) -> list[_LevelPlan]:
        """Segment/ring arithmetic.  Level index 1..L; plans[l-1]."""
        cfg = self.cfg
        assert T % cfg.chunk_product == 0, \
            f"T={T} must be a multiple of C_<=L={cfg.chunk_product}"
        assert 0 < span <= T
        plans: list[_LevelPlan] = []
        seg_below = span  # tokens at "level 0"
        for l in range(1, self.L + 1):
            lvl = self.model.levels[l - 1]
            R, C, w = lvl.width, lvl.chunk, lvl.stream
            n = lvl.decoder.cfg.n_layers
            warm = math.ceil(n * (w - 1) / (R + C))
            if warm_override and l in warm_override:
                warm = warm_override[l]
            exact = math.ceil(seg_below / C)
            groups_l = T // cfg.cumulative_chunk(l)
            seg = min(exact + warm, groups_l)
            # A level whose warm-up would reach past the stream's start is
            # decoded WHOLE (g0=0): the full stream from position 0 is exact at
            # every row, so warm-up is void.  Once one level goes whole, every
            # level above it does too.
            plans.append(_LevelPlan(seg=seg,
                                    warm=(seg - exact) if seg > exact else 0,
                                    g0=groups_l - seg, ring=0))
            if plans[-1].g0 == 0:
                plans[-1].warm = 0
            seg_below = seg
        # rings: level j (1..L-1) holds content for level j+1's decoder;
        # level L holds the CAP decoder's conditioning (one extra unit for
        # the shift when the segment is proper; the whole stream when not)
        for j in range(1, self.L):
            plans[j - 1].ring = self.model.levels[j].chunk * plans[j].seg
        top = plans[self.L - 1]
        top.ring = top.seg + (1 if top.g0 >= 1 else 0)
        return plans

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _feed(self, tokens: torch.Tensor, plans: list[_LevelPlan]):
        """Phase A: stream all tokens bottom-up; return per-level rings."""
        m, cfg = self.model, self.cfg
        B, T = tokens.shape
        cp = cfg.chunk_product
        assert T % cp == 0, f"T={T} must be a multiple of C_<=L={cp}"
        # keep per-tile activation memory roughly batch-invariant
        tile = max(cp, self.tile_tokens // max(1, B) // cp * cp)

        # caches
        enc_caches, fills, abs_pos = [], [], []
        for l in range(1, self.L):
            lvl = m.levels[l - 1]
            cap_units = lvl.encoder_window + self.enc_block_units
            enc_caches.append(lvl.encoder.alloc_cache(
                B, cap_units, self.device, self.dtype))
            fills.append(0)
            abs_pos.append(0)
        cap_lvl = m.levels[-1]
        cap_cache = cap_lvl.encoder.alloc_cache(
            B, T // cp + 2, self.device, self.dtype)
        cap_written = 0

        rings: list[torch.Tensor | None] = [None] * self.L
        scale = math.sqrt(cfg.d_token) if cfg.scale_embeddings else 1.0

        for t0 in range(0, T, tile):
            x = m.embed(tokens[:, t0 : t0 + tile].to(self.device))
            if cfg.scale_embeddings:
                x = x * scale
            x = x.to(self.dtype)
            for l in range(1, self.L):            # windowed levels
                lvl = m.levels[l - 1]
                a = lvl.chunker(x)                # (B, units_in_tile, d)
                outs = []
                w = lvl.encoder_window
                for i in range(0, a.shape[1], self.enc_block_units):
                    blk = a[:, i : i + self.enc_block_units]
                    outs.append(lvl.encoder(
                        blk, cache=enc_caches[l - 1],
                        pos_offset=abs_pos[l - 1],
                        cache_start=fills[l - 1]))
                    abs_pos[l - 1] += blk.shape[1]
                    fills[l - 1] += blk.shape[1]
                    if fills[l - 1] > w:
                        enc_caches[l - 1].roll_left(fills[l - 1] - w)
                        fills[l - 1] = w
                x = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
                rings[l - 1] = self._keep_tail(rings[l - 1], x,
                                               plans[l - 1].ring)
            # CAP: NoPE latent append
            a = cap_lvl.chunker(x)
            out = cap_lvl.encoder(a, cache=cap_cache,
                                  pos_offset=cap_written,
                                  cache_start=cap_written)
            cap_written += a.shape[1]
            rings[self.L - 1] = self._keep_tail(rings[self.L - 1], out,
                                                plans[self.L - 1].ring)
            del x, a, out
        return rings

    @staticmethod
    def _keep_tail(ring: torch.Tensor | None, new: torch.Tensor,
                   keep: int) -> torch.Tensor:
        t = new if ring is None else torch.cat([ring, new], dim=1)
        return t[:, -keep:].contiguous() if t.shape[1] > keep else t

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score_span(self, tokens: torch.Tensor, span: int,
                   warm_override: dict[int, int] | None = None,
                   ce_chunk: int = 2048):
        """Per-position CE for the last ``span`` tokens of ``tokens``.

        Returns ``(mean_ce, per_pos)``: a float (mean over B*span, the same
        reduction as ``model(..., loss_mask=tail)``) and a (B, span) fp32 CPU
        tensor.
        """
        m, cfg = self.model, self.cfg
        B, T = tokens.shape
        plans = self._plan(T, span, warm_override)
        rings = self._feed(tokens, plans)

        # ---- phase B: top-down segment decode --------------------------- #
        # CAP: cond is enc[L] shifted by one unit.  For a proper segment
        # (g0 >= 1), slicing [g0-1 : end-1] of the ring IS the shifted tensor
        # restricted to the segment (shift=False).  A whole-stream top
        # (g0 == 0, the deep-depth case) takes the ordinary shift path.
        pl = plans[-1]
        ringL = rings[-1]
        shift_top = pl.g0 == 0
        if shift_top:
            assert ringL.shape[1] == pl.seg, "CAP ring underflow"
            cond = ringL
        else:
            assert ringL.shape[1] >= pl.seg + 1, "CAP ring underflow"
            cond = ringL[:, -(pl.seg + 1) : -1]
        readout = None
        for l in range(self.L, 0, -1):
            lvl = m.levels[l - 1]
            pl = plans[l - 1]
            R, C = lvl.width, lvl.chunk
            if l < self.L:
                above = plans[l]
                off = pl.g0 - m.levels[l].chunk * above.g0
                # a whole-stream upper level is exact from row 0; a proper
                # segment is exact only past its warm-up rows
                assert off >= m.levels[l].chunk * above.warm, \
                    "segment reads warm-up (inexact) conditioning"
                cond = readout[:, off : off + pl.seg]
            if l > 1:
                content = rings[l - 2][:, -C * pl.seg :]
                assert content.shape[1] == C * pl.seg, \
                    f"ring underflow at level {l - 1}"
            else:
                ids = tokens[:, T - C * pl.seg :].to(self.device)
                content = m.embed(ids)
                if cfg.scale_embeddings:
                    content = content * math.sqrt(cfg.d_token)
                content = content.to(self.dtype)
            readout = lvl.decode(cond, content,
                                 shift=(shift_top and l == self.L),
                                 pos_offset=(R + C) * pl.g0)
            del content

        # ---- head ------------------------------------------------------- #
        h = readout[:, -span:]
        h = m.final_norm(h)
        labels = tokens[:, -span:].to(self.device)
        per_pos = torch.empty(B, span, dtype=torch.float32)
        flat_h = h.reshape(B * span, -1)
        flat_l = labels.reshape(-1)
        out_rows = []
        for i in range(0, flat_h.shape[0], ce_chunk):
            logits = m.lm_head(flat_h[i : i + ce_chunk])
            if cfg.logit_softcap:
                cap = cfg.logit_softcap
                logits = torch.tanh(logits / cap) * cap
            out_rows.append(F.cross_entropy(
                logits.float(), flat_l[i : i + ce_chunk],
                reduction="none").cpu())
        per_pos = torch.cat(out_rows).view(B, span)
        return float(per_pos.mean()), per_pos
