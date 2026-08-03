"""Hierarchical generation for SCALA (reference: arXiv:2512.20687).

``_emit_group(l, cond)`` emits the C_l level-(l-1) units under one level-l
state using R_l + C_l decoder positions; at l=1 units are sampled tokens.
Protocols differ in how a finished group is summarised for the level above.
Unaligned prompts: encode the largest aligned prefix, teacher-force the rest.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as F

from ..model.layers import KVCache, LatentKVCache
from ..model.hierarchy import ScalaForCausalLM

__all__ = ["GenerationConfig", "ScalaGenerator", "generate"]

Mode = Literal["hiergen", "recgen", "recgen_paper", "chunkgen",
               "content_only", "up_only", "xhat_content"]


@dataclass(frozen=True)
class Protocol:
    """Substitution sites for a finished group of level-(l-1) units.

    ``content``: what the level-l decoder consumes at its next position.
    ``up``: folded by the level-l chunker into the level-l encoder input.
    The two sites are independent; substituting in ``up`` corrupts the
    top-level conditioning state, which the training objective never covers.
    """

    name: str
    lower_encoder: bool     # run the lower encoders incrementally, with caches
    content: str            # "encoder" | "chunker" | "xhat"
    up: str                 # "encoder" | "chunker" | "xhat"
    #: bound each lower encoder cache to ``enc_window_groups`` blocks of
    #: ``C_{l+1}`` units (O(1) in T).  Only meaningful with ``lower_encoder``.
    windowed: bool = False


PROTOCOLS: dict[str, Protocol] = {
    #                       name           cache  content    up
    "hiergen":     Protocol("hiergen",     True,  "encoder", "encoder"),
    #: only the top-level cache grows with T; lower encoder caches bounded
    "recgen":      Protocol("recgen",      True,  "encoder", "encoder",
                            windowed=True),
    #: the paper's rule verbatim (Eq. (6) content, step (ii) upward);
    #: diagnostic only, not for generating
    "recgen_paper": Protocol("recgen_paper", False, "xhat",  "xhat"),
    #: cache-free fallback: chunker summary of the emitted group, no KV below
    #: the top level
    "chunkgen":    Protocol("chunkgen",    False, "chunker", "chunker"),
    # diagnostic crosses -- one substitution each, lower caches kept
    "content_only": Protocol("content_only", True, "chunker", "encoder"),
    "up_only":      Protocol("up_only",      True, "encoder", "chunker"),
    #: content substitution alone; matches the training forward exactly under
    #: ``recursive_decoder_input``, not under teacher forcing
    "xhat_content": Protocol("xhat_content", True, "xhat", "encoder"),
}

#: protocols that keep no KV cache below the top level
_NO_LOWER_ENCODER = tuple(k for k, p in PROTOCOLS.items() if not p.lower_encoder)

#: windowed-encoder history, in blocks of ``C_{l+1}`` units.  1 == the
#: ``encoder_block_local`` receptive field; the limit as it grows is HierGen.
DEFAULT_ENC_WINDOW_GROUPS = 4


# --------------------------------------------------------------------------- #
@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    greedy: bool = False
    eos_token_id: Optional[int] = None
    mode: Mode = "hiergen"
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
def sample_from_logits(logits: torch.Tensor, cfg: GenerationConfig,
                       history: Optional[torch.Tensor],
                       generator: Optional[torch.Generator]) -> torch.Tensor:
    """logits: (B, V) -> (B,) token ids."""
    logits = logits.float()

    if cfg.repetition_penalty != 1.0 and history is not None and history.numel():
        for b in range(logits.shape[0]):
            uniq = torch.unique(history[b])
            sel = logits[b, uniq]
            logits[b, uniq] = torch.where(sel > 0, sel / cfg.repetition_penalty,
                                          sel * cfg.repetition_penalty)

    if cfg.greedy or cfg.temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits / cfg.temperature

    if cfg.top_k and cfg.top_k > 0:
        k = min(cfg.top_k, logits.shape[-1])
        kth = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = logits.softmax(dim=-1)

    if cfg.min_p and cfg.min_p > 0:
        thresh = cfg.min_p * probs.max(dim=-1, keepdim=True).values
        probs = probs.masked_fill(probs < thresh, 0.0)

    if cfg.top_p and 0 < cfg.top_p < 1:
        srt, idx = probs.sort(dim=-1, descending=True)
        cum = srt.cumsum(dim=-1)
        drop = cum - srt > cfg.top_p
        srt = srt.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, idx, srt)

    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


# --------------------------------------------------------------------------- #
@dataclass
class _LevelState:
    """Per-level inference state."""

    enc_cache: Optional[KVCache] = None
    #: absolute unit index of the next write -- drives RoPE
    enc_pos: int = 0
    #: buffer index of the next write; equals ``enc_pos`` unless the cache
    #: slides, then saturates at ``enc_window``
    enc_fill: int = 0
    #: cache capacity in units; ``None`` == unbounded.  When set, ``_enc_step``
    #: tiles: it recycles from position 0 at fill (O(1) footprint in T).
    enc_capacity: Optional[int] = None
    #: sliding-history units for ``encoder_window`` models.  Mutually exclusive
    #: with ``enc_capacity``; sliding is bit-exact with training, tiling is not.
    enc_window: Optional[int] = None

    #: rolling KV cache for a ``decoder_stream`` level, one per sequence,
    #: bounded at ``dec_window + dec_block`` entries (O(1) in T)
    dec_cache: Optional[KVCache] = None
    #: absolute stream position of the next write (drives RoPE)
    dec_pos: int = 0
    #: buffer index of the next write; saturates at ``dec_window``
    dec_fill: int = 0
    dec_window: Optional[int] = None
    #: largest block this cache can absorb before it must roll
    dec_block: int = 1
    #: held-back last content position of the emitted group; written with the
    #: next group's conditioning vectors, keeping decoder calls at C_l/group
    dec_pending: Optional[torch.Tensor] = None


class ScalaGenerator:
    """Stateful hierarchical decoder.

    Usage::

        gen = ScalaGenerator(model, device)
        out = gen.generate(prompt_ids, GenerationConfig(mode="recgen"))
    """

    def __init__(self, model: ScalaForCausalLM, device: torch.device | str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 enc_window_groups: int = DEFAULT_ENC_WINDOW_GROUPS,
                 speculative: bool = True):
        self.model = model.eval()
        self.cfg = model.cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.L = model.n_levels
        #: draft a chunk and verify in one pass; inert unless the model has an
        #: MTP head and the request is greedy -- see ``_can_speculate``
        self.speculative = speculative
        self.spec_stats = {"chunks": 0, "drafted": 0, "accepted": 0,
                           "dec_calls_l1": 0}
        #: blocks of ``C_{l+1}`` units a windowed lower encoder may read
        self.enc_window_groups = max(1, int(enc_window_groups))
        self.stats: dict[str, float] = {}
        #: top-level conditioning state per meta-group, recorded by
        #: ``forced_logits``
        self.top_states: list[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # cache management
    # ------------------------------------------------------------------ #
    def _window_units(self, l: int, proto: Protocol) -> Optional[int]:
        """Units level ``l``'s encoder cache may hold; ``None`` == unbounded.

        The top level is never bounded: it is the one stream that grows.
        """
        lvl = self.model.levels[l - 1]
        if lvl.encoder_window:
            # sliding-window models use `_enc_step`'s sliding branch, not this
            return None
        block = lvl.encoder_block
        if block:
            # block-local: nothing outside the current meta-group is readable,
            # so the cache holds one group and is rewritten at every boundary
            return block
        if proto.windowed and l != self.L:
            # C_{l+1} level-l units = one meta-group, the natural history unit
            return self.cfg.levels[l].chunk_size * self.enc_window_groups
        return None

    def _alloc_caches(self, batch: int, max_tokens: int, proto: Protocol) -> None:
        cfg = self.cfg
        self.state = [_LevelState() for _ in range(self.L + 1)]
        for l in range(1, self.L + 1):
            # `recgen_paper` and `chunkgen` only ever run the *top* encoder
            if not proto.lower_encoder and l != self.L:
                continue
            lvl = self.model.levels[l - 1]
            cap = self._window_units(l, proto)
            win = lvl.encoder_window if l != self.L else None
            if win:
                # `win` behind the write head plus room for the written block;
                # `_enc_step` rolls back to `win` after each write, so buffer-
                # index deltas equal position deltas for the trained mask
                units = win + max(1, cfg.levels[l].chunk_size)
            else:
                units = cap if cap else max_tokens // cfg.cumulative_chunk(l) + 2
            self.state[l].enc_cache = lvl.encoder.alloc_cache(
                batch, units, self.device, self.dtype
            )
            self.state[l].enc_pos = 0
            self.state[l].enc_fill = 0
            self.state[l].enc_capacity = cap
            self.state[l].enc_window = win

        # `dec_block` is both the prefill write size and the headroom above the
        # window, so a write can never overrun before the roll
        for l in range(1, self.L + 1):
            lvl = self.model.levels[l - 1]
            if not lvl.stream:
                continue
            blk = max(lvl.width + lvl.chunk, 64)
            st = self.state[l]
            st.dec_cache = lvl.decoder.alloc_cache(
                batch, lvl.stream + blk, self.device, self.dtype)
            st.dec_pos = st.dec_fill = 0
            st.dec_window = lvl.stream
            st.dec_block = blk
            st.dec_pending = None

    def cache_bytes(self, decoders: bool = True) -> int:
        """Bytes held by the KV caches.

        ``KVCache`` stores per-head K/V; ``LatentKVCache`` stores the MLA
        latent plus the decoupled RoPE key.  ``decoders`` includes the
        streaming decoders' rolling caches.
        """
        n = 0
        for st in self.state:
            caches = [st.enc_cache]
            if decoders:
                caches.append(st.dec_cache)
            for c in caches:
                if c is None:
                    continue
                tensors = (c.c + c.k_rope) if isinstance(c, LatentKVCache) \
                    else (c.k + c.v)
                for t in tensors:
                    n += t.numel() * t.element_size()
        return n

    # ------------------------------------------------------------------ #
    def _enc_step(self, l: int, a: torch.Tensor) -> torch.Tensor:
        """Push chunk summaries ``a`` (B, n, D_l) through encoder l with cache.

        A bounded (tiled) level recycles its buffer from position 0 at fill,
        so a unit sees 0..``enc_capacity`` units of history.  Rewinding is
        exact: RoPE is relative, so a rewound window scores the same as at
        its absolute offsets.
        """
        st = self.state[l]
        stack = self.model.levels[l - 1].encoder

        if st.enc_window:
            # sliding: write index (enc_fill) != RoPE offset (enc_pos);
            # roll_left keeps baked phases, so the trained window mask holds
            w, block = st.enc_window, max(1, self.cfg.levels[l].chunk_size)
            outs = []
            for i in range(0, a.shape[1], block):
                blk = a[:, i : i + block]
                outs.append(stack(blk, cache=st.enc_cache, pos_offset=st.enc_pos,
                                  cache_start=st.enc_fill))
                st.enc_pos += blk.shape[1]
                st.enc_fill += blk.shape[1]
                if st.enc_fill > w:
                    st.enc_cache.roll_left(st.enc_fill - w)
                    st.enc_fill = w
            return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

        cap = st.enc_capacity
        if cap is None:
            out = stack(a, cache=st.enc_cache, pos_offset=st.enc_pos)
            st.enc_pos += a.shape[1]
            return out

        outs = []
        for i in range(0, a.shape[1], cap):
            blk = a[:, i : i + cap]
            if st.enc_pos + blk.shape[1] > cap:
                st.enc_pos = 0
            outs.append(stack(blk, cache=st.enc_cache, pos_offset=st.enc_pos))
            st.enc_pos += blk.shape[1]
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]

    # ------------------------------------------------------------------ #
    def _dec_step(self, l: int, x: torch.Tensor) -> torch.Tensor:
        """Append ``x`` to level ``l``'s decoder stream and return its output.

        Write index (``dec_fill``) != RoPE offset (``dec_pos``); ``roll_left``
        keeps baked phases, so the trained window mask holds.  The roll is
        lazy but exact: the buffer may run to ``dec_window + dec_block``, and
        entries past the window are masked out either way.
        """
        st = self.state[l]
        n = x.shape[1]
        if l == 1:
            self.spec_stats["dec_calls_l1"] += 1
        if st.dec_fill + n > st.dec_window + st.dec_block:
            st.dec_cache.roll_left(st.dec_fill - st.dec_window)
            st.dec_fill = st.dec_window
        out = self.model.levels[l - 1].decoder(
            x, cache=st.dec_cache, pos_offset=st.dec_pos,
            cache_start=st.dec_fill)
        st.dec_pos += n
        st.dec_fill += n
        return out

    # ------------------------------------------------------------------ #
    # the recursive core
    # ------------------------------------------------------------------ #
    def _emit_group(self, l: int, cond: torch.Tensor, proto: Protocol,
                    ctx: "_GenContext",
                    top: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Produce the C_l level-(l-1) units under one level-l state.

        cond: (B, D_l) -> (B, C_l, D_{l-1}), the units handed upward (folded
        through the level-l chunker).  What the level-l decoder consumes next
        is chosen separately (``proto.content``).  ``top`` is X^(L)_{g-1},
        threaded down for ``global_skip``.
        """
        lvl = self.model.levels[l - 1]
        R, C = lvl.width, lvl.chunk
        B = cond.shape[0]

        if l >= 2 and self.model.levels[l - 2].encoder_block:
            # block-local encoder cannot read across the meta-group boundary;
            # rewinding is exact under relative RoPE
            st = self.state[l - 1]
            if st.enc_cache is not None:
                st.enc_pos = 0

        skip = None if lvl.d_skip == 0 else top[:, None]
        u = lvl.converter(cond[:, None], skip)[:, 0]           # (B, R, D_below)
        pre = R + lvl.lookback * C
        cache = None
        if lvl.stream:
            # pending last content position enters the stream first, then this
            # group's conditioning vectors -- keeps the stream contiguous
            st = self.state[l]
            prefix = u if st.dec_pending is None else \
                torch.cat([st.dec_pending, u], dim=1)
            st.dec_pending = None
            h = self._dec_step(l, prefix)[:, -1]               # slot 0
        else:
            cache = lvl.decoder.alloc_cache(B, pre + C, self.device, self.dtype)
            prefix = u
            if lvl.lookback:
                # `lookback * C` preceding content vectors, from a rolling
                # buffer; `start_content` before the first chunk
                prev = ctx.recent_content(lvl.lookback * C, B, lvl)
                prefix = torch.cat([u, prev], dim=1)
            h = lvl.decoder(prefix, cache=cache, pos_offset=0)[:, -1]  # slot 0

        units: list[torch.Tensor] = []
        j0 = 0
        if self._can_speculate(ctx, l, lvl):
            embs = self._speculate_chunk(h, ctx, C)
            units.extend(embs)
            j0 = len(embs)
            if j0 < C:
                # resume the sequential path: write the last known slot and
                # take the state that predicts the next one
                h = self._dec_step(l, embs[-1][:, None])[:, -1]
            else:
                self.state[l].dec_pending = embs[-1][:, None]

        for j in range(j0, C):
            if l == 1:
                content = up = self._emit_token(h, ctx)
            else:
                sub = self._emit_group(l - 1, h, proto, ctx, top)
                # summaries: xhat = `h` (computed before `sub` existed),
                # chunker = linear over `sub`, encoder = what training used
                summ = {"xhat": h}
                if proto.lower_encoder or "chunker" in (proto.content, proto.up):
                    a = self.model.levels[l - 2].chunker(sub)  # (B, 1, D_{l-1})
                    summ["chunker"] = a[:, 0]
                if proto.lower_encoder:
                    summ["encoder"] = self._enc_step(l - 1, a)[:, 0]
                content, up = summ[proto.content], summ[proto.up]
            units.append(up)
            if lvl.lookback:
                ctx.push_content(content)
            if lvl.stream:
                if j + 1 < C:
                    h = self._dec_step(l, content[:, None])[:, -1]
                else:
                    self.state[l].dec_pending = content[:, None]
            elif j + 1 < C:
                h = lvl.decoder(content[:, None], cache=cache,
                                pos_offset=pre + j)[:, -1]
            if ctx.finished_all():
                # pad the rest of the group so shapes stay rectangular
                while len(units) < C:
                    units.append(torch.zeros_like(up))
                break
        return torch.stack(units, dim=1)

    # ------------------------------------------------------------------ #
    # chunk-level self-speculative decoding
    # ------------------------------------------------------------------ #
    def _can_speculate(self, ctx: "_GenContext", l: int, lvl) -> bool:
        """Whether this chunk may be drafted.  All conditions are correctness
        requirements: streaming level-1 cache (rejected writes are undone by
        decrementing two counters); MTP chain present and deep enough (it is
        the drafter); greedy with no repetition penalty (exact-match
        acceptance reproduces greedy only); no forced tokens or logit capture
        (the equivalence harness must step one position at a time)."""
        m = self.model
        return (self.speculative and l == 1 and bool(lvl.stream)
                and lvl.chunk > 1 and m.mtp is not None
                and len(m.mtp) >= lvl.chunk - 1
                and ctx.cfg.greedy and ctx.cfg.repetition_penalty == 1.0
                and not ctx.capture_logits
                and ctx.forced_pos >= ctx.forced.shape[1])

    def _head(self, h: torch.Tensor) -> torch.Tensor:
        lg = self.model.lm_head(self.model.final_norm(h))
        cap = self.model.cfg.logit_softcap
        return torch.tanh(lg / cap) * cap if cap else lg

    def _embed(self, tok: torch.Tensor) -> torch.Tensor:
        e = self.model.embed(tok)
        if self.model.cfg.scale_embeddings:
            e = e * math.sqrt(self.model.cfg.d_token)
        return e

    def _speculate_chunk(self, h: torch.Tensor, ctx: "_GenContext",
                         C: int) -> list[torch.Tensor]:
        """Draft a chunk via the MTP chain, verify in one causal decoder pass
        (writing slots 0..C-2 yields the true slots 1..C-1 at once).

        Contract: on return the cache holds writes for slots 0..len-2 and slot
        len-1 is unwritten -- the state the sequential path expects.
        Acceptance is all-rows: a draft counts only if every batch row
        matches, since one shared cache cannot advance rows unevenly.
        """
        m = self.model
        # -- slot 0 is real ------------------------------------------------ #
        t0 = torch.where(ctx.done, ctx.pad_id, self._head(h).argmax(-1))
        ctx.record(t0)
        e0 = self._embed(t0)
        embs = [e0]

        # -- draft slots 1 .. C-1 through the MTP chain --------------------- #
        drafts, hd, e = [], h, e0
        for k in range(C - 1):
            hd = m.mtp[k](hd, e)
            d = self._head(hd).argmax(-1)
            drafts.append(d)
            e = self._embed(d)

        # -- verify all of them in ONE decoder pass ------------------------- #
        ver = torch.stack([e0] + [self._embed(d) for d in drafts[:-1]], dim=1)
        truth = self._head(self._dec_step(1, ver)).argmax(-1)      # (B, C-1)
        match = (torch.stack(drafts, dim=1) == truth).all(dim=0)   # (C-1,)
        n_ok = int(torch.cumprod(match.int(), 0).sum())            # leading run

        self.spec_stats["chunks"] += 1
        self.spec_stats["drafted"] += C - 1
        self.spec_stats["accepted"] += n_ok

        # un-write rejects: C-1 went in, keep slots 0..min(n_ok, C-2)
        keep = min(n_ok, C - 2) + 1
        st = self.state[1]
        st.dec_pos -= (C - 1) - keep
        st.dec_fill -= (C - 1) - keep

        for k in range(n_ok):
            tok = torch.where(ctx.done, ctx.pad_id, drafts[k])
            ctx.record(tok)
            embs.append(self._embed(tok))
        if n_ok < C - 1:
            # first rejected slot is corrected from the verify pass
            tok = torch.where(ctx.done, ctx.pad_id, truth[:, n_ok])
            ctx.record(tok)
            embs.append(self._embed(tok))
        return embs

    def _emit_token(self, h: torch.Tensor, ctx: "_GenContext") -> torch.Tensor:
        """Sample (or teacher-force) one token from a level-0 decoder state."""
        model = self.model
        forced = ctx.next_forced()
        if forced is not None and not ctx.capture_logits:
            tok = forced
        else:
            logits = model.lm_head(model.final_norm(h))
            if model.cfg.logit_softcap:
                cap = model.cfg.logit_softcap
                logits = torch.tanh(logits / cap) * cap
            if ctx.capture_logits:
                ctx.logits.append(logits)
            if forced is not None:
                tok = forced
            else:
                tok = sample_from_logits(logits, ctx.cfg, ctx.history(),
                                         ctx.generator)
                tok = torch.where(ctx.done, ctx.pad_id, tok)
                ctx.record(tok)
        emb = model.embed(tok)
        if model.cfg.scale_embeddings:
            emb = emb * math.sqrt(model.cfg.d_token)
        return emb

    # ------------------------------------------------------------------ #
    # prefill
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _prefill(self, prompt: torch.Tensor, proto: Protocol) -> tuple[torch.Tensor, list[int]]:
        """Encode the aligned prefix of ``prompt``; return (x_top, remainder)."""
        cfg = self.cfg
        cp = cfg.chunk_product
        B, P = prompt.shape
        n_full = (P // cp) * cp

        self._tail_content = []
        if n_full == 0:
            # no complete meta-context yet: start from the learned latent
            top = self.model.levels[-1].start_latent
            x_top = top.to(self.dtype).view(1, -1).expand(B, -1).contiguous()
            return x_top, prompt

        x = self.model.embed(prompt[:, :n_full])
        if cfg.scale_embeddings:
            x = x * math.sqrt(cfg.d_token)
        x = x.to(self.dtype)
        # seed `decoder_lookback` with the prompt's last content vectors so
        # the first generated chunk matches training across the prefill seam
        k = self.model.levels[0].lookback * self.model.levels[0].chunk
        self._tail_content = [x[:, i] for i in range(max(x.shape[1] - k, 0),
                                                     x.shape[1])] if k else []

        enc_states = [x]
        for l in range(1, self.L + 1):
            lvl = self.model.levels[l - 1]
            a = lvl.chunker(x)
            st = self.state[l]
            if (not proto.lower_encoder and l != self.L) or lvl.encoder_block:
                # single batched pass, no cache: recgen_paper/chunkgen keep no
                # lower cache; a block-local cache holds one group and the
                # aligned prefix ends on a block boundary anyway
                x = lvl.encoder(a)
            elif st.enc_capacity:
                # prefill stays exact (full encode); only the cache is bounded.
                # Refill just the partial tile -- seeding a full window would
                # be wiped by the first generated unit.
                x = lvl.encoder(a)
                rem = a.shape[1] % st.enc_capacity
                st.enc_pos = 0
                if rem:
                    self._enc_step(l, a[:, -rem:])
            else:
                x = self._enc_step(l, a)
            enc_states.append(x)

        if any(lvl.stream for lvl in self.model.levels):
            self._prefill_decoders(enc_states)
        return x[:, -1], prompt[:, n_full:]

    # ------------------------------------------------------------------ #
    def _prefill_decoders(self, enc: list[torch.Tensor]) -> None:
        """Replay the prompt through every streaming decoder.

        A streaming decoder's window crosses the prompt boundary, so the
        prompt's stream positions must already be in the cache; block-layout
        decoders need no prefill.  Non-streaming levels still run: the level
        below needs their output as conditioning.
        """
        model, L = self.model, self.L
        B = enc[0].shape[0]
        top = (model.levels[-1].shift_cond(enc[L]) if model.cfg.global_skip
               else None)
        cond = enc[L]
        for l in range(L, 0, -1):
            lvl = model.levels[l - 1]
            skip = None
            if top is not None and l < L:
                skip = top.repeat_interleave(enc[l].shape[1] // top.shape[1],
                                             dim=1)
            if not lvl.stream:
                cond = lvl.decode(cond, enc[l - 1], shift=(l == L), skip=skip)
                continue
            R, C = lvl.width, lvl.chunk
            src = lvl.shift_cond(cond) if l == L else cond
            M = src.shape[1]
            u = lvl.converter(src, skip)                  # (B, M, R, D_below)
            c = enc[l - 1].view(B, M, C, lvl.d_below)
            seq = torch.cat([u, c], dim=2).reshape(B, M * (R + C), lvl.d_below)
            st = self.state[l]
            outs = [self._dec_step(l, seq[:, i : i + st.dec_block])
                    for i in range(0, seq.shape[1], st.dec_block)]
            out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
            # prompt's final content position is already in the stream, so
            # nothing is held pending
            st.dec_pending = None
            cond = out.view(B, M, R + C, lvl.d_below)[:, :, R - 1 : R - 1 + C] \
                      .reshape(B, M * C, lvl.d_below)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forced_logits(self, tokens: torch.Tensor, mode: Mode = "hiergen",
                      prefix_meta: int = 1) -> torch.Tensor:
        """Replay ``tokens`` through the incremental path; return the logits
        for every position after the first ``prefix_meta`` meta-contexts.
        Equivalence harness: HierGen must match the teacher-forced forward.
        """
        tokens = tokens.to(self.device)
        if tokens.ndim == 1:
            tokens = tokens[None]
        B, T = tokens.shape
        cp = self.cfg.chunk_product
        split = prefix_meta * cp
        proto = PROTOCOLS[mode]

        self._alloc_caches(B, T + cp, proto)
        x_top, remainder = self._prefill(tokens[:, :split], proto)

        cfg = GenerationConfig(max_new_tokens=0, mode=mode)
        ctx = _GenContext(cfg, B, self.device, tokens[:, split:], None)
        ctx.content = list(self._tail_content)
        ctx.capture_logits = True
        self.top_states = [x_top]

        while not ctx.finished_all():
            sub = self._emit_group(self.L, x_top, proto, ctx, x_top)
            a = self.model.levels[-1].chunker(sub)
            x_top = self._enc_step(self.L, a)[:, 0]
            self.top_states.append(x_top)
        return torch.stack(ctx.logits, dim=1)[:, : T - split]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, prompt: torch.Tensor,
                 cfg: GenerationConfig = GenerationConfig()) -> torch.Tensor:
        model_cfg = self.cfg
        prompt = prompt.to(self.device)
        if prompt.ndim == 1:
            prompt = prompt[None]
        B, P = prompt.shape

        gen = None
        if cfg.seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(cfg.seed)

        proto = PROTOCOLS[cfg.mode]
        max_tokens = P + cfg.max_new_tokens + model_cfg.chunk_product
        self._alloc_caches(B, max_tokens, proto)

        t0 = time.perf_counter()
        x_top, remainder = self._prefill(prompt, proto)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_prefill = time.perf_counter() - t0

        ctx = _GenContext(cfg, B, self.device, remainder, gen,
                          pad_id=cfg.eos_token_id or 0)
        ctx.content = list(self._tail_content)

        t0 = time.perf_counter()
        while not ctx.finished_all():
            sub = self._emit_group(self.L, x_top, proto, ctx, x_top)  # (B, C_L, D_{L-1})
            a = self.model.levels[-1].chunker(sub)                 # (B, 1, D_L)
            x_top = self._enc_step(self.L, a)[:, 0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_decode = time.perf_counter() - t0

        out = ctx.stack()
        n_new = out.shape[1]
        self.stats = {
            "prefill_s": t_prefill,
            "decode_s": t_decode,
            "prompt_tokens": P,
            "new_tokens": n_new,
            "decode_tok_per_s": (B * n_new) / max(t_decode, 1e-9),
            "kv_cache_bytes": float(self.cache_bytes()),
            "kv_cache_bytes_per_token": self.cache_bytes() / max(B * (P + n_new), 1),
        }
        s = self.spec_stats
        if s["chunks"]:
            self.stats["spec_chunks"] = s["chunks"]
            self.stats["spec_accept_rate"] = s["accepted"] / max(s["drafted"], 1)
        # level-1 decoder runs per token; sequential decoding costs exactly
        # C_1 calls per chunk
        if s["chunks"] or self.spec_stats["dec_calls_l1"]:
            self.stats["dec_calls_per_chunk"] = (
                self.spec_stats["dec_calls_l1"]
                / max(n_new / self.cfg.levels[0].chunk_size, 1))
        return torch.cat([prompt, out[:, : cfg.max_new_tokens]], dim=1)


# --------------------------------------------------------------------------- #
class _GenContext:
    """Tracks emitted tokens, teacher-forced prompt remainder, and stop state."""

    def __init__(self, cfg: GenerationConfig, batch: int, device: torch.device,
                 forced: torch.Tensor, generator: Optional[torch.Generator],
                 pad_id: int = 0):
        self.cfg = cfg
        self.device = device
        self.generator = generator
        self.forced = forced                     # (B, n_remaining_prompt)
        self.forced_pos = 0
        self.tokens: list[torch.Tensor] = []
        self.done = torch.zeros(batch, dtype=torch.bool, device=device)
        self.pad_id = torch.full((batch,), pad_id, dtype=torch.long, device=device)
        #: set by the equivalence tests -- keeps every step's logits,
        #: including teacher-forced prompt positions
        self.capture_logits = False
        self.logits: list[torch.Tensor] = []
        #: rolling buffer of recent level-0 content for ``decoder_lookback``;
        #: only the last ``lookback * C_1`` entries are read
        self.content: list[torch.Tensor] = []

    def push_content(self, c: torch.Tensor) -> None:
        self.content.append(c)
        if len(self.content) > 64:
            del self.content[:-32]

    def recent_content(self, n: int, batch: int, lvl) -> torch.Tensor:
        """Last ``n`` emitted content vectors, left-padded with the learned
        ``start_content`` the training forward uses."""
        have = self.content[-n:]
        start = lvl.start_content.to(have[0].dtype) if have else lvl.start_content
        pad = n - len(have)
        out = []
        if pad:
            out.append(start[:pad].unsqueeze(0).expand(batch, pad, -1))
        if have:
            out.append(torch.stack(have, dim=1))
        return torch.cat(out, dim=1) if len(out) > 1 else out[0]

    def next_forced(self) -> Optional[torch.Tensor]:
        if self.forced_pos < self.forced.shape[1]:
            tok = self.forced[:, self.forced_pos]
            self.forced_pos += 1
            return tok
        return None

    def record(self, tok: torch.Tensor) -> None:
        self.tokens.append(tok)
        if self.cfg.eos_token_id is not None:
            self.done |= tok.eq(self.cfg.eos_token_id)

    def history(self) -> Optional[torch.Tensor]:
        if self.cfg.repetition_penalty == 1.0 or not self.tokens:
            return None
        return torch.stack(self.tokens, dim=1)

    def finished_all(self) -> bool:
        if self.forced_pos < self.forced.shape[1]:
            return False       # still replaying the prompt remainder
        if len(self.tokens) >= self.cfg.max_new_tokens:
            return True
        return bool(self.done.all())

    def stack(self) -> torch.Tensor:
        if not self.tokens:
            return torch.zeros(self.done.shape[0], 0, dtype=torch.long,
                               device=self.device)
        return torch.stack(self.tokens, dim=1)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model: ScalaForCausalLM, prompt: torch.Tensor,
             cfg: GenerationConfig = GenerationConfig(),
             device: torch.device | str = "cuda",
             dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return ScalaGenerator(model, device, dtype).generate(prompt, cfg)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def teacher_forced_logits(model: ScalaForCausalLM,
                          tokens: torch.Tensor) -> torch.Tensor:
    """Single full forward; reference for the HierGen equivalence tests."""
    return model(tokens, return_logits=True).logits
