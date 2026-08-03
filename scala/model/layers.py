"""Transformer primitives: RMSNorm, RotaryEmbedding, Attention (GQA or MLA,
QK-Norm, sliding/blocked causal masks, learned sinks), SwiGLU, pre-norm
TransformerBlock with a dense or MoE FFN, TransformerStack, and the KV caches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

from .config import AttentionConfig, StackConfig
from .moe import MoELayer

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "Attention",
    "SwiGLU",
    "TransformerBlock",
    "TransformerStack",
    "KVCache",
]


# --------------------------------------------------------------------------- #
# Norm
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # match the weight dtype to the (autocast) input so the fused kernel is
        # reachable; rms_norm still accumulates the mean-square in fp32.
        w = self.weight
        if w is not None and w.dtype != x.dtype:
            w = w.to(x.dtype)
        return F.rms_norm(x, x.shape[-1:], w, self.eps)

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)


# --------------------------------------------------------------------------- #
# Rotary position embeddings
# --------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    """RoPE with partial rotation and optional YaRN context extension.

    ``partial`` < 1 leaves the tail of each head un-rotated (NoPE tail).
    """

    def __init__(
        self,
        head_dim: int,
        theta: float = 1_000_000.0,
        partial: float = 1.0,
        max_seq_len: int = 8192,
        yarn_factor: float = 1.0,
        yarn_original_len: int = 4096,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
    ):
        super().__init__()
        rot = int(head_dim * partial)
        rot -= rot % 2  # must be even
        self.rot_dim = rot
        self.head_dim = head_dim
        self.theta = theta
        self.yarn_factor = yarn_factor

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot)
        )
        if yarn_factor > 1.0:
            inv_freq, self.attn_scale = self._yarn(
                inv_freq, rot, yarn_factor, yarn_original_len,
                yarn_beta_fast, yarn_beta_slow, theta,
            )
        else:
            self.attn_scale = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0
        self._build_cache(max_seq_len)

    @staticmethod
    def _yarn(inv_freq, rot, factor, orig_len, beta_fast, beta_slow, theta):
        """YaRN "NTK-by-parts" interpolation."""

        def _dim_for_wavelen(n_rot: int) -> float:
            return (
                rot * math.log(orig_len / (n_rot * 2 * math.pi)) / (2 * math.log(theta))
            )

        low = math.floor(_dim_for_wavelen(beta_fast))
        high = math.ceil(_dim_for_wavelen(beta_slow))
        low, high = max(low, 0), min(high, rot // 2 - 1)
        ramp = torch.arange(rot // 2, dtype=torch.float32)
        ramp = ((ramp - low) / max(high - low, 1e-3)).clamp(0, 1)
        # mask == 1 -> keep original (high frequency), 0 -> full interpolation
        mask = 1.0 - ramp
        inv_freq = inv_freq / factor * (1 - mask) + inv_freq * mask
        attn_scale = 0.1 * math.log(factor) + 1.0
        return inv_freq, attn_scale

    #: Above this length cos/sin are computed per call instead of cached: the
    #: table rebuilds from zero as it grows (unbounded memory at 1M positions),
    #: and fp32 phases quantize to ~0.1 rad near pos ~1e6, so the bypass uses
    #: fp64.
    MAX_TABLE_LEN = 131_072

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self._cache_len:
            return
        # grow geometrically (capped) so a streaming caller pays O(log T) rebuilds
        seq_len = min(max(seq_len, 2 * self._cache_len), self.MAX_TABLE_LEN)
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
        self._cache_len = seq_len

    def _rows_beyond_table(self, seq_len: int, offset: int):
        t = torch.arange(offset, offset + seq_len, dtype=torch.float64,
                         device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq.double())
        return freqs.cos().float(), freqs.sin().float()

    def forward(self, seq_len: int, offset: int = 0, device=None):
        """cos/sin for absolute positions ``[offset, offset+seq_len)``.

        The cached-vs-per-call-fp64 choice is made per ROW, from the row's own
        absolute position, never from this call's ``(offset, seq_len)`` as a
        whole. Two callers requesting the same absolute positions in
        differently sized chunks (e.g. ``TiledScorer``'s small per-block calls
        vs. one ``model(tokens)`` call spanning a whole level) must get
        bit-identical rows; keying the split on the call's own span instead of
        each row's position previously broke that for any level whose
        position range straddled ``MAX_TABLE_LEN``.
        """
        hi = offset + seq_len
        if hi <= self.MAX_TABLE_LEN:
            self._build_cache(hi)
            cos = self.cos_cached[offset:hi]
            sin = self.sin_cached[offset:hi]
        elif offset >= self.MAX_TABLE_LEN:
            cos, sin = self._rows_beyond_table(seq_len, offset)
        else:
            # straddles the boundary: split so the cached rows are exactly
            # the ones any call touching only positions < MAX_TABLE_LEN would
            # get, and the fp64 rows are exactly the ones any call touching
            # only positions >= MAX_TABLE_LEN would get
            n_cached = self.MAX_TABLE_LEN - offset
            self._build_cache(self.MAX_TABLE_LEN)
            cos_lo = self.cos_cached[offset:self.MAX_TABLE_LEN]
            sin_lo = self.sin_cached[offset:self.MAX_TABLE_LEN]
            cos_hi, sin_hi = self._rows_beyond_table(seq_len - n_cached, self.MAX_TABLE_LEN)
            cos = torch.cat([cos_lo, cos_hi], dim=0)
            sin = torch.cat([sin_lo, sin_hi], dim=0)
        if device is not None and cos.device != device:
            cos, sin = cos.to(device), sin.to(device)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot_dim: int):
    """x: (B, H, S, Dh).  cos/sin: (S, rot_dim // 2).

    ``rot_dim == 0`` is the NoPE path: ``x`` is returned untouched and
    ``cos``/``sin`` may be ``None``.
    """
    if rot_dim == 0:
        return x
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x1, x2 = x_rot.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(x1.dtype)
    sin = sin[None, None, :, :].to(x1.dtype)
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)
    return torch.cat([out, x_pass], dim=-1) if x_pass.numel() else out


# --------------------------------------------------------------------------- #
# KV cache
# --------------------------------------------------------------------------- #
@dataclass
class KVCache:
    """Per-layer KV cache with a fixed capacity, ring-free layout.

    Keys and values have separate head dims: under MLA a key carries
    ``qk_nope_head_dim + qk_rope_head_dim`` channels, a value ``v_head_dim``.
    """

    k: list[torch.Tensor]
    v: list[torch.Tensor]
    length: int = 0

    @classmethod
    def alloc(
        cls, n_layers: int, batch: int, n_kv_heads: int, k_head_dim: int,
        capacity: int, device, dtype, v_head_dim: int | None = None,
    ) -> "KVCache":
        v_head_dim = k_head_dim if v_head_dim is None else v_head_dim
        ks = (batch, n_kv_heads, capacity, k_head_dim)
        vs = (batch, n_kv_heads, capacity, v_head_dim)
        return cls(
            k=[torch.zeros(ks, device=device, dtype=dtype) for _ in range(n_layers)],
            v=[torch.zeros(vs, device=device, dtype=dtype) for _ in range(n_layers)],
            length=0,
        )

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor, start: int):
        n = k.shape[2]
        self.k[layer][:, :, start : start + n] = k
        self.v[layer][:, :, start : start + n] = v
        return self.k[layer][:, :, : start + n], self.v[layer][:, :, : start + n]

    def roll_left(self, n: int) -> None:
        """Drop the oldest ``n`` entries, keeping the rest contiguous (sliding
        window).  Baked-in RoPE phases stay as written; after a roll the write
        index (``cache_start``) is no longer the RoPE offset (``pos_offset``).
        """
        if n <= 0:
            return
        for t in self.k + self.v:
            t[:, :, : t.shape[2] - n] = t[:, :, n:].clone()

    def reset(self) -> None:
        self.length = 0


@dataclass
class LatentKVCache:
    """MLA cache storing the compressed latent instead of K and V.

    ``kv_lora_rank + qk_rope_head_dim`` values per unit per layer, independent
    of head count.  Reading it requires ``Attention.forward_mla_absorbed``.
    """

    c: list[torch.Tensor]        # (B, capacity, kv_lora_rank)
    k_rope: list[torch.Tensor]   # (B, capacity, qk_rope_head_dim)
    length: int = 0

    @classmethod
    def alloc(cls, n_layers: int, batch: int, kv_lora: int, qk_rope: int,
              capacity: int, device, dtype) -> "LatentKVCache":
        return cls(
            c=[torch.zeros(batch, capacity, kv_lora, device=device, dtype=dtype)
               for _ in range(n_layers)],
            k_rope=[torch.zeros(batch, capacity, qk_rope, device=device,
                                dtype=dtype) for _ in range(n_layers)],
            length=0,
        )

    def update(self, layer: int, c: torch.Tensor, k_rope: torch.Tensor,
               start: int):
        n = c.shape[1]
        self.c[layer][:, start : start + n] = c
        self.k_rope[layer][:, start : start + n] = k_rope
        return (self.c[layer][:, : start + n],
                self.k_rope[layer][:, : start + n])

    def roll_left(self, n: int) -> None:
        """Sliding-window counterpart of ``KVCache.roll_left``; the sequence
        axis is 1, not 2, because the latent is stored per unit, not per head."""
        if n <= 0:
            return
        for t in self.c + self.k_rope:
            t[:, : t.shape[1] - n] = t[:, n:].clone()

    def reset(self) -> None:
        self.length = 0


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, cfg: AttentionConfig, d_model: int, n_layers: int,
                 scale_residual_init: bool = True):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.kind = cfg.kind
        self.n_heads = cfg.n_heads
        self.window = cfg.window
        self.block = cfg.block
        #: either restriction needs an explicit mask; plain causal does not
        self._local = bool(cfg.window or cfg.block)

        if cfg.kind == "gqa":
            self.head_dim = cfg.head_dim or (d_model // cfg.n_heads)
            self.n_kv_heads = cfg.n_kv_heads
            self.n_rep = cfg.n_heads // cfg.n_kv_heads
            self.wq = nn.Linear(d_model, cfg.n_heads * self.head_dim, bias=False)
            self.wk = nn.Linear(d_model, cfg.n_kv_heads * self.head_dim, bias=False)
            self.wv = nn.Linear(d_model, cfg.n_kv_heads * self.head_dim, bias=False)
            self.wo = nn.Linear(cfg.n_heads * self.head_dim, d_model, bias=False)
            self.v_head_dim = self.head_dim
        else:  # MLA
            self.qk_nope = cfg.mla_qk_nope_head_dim
            self.qk_rope = cfg.mla_qk_rope_head_dim
            self.head_dim = self.qk_nope + self.qk_rope
            self.v_head_dim = cfg.mla_v_head_dim
            self.kv_lora = cfg.mla_kv_lora_rank
            self.q_lora = cfg.mla_q_lora_rank
            self.n_kv_heads = cfg.n_heads  # MLA is MHA after decompression

            if self.q_lora:
                self.wq_a = nn.Linear(d_model, self.q_lora, bias=False)
                self.q_norm = RMSNorm(self.q_lora, eps=1e-6)
                self.wq_b = nn.Linear(self.q_lora, cfg.n_heads * self.head_dim, bias=False)
            else:
                self.wq = nn.Linear(d_model, cfg.n_heads * self.head_dim, bias=False)
            # compressed KV latent + the decoupled RoPE key (shared across heads)
            self.wkv_a = nn.Linear(d_model, self.kv_lora + self.qk_rope, bias=False)
            self.kv_norm = RMSNorm(self.kv_lora, eps=1e-6)
            self.wkv_b = nn.Linear(
                self.kv_lora, cfg.n_heads * (self.qk_nope + self.v_head_dim), bias=False
            )
            self.wo = nn.Linear(cfg.n_heads * self.v_head_dim, d_model, bias=False)

        self.softmax_scale = cfg.softmax_scale or (self.head_dim ** -0.5)

        if cfg.qk_norm:
            self.qnorm = RMSNorm(self.head_dim, eps=1e-6)
            self.knorm = RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.qnorm = self.knorm = None

        self.sink = nn.Parameter(torch.zeros(cfg.n_heads)) if cfg.attn_sink else None
        self._absorb_cache = None
        # the folded matrices are built from the current weights, so any
        # state_dict load must invalidate them
        self.register_load_state_dict_post_hook(
            lambda mod, incompatible: mod.invalidate_absorbed()
        )

        # output gate: one sigmoid per head, driven by the input
        if cfg.output_gate:
            self.w_gate = nn.Linear(d_model, cfg.n_heads, bias=False)
            nn.init.zeros_(self.w_gate.weight)   # sigmoid(0) = 0.5, neutral start
        else:
            self.w_gate = None

        self._n_layers = n_layers
        self._scale_residual_init = scale_residual_init
        if scale_residual_init:
            with torch.no_grad():
                self.wo.weight.mul_(1.0 / math.sqrt(2 * n_layers))

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for name, p in self.named_parameters(recurse=False):
            if p.ndim >= 2:
                nn.init.normal_(p, std=p.shape[-2] ** -0.5)
            else:
                p.zero_()
        if self.w_gate is not None:
            nn.init.zeros_(self.w_gate.weight)   # sigmoid(0) = 0.5, neutral
        if self.sink is not None:
            nn.init.zeros_(self.sink)
        for m in (self.wq, self.wk, self.wv, self.wo) if self.kind == "gqa" else ():
            nn.init.normal_(m.weight, std=m.weight.shape[-1] ** -0.5)
        if self._scale_residual_init:
            self.wo.weight.mul_(1.0 / math.sqrt(2 * self._n_layers))

    # ------------------------------------------------------------------ #
    # MLA with weight absorption
    # ------------------------------------------------------------------ #
    @property
    def supports_latent_cache(self) -> bool:
        # qk_norm normalises the decompressed head vectors, so it forces K to be
        # materialised -- exactly what absorption avoids.
        return self.kind == "mla" and self.qnorm is None

    def _absorbed_weights(self):
        """W^{UK} and (W^{UV} folded into W^O), built once and cached.

        ``q_nope . (c W^{UK}) = (q_nope W^{UK,T}) . c`` keeps ``c`` compressed;
        ``sum_j a_j (c_j W^{UV}) = (sum_j a_j c_j) W^{UV}`` folds W^{UV} into
        W^O.  Shapes: (H, qk_nope, kv_lora) and (H, kv_lora, d_model).
        """
        if getattr(self, "_absorb_cache", None) is not None:
            return self._absorb_cache
        # wkv_b: (n_heads * (qk_nope + v_head_dim), kv_lora)
        w = self.wkv_b.weight.view(self.n_heads, self.qk_nope + self.v_head_dim,
                                   self.kv_lora)
        w_uk = w[:, : self.qk_nope, :]            # (H, qk_nope, kv_lora)
        w_uv = w[:, self.qk_nope :, :]            # (H, v_head_dim, kv_lora)
        # wo: (d_model, n_heads * v_head_dim) -> (H, v_head_dim, d_model)
        wo = self.wo.weight.t().view(self.n_heads, self.v_head_dim, -1)
        # (H, kv_lora, d_model)
        wo_absorbed = torch.einsum("hvl,hvd->hld", w_uv, wo)
        self._absorb_cache = (w_uk, wo_absorbed)
        return self._absorb_cache

    def invalidate_absorbed(self) -> None:
        """Call after any weight update; the folded matrices go stale."""
        self._absorb_cache = None

    def forward_mla_absorbed(self, x, cos, sin, cache: "LatentKVCache",
                             layer_idx: int, cache_start: int) -> torch.Tensor:
        B, S, _ = x.shape
        w_uk, wo_absorbed = self._absorbed_weights()

        q = self.wq_b(self.q_norm(self.wq_a(x))) if self.q_lora else self.wq(x)
        q = q.view(B, S, self.n_heads, self.head_dim)
        q_nope, q_rope = q.split([self.qk_nope, self.qk_rope], dim=-1)

        kv = self.wkv_a(x)
        c, k_rope = kv.split([self.kv_lora, self.qk_rope], dim=-1)
        c = self.kv_norm(c)

        q_rope = apply_rope(q_rope.transpose(1, 2), cos, sin, self.qk_rope)
        k_rope = apply_rope(k_rope[:, :, None].transpose(1, 2), cos, sin,
                            self.qk_rope).transpose(1, 2)[:, :, 0]

        c_all, kr_all = cache.update(layer_idx, c, k_rope, cache_start)

        # push the query into latent space: (B, H, S, kv_lora)
        q_lat = torch.einsum("bshn,hnl->bhsl", q_nope, w_uk.to(q_nope.dtype))

        scores = torch.einsum("bhsl,btl->bhst", q_lat, c_all)
        if self.qk_rope:
            # under NoPE this contracts over zero channels, so skipping it is exact
            scores = scores + torch.einsum("bhsd,btd->bhst", q_rope, kr_all)
        scores = scores * self.softmax_scale

        t_total = c_all.shape[1]
        mask = None
        if self._local:
            mask = self._build_mask(S, t_total, x.device)
        elif S > 1:
            qi = torch.arange(t_total - S, t_total, device=x.device)[:, None]
            ki = torch.arange(t_total, device=x.device)[None, :]
            mask = (ki <= qi)[None, None]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        if self.sink is not None:
            # one extra logit per head owning no value vector, so a head can send
            # probability mass nowhere; costs one concatenated column
            sink = self.sink.view(1, -1, 1, 1).float().expand(
                scores.shape[0], -1, S, 1)
            attn = torch.softmax(
                torch.cat([sink, scores.float()], dim=-1), dim=-1
            )[..., 1:].to(c_all.dtype)
        else:
            attn = scores.float().softmax(-1).to(c_all.dtype)
        o_lat = torch.einsum("bhst,btl->bhsl", attn, c_all)   # (B,H,S,kv_lora)

        # the gate is one scalar per (head, position) and everything downstream
        # (W^{UV}, W^O, folded into `wo_absorbed`) is linear, so scaling the
        # latent equals scaling the decompressed head output
        if self.w_gate is not None:
            g = (2.0 * torch.sigmoid(self.w_gate(x).float())).to(o_lat.dtype)
            o_lat = o_lat * g.transpose(1, 2).unsqueeze(-1)

        return torch.einsum("bhsl,hld->bsd", o_lat, wo_absorbed.to(o_lat.dtype))

    # ------------------------------------------------------------------ #
    def _project(self, x: torch.Tensor):
        B, S, _ = x.shape
        if self.kind == "gqa":
            q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
            k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
            v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)
        else:
            if self.q_lora:
                q = self.wq_b(self.q_norm(self.wq_a(x)))
            else:
                q = self.wq(x)
            q = q.view(B, S, self.n_heads, self.head_dim)

            kv = self.wkv_a(x)
            kv_c, k_rope = kv.split([self.kv_lora, self.qk_rope], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv_c)).view(
                B, S, self.n_heads, self.qk_nope + self.v_head_dim
            )
            k_nope, v = kv.split([self.qk_nope, self.v_head_dim], dim=-1)
            # decoupled RoPE key is shared across heads
            k_rope = k_rope.view(B, S, 1, self.qk_rope).expand(B, S, self.n_heads, -1)
            k = torch.cat([k_nope, k_rope], dim=-1)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rot_dim: int,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        cache_start: int = 0,
    ) -> torch.Tensor:
        if isinstance(cache, LatentKVCache):
            return self.forward_mla_absorbed(x, cos, sin, cache, layer_idx,
                                             cache_start)
        B, S, _ = x.shape
        q, k, v = self._project(x)
        q = q.transpose(1, 2)  # (B, H, S, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qnorm is not None:
            q, k = self.qnorm(q), self.knorm(k)

        if self.kind == "mla":
            # only the last qk_rope channels are rotated
            nope, qr = q.split([self.qk_nope, self.qk_rope], dim=-1)
            knope, kr = k.split([self.qk_nope, self.qk_rope], dim=-1)
            qr = apply_rope(qr, cos, sin, self.qk_rope)
            kr = apply_rope(kr, cos, sin, self.qk_rope)
            q = torch.cat([nope, qr], dim=-1)
            k = torch.cat([knope, kr], dim=-1)
        else:
            q = apply_rope(q, cos, sin, rot_dim)
            k = apply_rope(k, cos, sin, rot_dim)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v, cache_start)

        if self.kind == "gqa" and self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        kv_len = k.shape[2]
        is_decode = S == 1 and kv_len > 1

        if self.sink is not None:
            out = self._attend_with_sink(q, k, v, kv_len, is_decode)
        else:
            mask = self._build_mask(S, kv_len, q.device) if self._local else None
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                is_causal=(mask is None and not is_decode and S > 1),
                scale=self.softmax_scale,
                enable_gqa=False,
            )
        if self.w_gate is not None:
            # 2*sigmoid so the gate is exactly 1.0 at init (w_gate starts at zero)
            g = (2.0 * torch.sigmoid(self.w_gate(x).float())).to(out.dtype)
            out = out * g.transpose(1, 2).unsqueeze(-1)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.v_head_dim)
        return self.wo(out)

    # ------------------------------------------------------------------ #
    def _build_mask(self, q_len: int, kv_len: int, device) -> torch.Tensor:
        qi = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
        ki = torch.arange(kv_len, device=device)[None, :]
        keep = ki <= qi
        if self.window:
            keep = keep & (ki > qi - self.window)
        if self.block:
            keep = keep & (ki.div(self.block, rounding_mode="floor")
                           == qi.div(self.block, rounding_mode="floor"))
        return keep[None, None]

    def _attend_with_sink(self, q, k, v, kv_len, is_decode):
        """Explicit softmax so a learned per-head sink logit can be added."""
        scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * self.softmax_scale
        S = q.shape[2]
        if not is_decode:
            mask = self._build_mask(S, kv_len, q.device) if self._local else None
            if mask is None:
                qi = torch.arange(kv_len - S, kv_len, device=q.device)[:, None]
                ki = torch.arange(kv_len, device=q.device)[None, :]
                mask = (ki <= qi)[None, None]
            scores = scores.masked_fill(~mask, float("-inf"))
        elif self._local:
            scores = scores.masked_fill(
                self._build_mask(S, kv_len, q.device).logical_not(), float("-inf")
            )
        sink = self.sink.view(1, -1, 1, 1).float().expand(scores.shape[0], -1, S, 1)
        probs = torch.softmax(torch.cat([sink, scores], dim=-1), dim=-1)[..., 1:]
        return torch.einsum("bhqk,bhkd->bhqd", probs.to(v.dtype), v)


# --------------------------------------------------------------------------- #
# Dense FFN
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, inter: int, n_layers: int = 1,
                 scale_residual_init: bool = True):
        super().__init__()
        self.w_gate = nn.Linear(d_model, inter, bias=False)
        self.w_up = nn.Linear(d_model, inter, bias=False)
        self.w_down = nn.Linear(inter, d_model, bias=False)
        if scale_residual_init:
            with torch.no_grad():
                self.w_down.weight.mul_(1.0 / math.sqrt(2 * n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# --------------------------------------------------------------------------- #
# Block / stack
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    def __init__(self, cfg: StackConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        att = cfg.attention
        if cfg.full_attn_every and att.window:
            # every full_attn_every-th layer keeps a global view, the rest window
            from dataclasses import replace

            is_global = (layer_idx + 1) % cfg.full_attn_every == 0
            att = replace(att, window=None) if is_global else att
        self.is_global_attn = att.window is None and att.block is None

        self.attn = Attention(att, cfg.d_model, cfg.n_layers,
                              cfg.scale_residual_init)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        use_moe = cfg.moe.enabled and layer_idx >= cfg.n_dense_layers
        if use_moe:
            self.ffn = MoELayer(cfg.moe, cfg.d_model, cfg.n_layers,
                                cfg.scale_residual_init)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.ffn_inter_size, cfg.n_layers,
                              cfg.scale_residual_init)
        self.is_moe = use_moe

        if cfg.learned_residual_scale:
            d = cfg.d_model
            self.res_attn = nn.Parameter(torch.ones(d))
            self.out_attn = nn.Parameter(torch.ones(d))
            self.res_ffn = nn.Parameter(torch.ones(d))
            self.out_ffn = nn.Parameter(torch.ones(d))
        else:
            self.res_attn = self.out_attn = None
            self.res_ffn = self.out_ffn = None

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Re-initialise the parameters this module owns directly.

        Meta-device construction skips every initialiser, so this replays them;
        the residual gains must come back as 1.0, not 0.
        """
        for p in (self.res_attn, self.out_attn, self.res_ffn, self.out_ffn):
            if p is not None:
                p.fill_(1.0)

    def forward(self, x, cos, sin, rot_dim, cache=None, cache_start=0):
        a = self.attn(self.attn_norm(x), cos, sin, rot_dim,
                      cache, self.layer_idx, cache_start)
        if self.res_attn is not None:
            x = x * self.res_attn.to(x.dtype) + a * self.out_attn.to(a.dtype)
        else:
            x = x + a
        f = self.ffn(self.ffn_norm(x))
        if self.res_ffn is not None:
            x = x * self.res_ffn.to(x.dtype) + f * self.out_ffn.to(f.dtype)
        else:
            x = x + f
        return x


class TransformerStack(nn.Module):
    """A causal transformer stack over an arbitrary unit sequence."""

    def __init__(self, cfg: StackConfig, max_seq_len: int, yarn_factor: float = 1.0):
        super().__init__()
        self.cfg = cfg
        head_dim = cfg.attention.resolve_head_dim(cfg.d_model)
        if cfg.attention.pos == "nope":
            # NoPE: no table, no cos/sin buffers, no `apply_rope`; the causal
            # mask is the only order signal, so no position index exists
            self.rope = None
            self.rot_dim = 0
        elif cfg.attention.kind == "mla":
            self.rope = RotaryEmbedding(
                cfg.attention.mla_qk_rope_head_dim, cfg.attention.rope_theta,
                1.0, max_seq_len, yarn_factor,
            )
            self.rot_dim = cfg.attention.mla_qk_rope_head_dim
        else:
            self.rope = RotaryEmbedding(
                head_dim, cfg.attention.rope_theta, cfg.attention.rope_partial,
                max_seq_len, yarn_factor,
            )
            self.rot_dim = self.rope.rot_dim
        self.layers = nn.ModuleList(
            TransformerBlock(cfg, i) for i in range(cfg.n_layers)
        )
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        #: which block indices recompute activations; the trainer narrows this
        #: for selective checkpointing.  ``None`` == every block.
        self.ac_layers: set[int] | None = None

    @property
    def n_kv_heads(self) -> int:
        a = self.cfg.attention
        return a.n_heads if a.kind == "mla" else a.n_kv_heads

    @property
    def head_dim(self) -> int:
        """Key/query head width."""
        return self.cfg.attention.resolve_head_dim(self.cfg.d_model)

    @property
    def v_head_dim(self) -> int:
        a = self.cfg.attention
        return a.mla_v_head_dim if a.kind == "mla" else self.head_dim

    @property
    def supports_latent_cache(self) -> bool:
        return all(l.attn.supports_latent_cache for l in self.layers)

    def alloc_cache(self, batch: int, capacity: int, device, dtype,
                    latent: bool | None = None):
        """Allocate this stack's KV cache.

        ``latent=None`` picks the compressed MLA cache whenever the stack
        supports it.
        """
        a = self.cfg.attention
        if latent is None:
            latent = self.supports_latent_cache
        if latent:
            if not self.supports_latent_cache:
                raise ValueError(
                    "latent KV cache needs MLA with qk_norm disabled "
                    "(QK-Norm requires decompressing K, which defeats it)"
                )
            return LatentKVCache.alloc(
                self.cfg.n_layers, batch, a.mla_kv_lora_rank,
                a.mla_qk_rope_head_dim, capacity, device, dtype,
            )
        return KVCache.alloc(self.cfg.n_layers, batch, self.n_kv_heads,
                             self.head_dim, capacity, device, dtype,
                             v_head_dim=self.v_head_dim)

    def forward(self, x, cache: Optional[KVCache] = None, pos_offset: int = 0,
                grad_ckpt: bool = False, cache_start: Optional[int] = None):
        """``pos_offset`` is the absolute position of ``x[0]`` and drives RoPE.

        ``cache_start`` is the buffer write index and defaults to the same
        number; the two diverge for a sliding-window cache, whose rolls put a
        token's buffer index below its position.  RoPE stays absolute, so
        relative phases survive the roll.  Under NoPE ``pos_offset`` is unused.
        """
        cos, sin = (None, None) if self.rope is None else \
            self.rope(x.shape[1], pos_offset, x.device)
        start = pos_offset if cache_start is None else cache_start
        for i, layer in enumerate(self.layers):
            recompute = (
                grad_ckpt and self.training and cache is None
                and (self.ac_layers is None or i in self.ac_layers)
            )
            if recompute:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, self.rot_dim, cache, start,
                    use_reentrant=False,
                )
            else:
                x = layer(x, cos, sin, self.rot_dim, cache, start)
        return self.norm(x)
