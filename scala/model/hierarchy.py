"""SCALA hierarchical autoregressive LM (reference: arXiv:2512.20687).
Level l: M_l = M_{l-1} / C_l units of width D_l (level 0 = tokens).  Bottom-up:
chunker + causal encoder per level.  Top-down: converter emits R_l conditioning
vectors; the decoder sees only R_l + C_l positions per group, so decode cost is
independent of T.  X_hat^(0)_i attends to t_{<i} only: no shift in the loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

from .config import LevelConfig, ScalaConfig
from .layers import RMSNorm, TransformerStack, SwiGLU
from .moe import reset_moe_aux_losses, moe_aux_losses

__all__ = ["ScalaForCausalLM", "ScalaOutput", "ContextChunker", "ContextConverter"]


# --------------------------------------------------------------------------- #
# Context Chunker   C_theta^(l)
# --------------------------------------------------------------------------- #
class ContextChunker(nn.Module):
    """Fold ``C_l`` units of width ``d_in`` into one unit of width ``d_out``."""

    def __init__(self, d_in: int, d_out: int, chunk: int, kind: str = "concat"):
        super().__init__()
        self.chunk = chunk
        self.kind = kind
        self.d_in, self.d_out = d_in, d_out
        self.norm = RMSNorm(d_in)

        if kind == "concat":
            self.proj = nn.Linear(d_in * chunk, d_out, bias=False)
            nn.init.normal_(self.proj.weight, std=(d_in * chunk) ** -0.5)
        elif kind == "conv":
            self.conv = nn.Conv1d(d_in, d_out, kernel_size=chunk, stride=chunk, bias=False)
            self.proj = nn.Linear(d_out, d_out, bias=False)
        elif kind == "attnpool":
            self.query = nn.Parameter(torch.randn(1, 1, d_in) * d_in**-0.5)
            self.kv = nn.Linear(d_in, 2 * d_in, bias=False)
            self.proj = nn.Linear(d_in, d_out, bias=False)
        else:
            raise ValueError(f"unknown chunker kind: {kind}")
        self.out_norm = RMSNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M_{l-1}, d_in) -> (B, M_l, d_out)."""
        B, M, D = x.shape
        if M % self.chunk:
            raise ValueError(f"sequence {M} not divisible by chunk {self.chunk}")
        h = self.norm(x)

        if self.kind == "concat":
            out = self.proj(h.reshape(B, M // self.chunk, self.chunk * D))
        elif self.kind == "conv":
            out = self.proj(self.conv(h.transpose(1, 2)).transpose(1, 2))
        else:  # attnpool
            g = M // self.chunk
            hh = h.reshape(B * g, self.chunk, D)
            k, v = self.kv(hh).chunk(2, dim=-1)
            q = self.query.expand(B * g, 1, D)
            att = torch.softmax(q @ k.transpose(1, 2) * D**-0.5, dim=-1)
            out = self.proj((att @ v).squeeze(1)).view(B, g, self.d_out)
        return self.out_norm(out)


# --------------------------------------------------------------------------- #
# Context Converter   U_theta^(l)
# --------------------------------------------------------------------------- #
class ContextConverter(nn.Module):
    """Expand one level-l vector into ``R_l`` conditioning vectors of width
    ``d_out``: linear fan-out + 1-D conv along the ``R_l`` axis.

    ``d_skip`` concatenates the top-level ``X^(L)_{g-1}`` onto the input;
    parameters only, no extra sequence positions.
    """

    def __init__(self, d_in: int, d_out: int, width: int, kernel: int = 3,
                 d_skip: int = 0):
        super().__init__()
        self.width, self.d_out = width, d_out
        self.d_skip = d_skip
        self.norm = RMSNorm(d_in)
        if d_skip:
            self.skip_norm = RMSNorm(d_skip)
        self.fanout = nn.Linear(d_in + d_skip, width * d_out, bias=False)
        nn.init.normal_(self.fanout.weight, std=(d_in + d_skip)**-0.5)
        k = min(kernel, width) if width > 1 else 1
        self.conv = nn.Conv1d(d_out, d_out, kernel_size=k, padding=k // 2,
                              groups=d_out, bias=False)
        nn.init.zeros_(self.conv.weight)
        with torch.no_grad():  # start as identity
            self.conv.weight[:, :, k // 2] = 1.0
        self.out_norm = RMSNorm(d_out)
        self._trim = (k % 2 == 0)

    def forward(self, x: torch.Tensor,
                skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, M_l, d_in) -> (B, M_l, R_l, d_out).

        ``skip``: (B, M_l, d_skip), pre-aligned by the caller so that unit
        ``p`` sees ``X^(L)_{g-1}``.
        """
        B, M, _ = x.shape
        h = self.norm(x)
        if self.d_skip:
            if skip is None:
                raise ValueError("converter was built with a skip input")
            h = torch.cat([h, self.skip_norm(skip.to(h.dtype))], dim=-1)
        h = self.fanout(h).view(B * M, self.width, self.d_out)
        c = self.conv(h.transpose(1, 2)).transpose(1, 2)
        if self._trim:
            c = c[:, : self.width]
        return self.out_norm(h + c).view(B, M, self.width, self.d_out)


# --------------------------------------------------------------------------- #
# One hierarchy level
# --------------------------------------------------------------------------- #
class ScalaLevel(nn.Module):
    def __init__(self, cfg: LevelConfig, d_below: int, max_units: int,
                 level_idx: int, encoder_block: int | None = None,
                 d_skip: int = 0, encoder_window: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.level_idx = level_idx
        self.chunk = cfg.chunk_size
        self.width = cfg.converter_width
        self.d_below = d_below
        self.d_here = cfg.encoder.d_model
        #: C_{l+1} when this encoder is confined to one meta-group, else None.
        #: The generator reads it to size and recycle this level's KV cache.
        self.encoder_block = encoder_block
        #: units of *sliding* history when this encoder is windowed, else None.
        #: Takes precedence over ``encoder_block``.
        self.encoder_window = encoder_window

        enc_cfg = cfg.encoder
        from dataclasses import replace

        if self.encoder_window:
            enc_cfg = replace(
                enc_cfg,
                attention=replace(enc_cfg.attention, window=self.encoder_window,
                                  block=None),
            )
            self.encoder_block = None
        elif encoder_block:
            enc_cfg = replace(
                enc_cfg,
                attention=replace(enc_cfg.attention, block=encoder_block),
            )

        #: width of the top-level state fed straight into this level's
        #: converter, or 0 (top level gets none).
        self.d_skip = d_skip
        self.chunker = ContextChunker(d_below, self.d_here, cfg.chunk_size, cfg.chunker)
        self.encoder = TransformerStack(enc_cfg, max_units)
        self.converter = ContextConverter(self.d_here, d_below, cfg.converter_width,
                                          cfg.converter_kernel, d_skip=d_skip)
        #: chunks of previous content the local decoder may read (level 1
        #: only).
        self.lookback = cfg.decoder_lookback if level_idx == 1 else 0
        if cfg.decoder_lookback and level_idx != 1:
            raise ValueError(
                f"decoder_lookback is level-1 only (set on level {level_idx})")
        #: window, in decoder-stream positions, when this decoder runs as one
        #: sliding-window stream; ``None`` keeps independent R_l+C_l blocks.
        self.stream = cfg.decoder_stream
        dec_cfg = cfg.decoder
        if self.stream:
            dec_cfg = replace(
                dec_cfg,
                attention=replace(dec_cfg.attention, window=self.stream,
                                  block=None),
            )
            #: stream span: M_l groups of R_l + C_l; only sizes the RoPE table
            span = max_units * (cfg.converter_width + cfg.chunk_size)
        else:
            span = cfg.converter_width + (1 + self.lookback) * cfg.chunk_size
        self.decoder = TransformerStack(dec_cfg, span)
        # learned start latent  X_hat_0^(l)
        self.start_latent = nn.Parameter(torch.randn(self.d_here) * 0.02)
        # what the lookback positions hold for the first chunk of a sequence
        self.start_content = (
            nn.Parameter(torch.randn(self.lookback * cfg.chunk_size, d_below) * 0.02)
            if self.lookback else None)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.start_latent.normal_(0.0, 0.02)
        if self.start_content is not None:
            self.start_content.normal_(0.0, 0.02)

    def prev_content(self, content: torch.Tensor) -> torch.Tensor:
        """content: (B, M*C, D) -> (B, M, lookback*C, D); row ``g`` holds the
        tail of chunk ``g-1`` (learned start for ``g = 0``).  All positions are
        strictly earlier than chunk ``g`` -- no leakage.
        """
        B, MC, D = content.shape
        C, K = self.chunk, self.lookback
        M = MC // C
        start = self.start_content.to(content.dtype).view(1, 1, K * C, D)
        prev = content.view(B, M, C, D)[:, :-1].reshape(B, M - 1, C, D)
        if K == 1:
            return torch.cat([start.expand(B, 1, C, D), prev], dim=1)
        # K > 1: slide a (K*C)-wide window over the flat stream
        pad = start.expand(B, 1, K * C, D).reshape(B, K * C, D)
        flat = torch.cat([pad, content[:, : MC - C]], dim=1)      # (B, KC+MC-C, D)
        return flat.unfold(1, K * C, C).permute(0, 1, 3, 2)[:, :M]

    # -- bottom-up ------------------------------------------------------- #
    def encode(self, x_below: torch.Tensor, grad_ckpt: bool = False):
        a = self.chunker(x_below)
        return self.encoder(a, grad_ckpt=grad_ckpt)

    # -- top-down -------------------------------------------------------- #
    def shift_cond(self, cond_src: torch.Tensor) -> torch.Tensor:
        """Right-shift by one unit, inserting the learned start latent."""
        B = cond_src.shape[0]
        start = self.start_latent.to(cond_src.dtype).view(1, 1, -1).expand(B, 1, -1)
        return torch.cat([start, cond_src[:, :-1]], dim=1)

    def decode(self, cond_src: torch.Tensor, content: torch.Tensor,
               shift: bool = False, grad_ckpt: bool = False,
               skip: Optional[torch.Tensor] = None,
               pos_offset: int = 0) -> torch.Tensor:
        """cond_src (B, M_l, D_l), content (B, M_l*C_l, D_{l-1}) teacher-forced
        -> X_hat^(l-1) (B, M_l*C_l, D_{l-1}).
        ``shift=True`` (top level only): ``X^(L)_g`` already summarises group
        ``g``, so condition on ``X^(L)_{g-1}``; lower levels consume the upper
        decoder output unshifted.  ``pos_offset``: absolute stream position of
        row 0 for mid-sequence segment decodes (0 = whole sequence).
        """
        B, M, _ = cond_src.shape
        R, C = self.width, self.chunk
        src = self.shift_cond(cond_src) if shift else cond_src
        u = self.converter(src, skip)                            # (B, M, R, D_below)
        c = content.view(B, M, C, self.d_below)

        if self.stream:
            # one causal sliding-window stream of M*(R+C) positions; same
            # position count and readout offsets as the per-group block layout
            seq = torch.cat([u, c], dim=2).reshape(B, M * (R + C), self.d_below)
            out = self.decoder(seq, grad_ckpt=grad_ckpt, pos_offset=pos_offset)
            out = out.view(B, M, R + C, self.d_below)[:, :, R - 1 : R - 1 + C]
            return out.reshape(B, M * C, self.d_below)

        parts, pre = [u], R
        if self.lookback:
            parts.append(self.prev_content(content))
            pre += self.lookback * C
        parts.append(c)
        seq = torch.cat(parts, dim=2).reshape(B * M, pre + C, self.d_below)
        out = self.decoder(seq, grad_ckpt=grad_ckpt)
        # output at position pre-1+j predicts content slot j  (j = 0 .. C-1)
        out = out[:, pre - 1 : pre - 1 + C]
        return out.reshape(B, M * C, self.d_below)

    def decode_recursive(self, cond_src: torch.Tensor, shift: bool = False,
                         grad_ckpt: bool = False,
                         skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Eq. (6) reading: content positions hold the decoder's **own**
        reconstructions, not encoder states.  Recurrence of length ``C_l``
        within each group; the group axis stays parallel and the <= R_l + C_l
        prefix is recomputed each step (no cache).  Gradient flows through the
        recurrence.
        """
        if self.stream:
            raise ValueError(
                "recursive_decoder_input and decoder_stream are incompatible: "
                "the recurrence rebuilds a per-group prefix from scratch, "
                "which has no place to put the cross-group window")
        B, M, _ = cond_src.shape
        R, C = self.width, self.chunk
        src = self.shift_cond(cond_src) if shift else cond_src
        u = self.converter(src, skip)                            # (B, M, R, D)
        seq = u.reshape(B * M, R, self.d_below)

        outs = []
        for _ in range(C):
            out = self.decoder(seq, grad_ckpt=grad_ckpt)
            xhat = out[:, -1]                    # predicts the next slot
            outs.append(xhat)
            seq = torch.cat([seq, xhat[:, None]], dim=1)
        return torch.stack(outs, dim=1).reshape(B, M * C, self.d_below)

    # -- single-step decode used by the generator ------------------------ #
    def decode_step(self, prefix: torch.Tensor, step: int, cache, ) -> torch.Tensor:
        """Feed one position into the level-l decoder using its KV cache."""
        return self.decoder(prefix, cache=cache, pos_offset=step)


# --------------------------------------------------------------------------- #
# Multi-Token Prediction head (DeepSeek-V3 style)
# --------------------------------------------------------------------------- #
class MTPModule(nn.Module):
    def __init__(self, d_model: int, n_layers_hint: int = 1, inter: int = 2816):
        super().__init__()
        self.norm_h = RMSNorm(d_model)
        self.norm_e = RMSNorm(d_model)
        self.merge = nn.Linear(2 * d_model, d_model, bias=False)
        self.attn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, inter, n_layers_hint)
        self.out_norm = RMSNorm(d_model)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.merge(torch.cat([self.norm_h(h), self.norm_e(emb)], dim=-1))
        x = x + self.ffn(self.attn_norm(x))
        return self.out_norm(x)


# --------------------------------------------------------------------------- #
# Output container
# --------------------------------------------------------------------------- #
@dataclass
class ScalaOutput:
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    loss_token: Optional[torch.Tensor] = None
    loss_rec: Optional[torch.Tensor] = None
    loss_mtp: Optional[torch.Tensor] = None
    loss_aux: Optional[torch.Tensor] = None
    loss_z: Optional[torch.Tensor] = None
    enc_states: Optional[list[torch.Tensor]] = None
    dec_states: Optional[list[torch.Tensor]] = None
    rec_cos: Optional[list[float]] = None


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class ScalaForCausalLM(nn.Module):
    def __init__(self, cfg: ScalaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_token)
        nn.init.normal_(self.embed.weight, std=cfg.init_std)

        d_below = cfg.d_token
        levels = []
        mid: ScalaLevel | None = None
        for i, lv in enumerate(cfg.levels, start=1):
            max_units = cfg.max_seq_len // cfg.cumulative_chunk(i)
            top = i == len(cfg.levels)
            # block/window bounds apply to intermediate encoders only; the top
            # level must stay global
            block = (cfg.levels[i].chunk_size
                     if lv.encoder_block_local and not top else None)
            window = lv.encoder_window if (lv.encoder_window and not top) else None
            # no skip at the top: it already conditions on X^(L)_{g-1}
            skip = (cfg.levels[-1].encoder.d_model
                    if cfg.global_skip and not top else 0)
            if cfg.tie_mid_levels and 1 < i < len(cfg.levels):
                # one shared MID module for every middle level; nothing stored
                # is level-specific (RoPE grows on demand, inference caches
                # live in the generator, non-top start_latent never read)
                if mid is None:
                    mid = ScalaLevel(lv, d_below, max_units, i,
                                      encoder_block=block, d_skip=skip,
                                      encoder_window=window)
                levels.append(mid)
            else:
                levels.append(ScalaLevel(lv, d_below, max_units, i,
                                          encoder_block=block, d_skip=skip,
                                          encoder_window=window))
            d_below = lv.encoder.d_model
        if cfg.tie_mid_levels:
            # plain list, not nn.ModuleList: the shared MID must serialise
            # once, so the state_dict stays independent of depth
            self.level_token = levels[0]
            self.level_mid = levels[1]
            self.level_cap = levels[-1]
            self.levels = levels
        else:
            self.levels = nn.ModuleList(levels)

        self.final_norm = RMSNorm(cfg.d_token, eps=1e-6)
        self.lm_head = nn.Linear(cfg.d_token, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        else:
            nn.init.normal_(self.lm_head.weight, std=cfg.d_token**-0.5)

        if cfg.mtp_depth > 0:
            self.mtp = nn.ModuleList(
                MTPModule(cfg.d_token, 1, cfg.levels[0].decoder.ffn_inter_size)
                for _ in range(cfg.mtp_depth)
            )
        else:
            self.mtp = None

        self._warn_about_capacity_dropping(cfg)

        self.grad_ckpt = False
        #: probability of feeding X_hat rather than the encoder state X as the
        #: upper decoders' content input (scheduled sampling for RecGen).
        self.self_cond_prob = 0.0
        #: rows of hidden state per cross-entropy slice; ``None`` = one shot.
        #: Bounds fp32 logit memory (chunk * vocab * 4 B) vs the GEMM's M dim.
        self.loss_chunk_tokens: Optional[int] = 2048
        #: swapped for a compiled version by ``compile_loss()``
        self._ce_fn = self._ce_chunk

    # ------------------------------------------------------------------ #
    @staticmethod
    def _warn_about_capacity_dropping(cfg: ScalaConfig) -> None:
        """``capacity_factor`` makes the forward a function of batch shape:
        capacity = ceil(N * top_k / E * factor) over the rows of *this call*.
        Training drops overflow, one-group-at-a-time generation never does, so
        the two compute different functions; only ``None`` lets generation
        reproduce training.
        """
        offenders = [
            f"level{i}.{role}"
            for i, lv in enumerate(cfg.levels, start=1)
            for role, st in (("encoder", lv.encoder), ("decoder", lv.decoder))
            if st.moe.enabled and st.moe.capacity_factor
        ]
        if offenders:
            import warnings

            warnings.warn(
                "capacity_factor is set on " + ", ".join(offenders) +
                ". Token dropping depends on how many tokens are processed "
                "together, so generation cannot reproduce the training "
                "forward and HierGen is no longer exact. Set it to null "
                "unless you are deliberately trading that away.",
                stacklevel=3,
            )

    # ------------------------------------------------------------------ #
    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.grad_ckpt = enabled

    def compile_loss(self, **kw) -> None:
        """Compile the cross-entropy slice: fuses the LM-head GEMM with
        log-softmax so the (chunk, vocab) fp32 tensor is never materialised.
        """
        self._ce_fn = torch.compile(self._ce_chunk, dynamic=False, **kw)

    def reset_compiled_loss(self) -> None:
        self._ce_fn = self._ce_chunk

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    # ------------------------------------------------------------------ #
    def encode_all(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """Bottom-up pass.  Returns ``[X^(0), X^(1), ..., X^(L)]``."""
        x = self.embed(tokens)
        if self.cfg.scale_embeddings:
            x = x * math.sqrt(self.cfg.d_token)
        states = [x]
        for lvl in self.levels:
            states.append(lvl.encode(states[-1], self.grad_ckpt))
        return states

    def decode_all(self, enc: list[torch.Tensor]) -> list[torch.Tensor]:
        """Top-down pass; ``dec[l]`` holds ``X_hat^(l)``.  Pipelined by one
        unit: level-l output at slot ``j`` is the context for slot ``j`` one
        level down, so lower levels consume it unshifted; only the top shifts
        (``X^(L)_g`` contains group ``g``).  No shift in the loss.
        """
        L = self.n_levels
        dec: list[Optional[torch.Tensor]] = [None] * L
        cond = enc[L]                       # X_hat^(L) := X^(L)
        p = self.self_cond_prob if self.training else 0.0
        q = self.cfg.chunk_cond_prob if self.training else 0.0

        recursive = self.cfg.recursive_decoder_input
        # skip input = X^(L)_{g-1}, repeated across each meta-group's units
        top = (self.levels[-1].shift_cond(enc[L]) if self.cfg.global_skip
               else None)

        for l in range(L, 0, -1):
            lvl = self.levels[l - 1]
            content = enc[l - 1]
            skip = None
            if top is not None and l < L:
                skip = top.repeat_interleave(
                    enc[l].shape[1] // top.shape[1], dim=1)
            if recursive and l > 1:
                # Eq. (6) reading: no encoder state enters this decoder, so
                # self_cond_prob / chunk_cond_prob are skipped
                cond = dec[l - 1] = lvl.decode_recursive(
                    cond, shift=(l == L), grad_ckpt=self.grad_ckpt, skip=skip
                )
                continue
            xhat = lvl.decode(cond, content, shift=(l == L),
                              grad_ckpt=self.grad_ckpt, skip=skip)
            dec[l - 1] = xhat               # clean pass -> L_rec target
            if l > 1 and (p > 0 or q > 0):
                shape = content.shape[:2] + (1,)
                src = content
                if q > 0:
                    # chunkgen substitution: chunker summary of the units below
                    ch = self.levels[l - 2].chunker(enc[l - 2])
                    m = torch.rand(shape, device=content.device) < q
                    src = torch.where(m, ch.to(content.dtype), src)
                if p > 0:
                    # RecGen's substitution.
                    m = torch.rand(shape, device=content.device) < p
                    src = torch.where(m, xhat.detach().to(content.dtype), src)
                cond = lvl.decode(cond, src, shift=(l == L),
                                  grad_ckpt=self.grad_ckpt, skip=skip)
            else:
                cond = xhat                 # unshifted, see above
        return dec

    # ------------------------------------------------------------------ #
    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> ScalaOutput:
        cfg = self.cfg
        B, T = tokens.shape
        # ragged T: right-pad and drop the extra outputs -- exact, because the
        # padding is appended and the model is causal
        pad = (-T) % cfg.chunk_product
        tokens_in = F.pad(tokens, (0, pad)) if pad else tokens
        reset_moe_aux_losses()

        enc = self.encode_all(tokens_in)
        dec = self.decode_all(enc)

        h0 = self.final_norm(dec[0])                       # (B, T + pad, D0)
        if pad:
            # everything downstream -- loss, z-loss, MTP -- sees the true length
            h0 = h0[:, :T]
        out = ScalaOutput(enc_states=enc, dec_states=dec)

        if labels is None:
            labels = tokens

        # ---- token loss (no shift: the shift is architectural) --------- #
        if return_logits:
            logits = self.lm_head(h0)
            if cfg.logit_softcap:
                logits = torch.tanh(logits / cfg.logit_softcap) * cfg.logit_softcap
            out.logits = logits
            lt = self._ce(logits, labels, loss_mask)
            lz_raw = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        else:
            lt, lz_raw = self._lm_loss(h0, labels, loss_mask)
        out.loss_token = lt
        total = lt

        # ---- z-loss on the output logits ------------------------------- #
        if cfg.z_loss_weight > 0:
            lz = cfg.z_loss_weight * lz_raw
            out.loss_z = lz
            total = total + lz

        # ---- recursive consistency loss -------------------------------- #
        if cfg.rec_loss_alpha > 0:
            lrec, cosines = self._rec_loss(dec, enc)
            out.loss_rec, out.rec_cos = lrec, cosines
            total = total + cfg.rec_loss_alpha * lrec

        # ---- MTP ------------------------------------------------------- #
        if self.mtp is not None and cfg.mtp_depth > 0:
            lmtp = self._mtp_loss(h0, tokens, labels, loss_mask)
            out.loss_mtp = lmtp
            total = total + cfg.mtp_loss_weight * lmtp

        # ---- MoE side losses ------------------------------------------- #
        aux = moe_aux_losses()
        if aux.n_layers:
            out.loss_aux = aux.balance
            total = total + aux.balance + aux.router_z

        out.loss = total
        return out

    # ------------------------------------------------------------------ #
    # Loss over a ~100K vocabulary
    # ------------------------------------------------------------------ #
    # Slice the flattened sequence and recompute each slice's logits during
    # backward: peak fp32 logit memory is chunk*V instead of B*T*V.

    def _ce_chunk(self, h: torch.Tensor, labels: torch.Tensor):
        logits = self.lm_head(h)
        if self.cfg.logit_softcap:
            cap = self.cfg.logit_softcap
            logits = torch.tanh(logits / cap) * cap
        logits = logits.float()
        ce_i = F.cross_entropy(logits, labels, ignore_index=-100,
                               reduction="none")
        # log Z = CE_i + target logit (log_softmax identity): one gather, no
        # second full reduction
        valid = labels.ne(-100)
        safe = labels.masked_fill(~valid, 0).unsqueeze(-1)
        tgt_logit = logits.gather(-1, safe).squeeze(-1)
        logz = torch.where(valid, ce_i + tgt_logit, logits.new_zeros(()))
        return (ce_i * valid).sum(), logz.pow(2).sum(), valid.sum()

    def _lm_loss(self, h: torch.Tensor, labels: torch.Tensor,
                 loss_mask: Optional[torch.Tensor] = None):
        """Returns (mean cross-entropy, mean squared log-partition)."""
        flat_h = h.reshape(-1, h.shape[-1])
        lab = labels.reshape(-1)
        if loss_mask is not None:
            lab = lab.masked_fill(~loss_mask.reshape(-1).bool(), -100)

        n_total = flat_h.shape[0]
        chunk = self.loss_chunk_tokens or n_total
        recompute = self.training and torch.is_grad_enabled()
        ce_fn = self._ce_fn

        ce_sum = flat_h.new_zeros((), dtype=torch.float32)
        z_sum = flat_h.new_zeros((), dtype=torch.float32)
        n_valid = flat_h.new_zeros((), dtype=torch.float32)

        for i in range(0, n_total, chunk):
            hs, ls = flat_h[i : i + chunk], lab[i : i + chunk]
            if recompute and n_total > chunk:
                ce, z, n = torch.utils.checkpoint.checkpoint(
                    ce_fn, hs, ls, use_reentrant=False
                )
            else:
                ce, z, n = ce_fn(hs, ls)
            ce_sum = ce_sum + ce
            z_sum = z_sum + z
            n_valid = n_valid + n

        return ce_sum / n_valid.clamp_min(1.0), z_sum / max(n_total, 1)

    @staticmethod
    def _ce(logits, labels, loss_mask=None):
        """Plain cross-entropy from pre-computed logits (eval / tests)."""
        lg = logits.float().flatten(0, 1) if logits.ndim == 3 else logits.float()
        lb = labels.reshape(-1)
        if loss_mask is not None:
            lb = lb.masked_fill(~loss_mask.reshape(-1).bool(), -100)
        return F.cross_entropy(lg, lb, ignore_index=-100)

    def _rec_loss(self, dec: list[torch.Tensor], enc: list[torch.Tensor]):
        cfg = self.cfg
        total = dec[0].new_zeros(())
        cosines: list[float] = []
        # starts at rec_loss_min_level: RecGen substitutes only for l >= 2; at
        # l == 1 the generator feeds real token embeddings
        for l in range(max(cfg.rec_loss_min_level, 1), self.n_levels + 1):
            pred = dec[l - 1]
            tgt = enc[l - 1]
            if cfg.rec_loss_detach_target:
                tgt = tgt.detach()
            w = cfg.levels[l - 1].rec_loss_weight
            if cfg.rec_loss_kind == "cosine":
                cos = F.cosine_similarity(pred.float(), tgt.float(), dim=-1)
                term = (1.0 - cos).mean()
                cosines.append(cos.mean().detach().item() if not torch.is_grad_enabled()
                               else float("nan"))
            elif cfg.rec_loss_kind == "smooth_l1":
                term = F.smooth_l1_loss(pred.float(), tgt.float())
            else:
                term = F.mse_loss(pred.float(), tgt.float())
            total = total + w * term
        return total, cosines

    def _mtp_loss(self, h0, tokens, labels, loss_mask):
        """Predict t_{i+1}, t_{i+2}, ... from the level-0 decoder state."""
        cfg = self.cfg
        emb = self.embed(tokens)
        if cfg.scale_embeddings:
            emb = emb * math.sqrt(cfg.d_token)
        h = h0
        losses = []
        for d, mod in enumerate(self.mtp, start=1):
            # h_i already predicts t_i; combine with emb(t_i) to predict t_{i+d}
            h = mod(h[:, :-1], emb[:, :-1] if d == 1 else emb[:, d - 1 : -1])
            tgt = labels[:, d:]
            n = min(h.shape[1], tgt.shape[1])
            m = loss_mask[:, d : d + n] if loss_mask is not None else None
            # shares the chunked loss path to bound logit memory
            ce, _ = self._lm_loss(self.final_norm(h[:, :n]), tgt[:, :n], m)
            losses.append(ce)
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------ #
    # parameter accounting
    # ------------------------------------------------------------------ #
    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_word_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def num_active_parameters(self) -> int:
        """Parameters touched while producing one token, counting each stack's
        contribution with its amortisation factor (a level-l encoder runs once
        per ``C_<=l`` tokens)."""
        from .accounting import active_parameters

        return active_parameters(self.cfg)
