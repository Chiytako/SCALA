"""Analytic parameter / FLOP accounting for SCALA.

``total``: every checkpoint parameter.  ``active``: parameters on one token's
forward path (only ``top_k`` experts counted).  ``amortised``: a level-l
encoder runs once per ``C_<=l`` tokens and its decoder once per ``C_<=(l-1)``,
so per-token cost divides by that factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import AttentionConfig, MoEConfig, ScalaConfig, StackConfig

__all__ = [
    "StackCount",
    "ModelCount",
    "count_model",
    "active_parameters",
    "decoder_cache_bytes_per_token",
    "scala_depth_for_context",
    "scala_state_bytes",
    "format_report",
]


# --------------------------------------------------------------------------- #
def attn_params(a: AttentionConfig, d: int) -> int:
    extra = d * a.n_heads if a.output_gate else 0
    return _attn_core(a, d) + extra


def _attn_core(a: AttentionConfig, d: int) -> int:
    if a.kind == "gqa":
        hd = a.head_dim or d // a.n_heads
        p = d * a.n_heads * hd            # Wq
        p += 2 * d * a.n_kv_heads * hd    # Wk, Wv
        p += a.n_heads * hd * d           # Wo
        if a.qk_norm:
            p += 2 * hd
        if a.attn_sink:
            p += a.n_heads
        return p
    # MLA
    qk = a.mla_qk_nope_head_dim + a.mla_qk_rope_head_dim
    p = 0
    if a.mla_q_lora_rank:
        p += d * a.mla_q_lora_rank + a.mla_q_lora_rank * a.n_heads * qk
        p += a.mla_q_lora_rank                                  # q latent norm
    else:
        p += d * a.n_heads * qk
    p += d * (a.mla_kv_lora_rank + a.mla_qk_rope_head_dim)      # W^{DKV}
    p += a.mla_kv_lora_rank                                     # kv latent norm
    p += a.mla_kv_lora_rank * a.n_heads * (a.mla_qk_nope_head_dim + a.mla_v_head_dim)
    p += a.n_heads * a.mla_v_head_dim * d                       # Wo
    if a.qk_norm:
        p += 2 * qk
    if a.attn_sink:
        p += a.n_heads      # the learned per-head sink logit is a parameter
    return p


def moe_params(m: MoEConfig, d: int) -> tuple[int, int]:
    """Returns (total, active) parameters of one MoE FFN."""
    routed = m.n_routed_experts * 3 * d * m.expert_inter_size
    shared = m.n_shared_experts * 3 * d * (m.shared_inter_size or m.expert_inter_size)
    router = m.n_routed_experts * d + m.n_routed_experts  # weight + bias buffer
    total = routed + shared + router
    active = m.top_k * 3 * d * m.expert_inter_size + shared + router
    return total, active


def dense_ffn_params(d: int, inter: int) -> int:
    return 3 * d * inter


@dataclass
class StackCount:
    name: str
    total: int = 0
    active: int = 0
    #: how many tokens one invocation of this stack covers
    amortisation: int = 1
    #: sequence length this stack attends over, for FLOP purposes
    attn_span: float = 0.0
    #: re-application of shared MID weights: still counted in active/amortised
    #: (cost is per application) but stored once, so ``ModelCount.total`` skips it
    tied: bool = False

    @property
    def amortised_active(self) -> float:
        return self.active / self.amortisation


def count_stack(cfg: StackConfig, name: str, amortisation: int = 1) -> StackCount:
    d = cfg.d_model
    total = active = 0
    for i in range(cfg.n_layers):
        p = attn_params(cfg.attention, d) + 2 * d  # + 2 RMSNorms
        if cfg.learned_residual_scale:
            p += 4 * d                             # ZAYA1 residual gains
        total += p
        active += p
        if cfg.moe.enabled and i >= cfg.n_dense_layers:
            t, a = moe_params(cfg.moe, d)
            total += t
            active += a
        else:
            f = dense_ffn_params(d, cfg.ffn_inter_size)
            total += f
            active += f
    total += d  # final norm
    active += d
    return StackCount(name, total, active, amortisation)


# --------------------------------------------------------------------------- #
@dataclass
class ModelCount:
    stacks: list[StackCount] = field(default_factory=list)
    embedding: int = 0
    lm_head: int = 0
    glue: int = 0            # chunkers, converters, start latents, final norm
    mtp: int = 0

    @property
    def total(self) -> int:
        return (self.embedding + self.lm_head + self.glue + self.mtp
                + sum(s.total for s in self.stacks if not s.tied))

    @property
    def active(self) -> int:
        return (self.embedding + self.lm_head + self.glue
                + sum(s.active for s in self.stacks))

    @property
    def active_non_embedding(self) -> int:
        return self.active - self.embedding - self.lm_head

    @property
    def amortised_active(self) -> float:
        return (self.embedding + self.lm_head + self.glue
                + sum(s.amortised_active for s in self.stacks))

    @property
    def total_non_embedding(self) -> int:
        return self.total - self.embedding - self.lm_head


def count_model(cfg: ScalaConfig) -> ModelCount:
    mc = ModelCount()
    mc.embedding = cfg.vocab_size * cfg.d_token
    mc.lm_head = 0 if cfg.tie_word_embeddings else cfg.vocab_size * cfg.d_token
    mc.glue = cfg.d_token  # final norm

    for i, lv in enumerate(cfg.levels, start=1):
        d_below = cfg.width(i - 1)
        d_here = lv.encoder.d_model
        c_le_l = cfg.cumulative_chunk(i)          # tokens per level-i unit
        c_le_lm1 = cfg.cumulative_chunk(i - 1)    # tokens per level-(i-1) unit
        # SCALA: levels 3..L-1 re-apply level 2's weights.
        tied_dup = cfg.tie_mid_levels and 2 < i < cfg.n_levels

        enc = count_stack(lv.encoder, f"L{i}.encoder", c_le_l)
        dec = count_stack(lv.decoder, f"L{i}.decoder", c_le_lm1)
        enc.tied = dec.tied = tied_dup
        mc.stacks += [enc, dec]

        if tied_dup:
            continue  # glue (chunker/converter/start latent) is shared too
        # chunker
        if lv.chunker == "concat":
            mc.glue += d_below * lv.chunk_size * d_here
        elif lv.chunker == "conv":
            mc.glue += d_below * d_here * lv.chunk_size + d_here * d_here
        else:
            mc.glue += d_below + 2 * d_below * d_below + d_below * d_here
        mc.glue += d_below + d_here  # chunker norms
        # converter: fanout + depthwise conv + norms
        mc.glue += d_here * lv.converter_width * d_below
        mc.glue += d_below * min(lv.converter_kernel, lv.converter_width)
        mc.glue += d_here + d_below
        # start latent
        mc.glue += d_here

    if cfg.mtp_depth > 0:
        d = cfg.d_token
        per = 2 * d * d + dense_ffn_params(d, cfg.levels[0].decoder.ffn_inter_size)
        per += 5 * d
        mc.mtp = cfg.mtp_depth * per

    return mc


def active_parameters(cfg: ScalaConfig) -> int:
    return count_model(cfg).active


# --------------------------------------------------------------------------- #
def flops_per_token(cfg: ScalaConfig, seq_len: int | None = None) -> dict[str, float]:
    """Forward FLOPs per generated token (2 * MACs), attention included.

    ``out["head"]`` is broken out: the LM head dominates at small widths and
    no hierarchy knob touches it.
    """
    T = seq_len or cfg.max_seq_len
    mc = count_model(cfg)
    out: dict[str, float] = {}
    matmul = 0.0
    attn = 0.0

    for i, lv in enumerate(cfg.levels, start=1):
        c_le_l = cfg.cumulative_chunk(i)
        c_le_lm1 = cfg.cumulative_chunk(i - 1)

        enc = next(s for s in mc.stacks if s.name == f"L{i}.encoder")
        dec = next(s for s in mc.stacks if s.name == f"L{i}.decoder")
        # decoder runs R_l + C_l positions to emit C_l units, so the converter
        # prefix is paid on every token; decoder_stream still costs R_l + C_l
        # positions per group (concatenation adds no positions)
        dec_span = ((lv.converter_width + lv.chunk_size) / lv.chunk_size
                    if lv.decoder_stream else
                    (lv.converter_width
                     + (1 + lv.decoder_lookback) * lv.chunk_size)
                    / lv.chunk_size)
        matmul += 2 * enc.active / c_le_l + 2 * dec.active * dec_span / c_le_lm1

        # attention score/AV cost
        e = lv.encoder
        hd = e.attention.resolve_head_dim(e.d_model)
        # a windowed encoder never scores more than its window, however long T is
        span_enc = T / c_le_l                      # units seen by the level-i encoder
        if lv.encoder_window:
            span_enc = min(span_enc, lv.encoder_window)
        elif lv.encoder_block_local and i < len(cfg.levels):
            span_enc = min(span_enc, cfg.levels[i].chunk_size)
        attn += 2 * 2 * e.n_layers * e.attention.n_heads * hd * span_enc / c_le_l

        d = lv.decoder
        hdd = d.attention.resolve_head_dim(d.d_model)
        if lv.decoder_stream:
            # each of the R+C stream positions scores up to `window` keys
            span_dec = ((lv.converter_width + lv.chunk_size)
                        * lv.decoder_stream / lv.chunk_size)
        else:
            span_dec = (lv.converter_width
                        + (1 + lv.decoder_lookback) * lv.chunk_size)
        attn += 2 * 2 * d.n_layers * d.attention.n_heads * hdd * span_dec / c_le_lm1

    head = 2 * cfg.vocab_size * cfg.d_token
    matmul += head
    out["head"] = head
    out["stacks"] = matmul - head
    out["matmul"] = matmul
    out["attention"] = attn
    out["total"] = matmul + attn
    return out


def uses_latent_cache(cfg: ScalaConfig) -> bool:
    """True when every MLA stack can run the weight-absorbed latent cache."""
    mla = [lv.encoder.attention for lv in cfg.levels
           if lv.encoder.attention.kind == "mla"]
    return bool(mla) and all(not a.qk_norm for a in mla)


def _per_unit_bytes_units(stack: StackConfig, absorbed: bool) -> int:
    """Cache entries (values, not bytes) one cached unit costs per layer."""
    a = stack.attention
    if a.kind == "mla":
        if absorbed:
            return a.mla_kv_lora_rank + a.mla_qk_rope_head_dim
        qk = a.mla_qk_nope_head_dim + a.mla_qk_rope_head_dim
        return a.n_heads * (qk + a.mla_v_head_dim)
    return 2 * a.n_kv_heads * (a.head_dim or stack.d_model // a.n_heads)


#: RecGen's fallback truncation for a lower level that has no structural
#: ``encoder_window``/``encoder_block_local`` of its own -- must track
#: ``scala.infer.generate.DEFAULT_ENC_WINDOW_GROUPS`` (checked by
#: ``test_accounting_recgen_window_groups_matches_generator_default``), kept
#: as a local constant so this module does not import from ``scala.infer``.
_DEFAULT_RECGEN_WINDOW_GROUPS = 4


def kv_cache_bytes_per_token(cfg: ScalaConfig, dtype_bytes: int = 2,
                             recgen: bool = False,
                             absorbed: bool | None = None,
                             context_tokens: int | None = None,
                             window_groups: int | None = None,
                             lower_encoder: bool = True) -> float:
    """Bytes of global KV cache per generated token.

    ``absorbed=None`` auto-detects: latent cache (``kv_lora_rank +
    qk_rope_head_dim`` per unit/layer) iff every MLA stack has qk_norm off --
    qk_norm forces the decompressed ``n_heads * (qk + v)`` cache.

    Mirrors ``ScalaGenerator``'s own two-stage decision. First,
    ``lower_encoder=False`` (``chunkgen``/``recgen_paper``: no substitute-free
    protocol keeps a lower cache at all, see ``PROTOCOLS`` in
    ``scala.infer.generate``) zeroes every level below the top outright.
    Otherwise a lower level's cache is bounded -- identically for ``recgen``
    and ``hiergen`` -- whenever it is structurally span-bounded at training
    time (``encoder_window``/``encoder_block_local``); this is what
    ``ScalaGenerator._window_units`` checks first, before ever consulting
    ``recgen``. Only a lower level with NEITHER gets RecGen's approximate
    cache-truncation fallback, sized at ``window_groups`` meta-groups
    (default: the same value ``ScalaGenerator`` itself defaults to). Bounded
    caches are amortised over ``context_tokens`` (default ``cfg.max_seq_len``).
    """
    if absorbed is None:
        absorbed = uses_latent_cache(cfg)
    if window_groups is None:
        window_groups = _DEFAULT_RECGEN_WINDOW_GROUPS
    ctx = context_tokens or cfg.max_seq_len
    total = 0.0
    top = cfg.n_levels
    for i in range(1, top + 1):
        if not lower_encoder and i < top:
            continue
        lvl = cfg.levels[i - 1]
        e = lvl.encoder
        per_unit = _per_unit_bytes_units(e, absorbed) * dtype_bytes
        bounded_units = None
        if lvl.encoder_window and i < top:
            # windowed level: cache is `window + one write block` for every protocol
            bounded_units = lvl.encoder_window + cfg.levels[i].chunk_size
        elif lvl.encoder_block_local and i < top:
            bounded_units = cfg.levels[i].chunk_size      # one group, ever
        elif recgen and i < top:
            bounded_units = cfg.levels[i].chunk_size * window_groups
        if bounded_units is not None:
            total += e.n_layers * per_unit * bounded_units / ctx
        else:
            total += e.n_layers * per_unit / cfg.cumulative_chunk(i)
    return total


def layer_positions_per_token(cfg: ScalaConfig) -> list[tuple[str, float]]:
    """Transformer-layer *positions* each stack runs per output token.

    A level-l encoder layer runs once per `C_<=l` tokens; a level-l decoder
    layer runs `R_l + C_l` positions to emit `C_l` units, only `C_l` of which
    are read back.  Positions, NOT compute: width, LM head and embedding are
    not counted -- use ``flops_per_token`` for cost claims.
    """
    out: list[tuple[str, float]] = []
    for i, lv in enumerate(cfg.levels, start=1):
        below = cfg.cumulative_chunk(i - 1) if i > 1 else 1
        out.append((f"level{i}.encoder",
                    lv.encoder.n_layers / cfg.cumulative_chunk(i)))
        span = (lv.converter_width + lv.chunk_size if lv.decoder_stream
                else lv.converter_width
                + (1 + lv.decoder_lookback) * lv.chunk_size)
        out.append((f"level{i}.decoder",
                    lv.decoder.n_layers * span / (lv.chunk_size * below)))
    return out


def decoder_cache_bytes_per_token(cfg: ScalaConfig, dtype_bytes: int = 2,
                                  context_tokens: int | None = None) -> float:
    """Bytes of *decoder* KV cache per generated token, separate from
    ``kv_cache_bytes_per_token``.

    Only ``decoder_stream`` levels hold persistent decoder state:
    ``window + max(R_l + C_l, 64)`` entries for the whole sequence, O(1) in
    ``T`` -- the ``64`` floor matches ``ScalaGenerator._alloc_caches``'s own
    ``blk = max(lvl.width + lvl.chunk, 64)``, which otherwise silently
    undersizes this estimate for any level whose converter width plus chunk
    size is small (true of the shipped probe geometry: 2 + 4 = 6).
    Non-streaming decoders use per-group scratch, not cached state.
    """
    ctx = context_tokens or cfg.max_seq_len
    total = 0.0
    for i, lv in enumerate(cfg.levels, start=1):
        if not lv.decoder_stream:
            continue
        d = lv.decoder
        per = _per_unit_bytes_units(d, absorbed=True)
        units = lv.decoder_stream + max(lv.converter_width + lv.chunk_size, 64)
        total += d.n_layers * per * dtype_bytes * units / ctx
    return total


# --------------------------------------------------------------------------- #
# SCALA: the bounded-top policy and its O(log T) resident state
# --------------------------------------------------------------------------- #
def scala_depth_for_context(cfg: ScalaConfig, context_tokens: int,
                            u_max: int = 32) -> int:
    """Bounded-top policy: the smallest depth ``k`` (never below the config's
    own) at which the CAP holds at most ``u_max`` units of a
    ``context_tokens``-token context.

    Under it no cache grows with ``T``; ``k`` grows as ``log_C(T)``.
    """
    if not cfg.tie_mid_levels:
        raise ValueError("the depth policy needs a tie_mid_levels config -- "
                         "an untied checkpoint cannot change depth")
    c1 = cfg.levels[0].chunk_size
    c_mid = cfg.levels[1].chunk_size
    c_cap = cfg.levels[-1].chunk_size
    k = max(1, cfg.n_levels - 2)
    while context_tokens > u_max * c1 * (c_mid ** k) * c_cap:
        k += 1
    return k


def scala_state_bytes(cfg: ScalaConfig, context_tokens: int,
                      u_max: int = 32, dtype_bytes: int = 2,
                      absorbed: bool | None = None) -> dict[str, float]:
    """Total resident cache at ``context_tokens`` under the bounded-top policy.

    Absolute bytes per sequence (O(log T)), not per token: one bounded encoder
    window and one bounded decoder stream per instantiated level, plus a CAP
    of at most ``u_max`` units.
    """
    from .scala import scala_config_at_depth  # local: avoid import cycles

    k = scala_depth_for_context(cfg, context_tokens, u_max)
    inst = cfg if k == cfg.n_levels - 2 else scala_config_at_depth(cfg, k)
    if absorbed is None:
        absorbed = uses_latent_cache(inst)
    top = inst.n_levels
    enc_windows = cap = decoders = 0.0
    for i, lv in enumerate(inst.levels, start=1):
        e = lv.encoder
        per = _per_unit_bytes_units(e, absorbed) * dtype_bytes
        if i < top:
            # sliding window + one write block, as allocated by the generator
            units = (lv.encoder_window or 0) + inst.levels[i].chunk_size
            enc_windows += e.n_layers * per * units
        else:
            units = min(u_max, max(1, -(-context_tokens // inst.chunk_product)))
            cap += e.n_layers * per * units
        if lv.decoder_stream:
            d = lv.decoder
            per_d = _per_unit_bytes_units(d, absorbed=True) * dtype_bytes
            # `64` floor matches `ScalaGenerator._alloc_caches`'s own
            # `blk = max(lvl.width + lvl.chunk, 64)`
            units_d = lv.decoder_stream + max(lv.converter_width + lv.chunk_size, 64)
            decoders += d.n_layers * per_d * units_d
    return {
        "depth": k,
        "levels": top,
        "bytes_windows": enc_windows,
        "bytes_cap": cap,
        "bytes_decoders": decoders,
        "bytes_total": enc_windows + cap + decoders,
    }


# --------------------------------------------------------------------------- #
def format_report(cfg: ScalaConfig) -> str:
    mc = count_model(cfg)
    B = 1e9
    M = 1e6
    lines = [
        "=" * 78,
        "SCALA parameter report",
        "=" * 78,
        f"vocab={cfg.vocab_size:,}  D0={cfg.d_token}  L={cfg.n_levels}  "
        f"C_<=L={cfg.chunk_product}  ctx={cfg.max_seq_len}",
        "",
        f"{'stack':<16}{'total':>14}{'active':>14}{'amort':>8}{'amort.act':>14}",
        "-" * 78,
    ]
    for s in mc.stacks:
        name = s.name + (" =L2" if s.tied else "")
        lines.append(
            f"{name:<16}{s.total/M:>12.1f}M{s.active/M:>12.1f}M"
            f"{s.amortisation:>8}{s.amortised_active/M:>12.1f}M"
        )
    lines += [
        "-" * 78,
        f"{'embedding':<16}{mc.embedding/M:>12.1f}M",
        f"{'lm_head':<16}{mc.lm_head/M:>12.1f}M",
        f"{'chunk/convert':<16}{mc.glue/M:>12.1f}M",
        f"{'mtp':<16}{mc.mtp/M:>12.1f}M",
        "=" * 78,
        f"TOTAL          : {mc.total/B:.3f} B   ({mc.total_non_embedding/B:.3f} B non-emb)",
        f"ACTIVE / token : {mc.active/B:.3f} B   "
        f"({mc.active_non_embedding/B:.3f} B non-emb)",
        f"AMORTISED      : {mc.amortised_active/B:.3f} B  "
        f"(FLOP-equivalent dense size)",
        f"sparsity       : {mc.total/max(mc.active,1):.2f}x",
    ]
    lp = layer_positions_per_token(cfg)
    lp_total = sum(v for _, v in lp)
    fl = flops_per_token(cfg)
    lines += [
        "",
        f"{'layer-positions / token':<24}{'':>10}{'share':>8}   "
        f"(positions, NOT compute -- see below)",
        "-" * 78,
    ]
    for name, v in lp:
        lines.append(f"{name:<24}{v:>10.3f}{100*v/max(lp_total,1e-9):>7.1f}%")
    lines.append(f"{'TOTAL':<24}{lp_total:>10.3f}")
    lines += [
        f"{'':<24}{'':>10}   the hierarchy knobs move this number; they do not",
        f"{'':<24}{'':>10}   move the LM head, which is "
        f"{100*fl['head']/max(fl['total'],1e-9):.0f}% of the FLOPs below.",
    ]

    lines += [
        "",
        f"fwd FLOPs/token @ ctx={cfg.max_seq_len}: {fl['total']/1e9:.2f} GFLOP "
        f"(stacks {fl['stacks']/1e9:.3f}, lm_head {fl['head']/1e9:.3f}, "
        f"attn {fl['attention']/1e9:.3f})",
        f"KV cache mode    : "
        f"{'MLA latent (weight-absorbed)' if uses_latent_cache(cfg) else 'decompressed K/V'}",
        f"KV cache  HierGen: {kv_cache_bytes_per_token(cfg)/1024:.3f} KiB/token"
        f"   (other mode: "
        f"{kv_cache_bytes_per_token(cfg, absorbed=not uses_latent_cache(cfg))/1024:.3f})",
        f"KV cache  RecGen : "
        f"{kv_cache_bytes_per_token(cfg, recgen=True)/1024:.3f} KiB/token"
        f"   (other mode: "
        f"{kv_cache_bytes_per_token(cfg, recgen=True, absorbed=not uses_latent_cache(cfg))/1024:.3f})",
        f"KV cache decoder : "
        f"{decoder_cache_bytes_per_token(cfg)/1024:.3f} KiB/token"
        f"   (streaming decoders only; PHOTON's is per-group scratch)",
        f"position scheme  : " + ", ".join(
            f"L{i}.enc={lv.encoder.attention.pos}"
            + (f"/win{lv.encoder_window}" if lv.encoder_window else "")
            for i, lv in enumerate(cfg.levels, start=1)),
        f"  -> {kv_cache_bytes_per_token(cfg)*cfg.max_seq_len/2**20:.1f} MiB "
        f"for a full {cfg.max_seq_len}-token context (HierGen, per sequence)",
        "=" * 78,
    ]
    return "\n".join(lines)
