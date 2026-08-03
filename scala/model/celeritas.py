"""Celeritas preset: windowed intermediate encoders, streaming decoders,
NoPE/MLA top.

PHOTON hierarchy (arXiv:2512.20687) with ``rec_loss_alpha=0`` and
``chunk_cond_prob=0``.  Usage: ``celeritas_config(d_token=384,
vocab_size=99584)`` returns a ``ScalaConfig``.
"""

from __future__ import annotations

from dataclasses import replace

from .config import (
    AttentionConfig, LevelConfig, MoEConfig, ScalaConfig, StackConfig,
)

__all__ = ["celeritas_config"]


def _gqa(d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
         theta: float) -> AttentionConfig:
    return AttentionConfig(
        kind="gqa", n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        qk_norm=True, output_gate=True, pos="rope", rope_theta=theta,
    )


def _mla_nope(n_heads: int, q_lora: int, kv_lora: int, qk: int,
              v: int) -> AttentionConfig:
    """Top encoder: NoPE MLA, no decoupled RoPE key, pure latent cache.

    ``qk_norm`` must stay off -- it normalises the decompressed head vectors,
    which weight absorption cannot reproduce.  ``attn_sink`` counters NoPE
    attention dispersal at long context.
    """
    return AttentionConfig(
        kind="mla", n_heads=n_heads, mla_q_lora_rank=q_lora,
        mla_kv_lora_rank=kv_lora, mla_qk_nope_head_dim=qk,
        mla_qk_rope_head_dim=0, mla_v_head_dim=v,
        qk_norm=False, output_gate=True, attn_sink=True, pos="nope",
    )


def celeritas_config(
    d_token: int = 384,
    vocab_size: int = 99_584,
    max_seq_len: int = 512,
    *,
    chunks: tuple[int, ...] = (4, 4),
    enc_layers: tuple[int, ...] = (4, 6),
    dec_layers: tuple[int, ...] = (3, 2),
    ffn_inter: int | None = None,
    n_heads: int = 6,
    n_kv_heads: int = 2,
    head_dim: int = 64,
    encoder_window: int = 16,
    decoder_stream: int = 64,
    converter_widths: tuple[int, ...] = (1, 2),
    moe: MoEConfig | None = None,
    tie_word_embeddings: bool = True,
    **overrides,
) -> ScalaConfig:
    """Build a Celeritas ``ScalaConfig``.

    ``max_seq_len`` is a training shape, not a limit: it only sizes the RoPE
    tables of the span-bounded stacks (the top encoder has no table), so a
    checkpoint trained at 512 runs at any length.  ``encoder_window`` is in
    units; ``decoder_stream`` is in stream positions.
    """
    if len(chunks) != len(enc_layers) or len(chunks) != len(dec_layers):
        raise ValueError("chunks, enc_layers and dec_layers must agree in length")
    if len(converter_widths) != len(chunks):
        raise ValueError("converter_widths must have one entry per level")
    inter = ffn_inter or int(round(d_token * 8 / 3 / 64)) * 64
    top = len(chunks) - 1

    levels: list[LevelConfig] = []
    for i, c in enumerate(chunks):
        is_top = i == top
        enc_attn = (_mla_nope(n_heads, q_lora=d_token // 2,
                              kv_lora=d_token // 3 // 8 * 8,
                              qk=head_dim + 32, v=head_dim)
                    if is_top else _gqa(d_token, n_heads, n_kv_heads, head_dim,
                                        500_000.0))
        levels.append(LevelConfig(
            chunk_size=c,
            converter_width=converter_widths[i],
            converter_kernel=3,
            chunker="concat",
            # only intermediate encoders are bounded; the top is the global channel
            encoder_window=None if is_top else encoder_window,
            decoder_stream=decoder_stream,
            encoder=StackConfig(
                d_model=d_token, n_layers=enc_layers[i], ffn_inter_size=inter,
                n_dense_layers=enc_layers[i] if moe is None else 1,
                learned_residual_scale=True, attention=enc_attn,
                moe=moe or MoEConfig(enabled=False),
            ),
            decoder=StackConfig(
                d_model=d_token, n_layers=dec_layers[i], ffn_inter_size=inter,
                n_dense_layers=dec_layers[i] if moe is None else 1,
                learned_residual_scale=True,
                attention=_gqa(d_token, n_heads, n_kv_heads, head_dim, 10_000.0),
                moe=moe or MoEConfig(enabled=False),
            ),
        ))

    return ScalaConfig(
        vocab_size=vocab_size, d_token=d_token, max_seq_len=max_seq_len,
        levels=levels, tie_word_embeddings=tie_word_embeddings,
        # nothing is substituted at inference, so the substitution-correction
        # mechanisms are off
        rec_loss_alpha=0.0, chunk_cond_prob=0.0,
        **overrides,
    )
