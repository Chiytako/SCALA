"""SCALA: self-similar PHOTON hierarchy (arXiv:2512.20687), depth an
inference-time parameter.  Three design decisions:

1. Self-similar middle levels: three module sets -- level 1, one shared MID
   applied at every level 2..L-1, and the MLA/NoPE/sink CAP on the last
   level.  Parameter count is independent of depth.
2. Depth-free state_dict (``level_token.* / level_mid.* / level_cap.*``):
   one checkpoint instantiates at any depth via ``scala_config_at_depth``.
3. Bounded-top policy: deepen whenever the CAP holds more than ``U_max``
   units, so resident state is O(log T) bounded windows.

Usage::

    from scala.model.scala import scala_config, scala_config_at_depth
    cfg = scala_config(depth=2)              # L=4
    model = ScalaForCausalLM(cfg)
    deeper = scala_config_at_depth(cfg, 3)   # same checkpoint, L=5
    model3 = ScalaForCausalLM(deeper)
    model3.load_state_dict(model.state_dict())   # strict -- keys are depth-free
"""

from __future__ import annotations

import copy
from dataclasses import replace

from .celeritas import _gqa, _mla_nope
from .config import LevelConfig, MoEConfig, ScalaConfig, StackConfig

__all__ = ["scala_config", "scala_config_at_depth"]


def scala_config(
    depth: int = 2,
    d_token: int = 384,
    vocab_size: int = 99_584,
    max_seq_len: int | None = None,
    *,
    chunk: int = 4,
    l1_layers: tuple[int, int] = (4, 3),
    mid_layers: tuple[int, int] = (2, 2),
    cap_layers: tuple[int, int] = (4, 2),
    ffn_inter: int | None = None,
    mid_ffn_inter: int | None = None,
    n_heads: int = 6,
    n_kv_heads: int = 2,
    head_dim: int = 64,
    encoder_window: int = 16,
    decoder_stream: int = 64,
    converter_widths: tuple[int, int, int] = (1, 2, 2),
    cap_units: int = 16,
    moe: MoEConfig | None = None,
    tie_word_embeddings: bool = True,
    **overrides,
) -> ScalaConfig:
    """Build a SCALA ``ScalaConfig`` with ``depth`` MID applications; L = depth + 2.

    Extra ``depth`` costs zero parameters (the MID is shared).  ``max_seq_len``
    defaults to ``cap_units * chunk**(depth+2)`` -- enough for the CAP to hold
    ``cap_units`` units -- and is a training shape, not a limit.
    """
    if depth < 1:
        raise ValueError("scala_config needs depth >= 1 (one MID application)")
    inter = ffn_inter or int(round(d_token * 8 / 3 / 64)) * 64
    mid_inter = mid_ffn_inter or inter // 2
    if max_seq_len is None:
        max_seq_len = cap_units * chunk ** (depth + 2)

    enc_attn = replace(
        _gqa(d_token, n_heads, n_kv_heads, head_dim, 500_000.0),
        rope_partial=0.5,
    )
    dec_attn = _gqa(d_token, n_heads, n_kv_heads, head_dim, 10_000.0)
    cap_attn = _mla_nope(n_heads, q_lora=d_token // 2,
                         kv_lora=d_token // 3 // 8 * 8,
                         qk=head_dim + 32, v=head_dim)

    def stack(attn, n_layers: int, inter_size: int) -> StackConfig:
        return StackConfig(
            d_model=d_token, n_layers=n_layers, ffn_inter_size=inter_size,
            n_dense_layers=n_layers if moe is None else 1,
            learned_residual_scale=True, attention=attn,
            moe=moe or MoEConfig(enabled=False),
        )

    def level(role: str) -> LevelConfig:
        r, (enc_n, dec_n), inter_n = {
            "l1": (converter_widths[0], l1_layers, (inter, inter)),
            "mid": (converter_widths[1], mid_layers, (mid_inter, mid_inter)),
            "cap": (converter_widths[2], cap_layers, (inter, mid_inter)),
        }[role]
        return LevelConfig(
            chunk_size=chunk, converter_width=r, converter_kernel=3,
            chunker="concat",
            encoder_window=None if role == "cap" else encoder_window,
            decoder_stream=decoder_stream,
            encoder=stack(cap_attn if role == "cap" else enc_attn,
                          enc_n, inter_n[0]),
            decoder=stack(dec_attn, dec_n, inter_n[1]),
        )

    levels = [level("l1")] + [level("mid") for _ in range(depth)] + [level("cap")]
    return ScalaConfig(
        vocab_size=vocab_size, d_token=d_token, max_seq_len=max_seq_len,
        levels=levels, tie_word_embeddings=tie_word_embeddings,
        tie_mid_levels=True,
        rec_loss_alpha=0.0, chunk_cond_prob=0.0,
        **overrides,
    )


def scala_config_at_depth(
    cfg: ScalaConfig, depth: int, max_seq_len: int | None = None,
) -> ScalaConfig:
    """Re-express a tied config at a different depth.

    Legal because the state_dict keys are depth-invariant.  ``max_seq_len``
    defaults to the smallest multiple of the new chunk product >= the old
    value; it only sizes RoPE tables.
    """
    if not cfg.tie_mid_levels:
        raise ValueError(
            "scala_config_at_depth needs a tie_mid_levels config -- an untied "
            "checkpoint's weights are per-level and cannot be re-deepened")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    d = cfg.to_dict()
    lv = d["levels"]
    new_levels = ([copy.deepcopy(lv[0])]
                  + [copy.deepcopy(lv[1]) for _ in range(depth)]
                  + [copy.deepcopy(lv[-1])])
    cp = new_levels[0]["chunk_size"]
    for entry in new_levels[1:]:
        cp *= entry["chunk_size"]
    want = max_seq_len or cfg.max_seq_len
    d["levels"] = new_levels
    d["max_seq_len"] = ((want + cp - 1) // cp) * cp
    return ScalaConfig.from_dict(d)
