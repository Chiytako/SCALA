"""Configuration dataclasses for SCALA (hierarchical MoE LM, arXiv:2512.20687).

Level l: chunker C^(l) folds C_l level-(l-1) units into 1; encoder F^(l) is causal
over the M_l units; converter U^(l) expands 1 unit -> R_l conditioning vectors;
decoder G^(l) is causal over [R_l conditioning ; C_l content].
Level 0 is the token level: X^(0) embeddings, X_hat^(0) -> vocab logits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "MoEConfig",
    "AttentionConfig",
    "StackConfig",
    "LevelConfig",
    "ScalaConfig",
]


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class MoEConfig:
    """Mixture-of-Experts settings for a single transformer stack.

    DeepSeek-V3 recipe: fine-grained routed experts + always-on shared experts,
    aux-loss-free load balancing via a per-expert bias updated outside the
    gradient.
    """

    enabled: bool = True
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    top_k: int = 6
    #: FFN hidden size of ONE routed expert (fine-grained: much smaller than dense).
    expert_inter_size: int = 768
    #: FFN hidden size of ONE shared expert. Defaults to ``expert_inter_size``.
    shared_inter_size: int | None = None

    #: Router gate. "sigmoid" (DeepSeek-V3) scores each expert independently.
    score_func: Literal["sigmoid", "softmax"] = "sigmoid"
    #: Renormalise the top-k gate values so they sum to 1 before scaling.
    norm_topk_prob: bool = True
    #: Multiplies the combined routed output (DeepSeek-V3 uses 2.5 for sigmoid).
    routed_scaling_factor: float = 2.5

    #: Aux-loss-free balancing: bias step size gamma; anneal to 0 in the final
    #: phase of training.
    bias_update_rate: float = 1e-3

    # -- bias controller ---------------------------------------------------- #
    #: "sign" = ``b += gamma * sign(mean - load)``; "pid" uses the magnitude of
    #: the imbalance and is correspondingly less sensitive to gamma.
    bias_controller: str = "sign"          # "sign" | "pid"
    pid_kp: float = 1.0
    pid_ki: float = 0.05
    pid_kd: float = 0.2
    #: Clamp on |b|.
    bias_clip: float = 2.0

    #: Tiny *sequence-wise* balance loss retained as a guard rail (DSv3: 1e-4).
    seq_aux_loss_weight: float = 1e-4
    #: Router logit z-loss (ST-MoE) to keep gate logits from drifting.
    router_z_loss_weight: float = 1e-3

    #: Node/group-limited routing (DeepSeek-V3 device-limited routing).
    n_groups: int = 1
    topk_groups: int = 1

    #: Capacity factor; ``None`` == dropless (all tokens routed).
    capacity_factor: float | None = None

    def __post_init__(self) -> None:
        if self.shared_inter_size is None:
            self.shared_inter_size = self.expert_inter_size
        if self.n_groups > 1 and self.n_routed_experts % self.n_groups != 0:
            raise ValueError("n_routed_experts must be divisible by n_groups")
        if self.top_k > self.n_routed_experts:
            raise ValueError("top_k cannot exceed n_routed_experts")


@dataclass
class AttentionConfig:
    """Attention settings for a single transformer stack."""

    kind: Literal["gqa", "mla"] = "gqa"

    n_heads: int = 16
    #: Grouped-query KV heads (``gqa`` only).
    n_kv_heads: int = 4
    head_dim: int | None = None  # defaults to d_model // n_heads

    # -- MLA (DeepSeek-V2/V3 multi-head latent attention) ------------------- #
    mla_q_lora_rank: int | None = None       # None -> no low-rank on Q
    mla_kv_lora_rank: int = 256
    mla_qk_nope_head_dim: int = 96
    mla_qk_rope_head_dim: int = 32
    mla_v_head_dim: int = 96

    # -- shared knobs ------------------------------------------------------- #
    qk_norm: bool = True                 # per-head RMSNorm on q and k
    #: ``"rope"`` rotates q/k by absolute position; ``"nope"`` applies none and
    #: lets the causal mask carry order (top encoder: the only unbounded stack).
    pos: Literal["rope", "nope"] = "rope"
    rope_theta: float = 1_000_000.0
    #: Fraction of head_dim that receives RoPE (partial RoPE, as in Qwen3-Next).
    rope_partial: float = 1.0
    #: ``None`` = full causal.  Otherwise a causal sliding window of this width.
    window: int | None = None
    #: ``None`` = whole sequence.  Otherwise causal *within* disjoint blocks of
    #: this width, never crossing one, so the KV cache is bounded by the block.
    block: int | None = None
    #: Learned per-head attention sink logit (GPT-OSS / StreamingLLM style).
    attn_sink: bool = False
    #: Per-head query-dependent sigmoid gate on the attention output before it
    #: rejoins the residual stream (Qwen3-Next).
    output_gate: bool = False
    softmax_scale: float | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.pos not in ("rope", "nope"):
            raise ValueError(f"AttentionConfig.pos must be rope|nope, got {self.pos!r}")
        if self.pos == "nope" and self.kind == "mla" and self.mla_qk_rope_head_dim:
            raise ValueError(
                "pos='nope' with kind='mla' requires mla_qk_rope_head_dim=0 -- "
                "the decoupled RoPE key exists only to carry RoPE past weight "
                "absorption, so under NoPE it is pure cache overhead "
                f"({self.mla_qk_rope_head_dim} channels/unit/layer)")

    def resolve_head_dim(self, d_model: int) -> int:
        if self.kind == "mla":
            return self.mla_qk_nope_head_dim + self.mla_qk_rope_head_dim
        return self.head_dim or (d_model // self.n_heads)


@dataclass
class StackConfig:
    """A transformer stack (used for encoders and decoders alike)."""

    d_model: int = 1024
    n_layers: int = 12
    #: Dense FFN hidden size: used when ``moe.enabled`` is False, and for the
    #: first ``n_dense_layers`` layers of an MoE stack.
    ffn_inter_size: int = 2816
    n_dense_layers: int = 1

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)

    norm_eps: float = 1e-6
    #: Scale residual-branch output projections by 1/sqrt(2*n_layers).
    scale_residual_init: bool = True
    #: Per-channel learned gain on both the residual pass-through and the
    #: sublayer output; 4*n_layers*d_model extra parameters.
    learned_residual_scale: bool = False
    #: Every ``full_attn_every``-th layer uses full attention, the rest use
    #: ``attention.window``.  ``0`` disables.
    full_attn_every: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.attention, dict):
            self.attention = AttentionConfig(**self.attention)
        if isinstance(self.moe, dict):
            self.moe = MoEConfig(**self.moe)


@dataclass
class LevelConfig:
    """One level of the PHOTON hierarchy (l = 1 .. L)."""

    #: C_l -- how many level-(l-1) units are folded into one level-l unit.
    chunk_size: int = 4
    #: R_l -- how many conditioning vectors the Context Converter emits.
    converter_width: int = 4
    #: Conv kernel size used by the Context Converter.
    converter_kernel: int = 3
    #: "concat" = concat C_l vectors then project; "conv" = depthwise-separable
    #: strided conv; "attnpool" = learned-query cross-attention pooling.
    chunker: Literal["concat", "conv", "attnpool"] = "concat"

    encoder: StackConfig = field(default_factory=StackConfig)
    decoder: StackConfig = field(default_factory=StackConfig)

    #: Weight of this level's term inside L_rec.
    rec_loss_weight: float = 1.0

    #: Confine this encoder to the C_{l+1} units that form one level-(l+1) unit
    #: (tiled, never crossing).  Ignored on the top level, which stays global.
    encoder_block_local: bool = False

    #: Sliding window of this many units for an ``l < L`` encoder; overrides
    #: ``encoder_block_local``.  A rolling cache of window+1 is exact.
    encoder_window: int | None = None

    #: Level 1 only: the local decoder also reads the previous
    #: ``decoder_lookback`` chunks of content (span ``R_l + (1+K)*C_l``).
    decoder_lookback: int = 0

    #: SCALA: run this decoder as one sliding-window stream of ``M_l*(R_l+C_l)``
    #: positions of this window width, not ``M_l`` blocks; excludes lookback.
    decoder_stream: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.encoder, dict):
            self.encoder = StackConfig(**self.encoder)
        if isinstance(self.decoder, dict):
            self.decoder = StackConfig(**self.decoder)
        if self.decoder_stream and self.decoder_lookback:
            raise ValueError(
                "decoder_stream subsumes decoder_lookback -- a stream window of "
                f"{self.decoder_stream} positions already reaches further back "
                f"than {self.decoder_lookback} chunk(s), and for less compute. "
                "Set one or the other.")
        if self.decoder_stream and self.decoder.full_attn_every:
            raise ValueError(
                "decoder_stream needs every decoder layer windowed; "
                "full_attn_every would give one of them an unbounded cache")
        if self.decoder_stream is not None and self.decoder_stream < (
                self.converter_width + self.chunk_size):
            raise ValueError(
                f"decoder_stream={self.decoder_stream} is narrower than one "
                f"group ({self.converter_width} + {self.chunk_size}); the last "
                "slot of a group could not see its own conditioning vector")


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass
class ScalaConfig:
    """Full SCALA model configuration."""

    vocab_size: int = 99_584
    d_token: int = 1024                 # D_0, the token-level width
    max_seq_len: int = 8192

    levels: list[LevelConfig] = field(default_factory=lambda: [LevelConfig()])

    tie_word_embeddings: bool = False
    #: Multiply token embeddings by sqrt(d_token) at input.
    scale_embeddings: bool = True
    init_std: float = 0.02

    # -- objective ---------------------------------------------------------- #
    #: alpha in  L = L_token + alpha * L_rec.
    rec_loss_alpha: float = 0.3
    #: "cosine" (paper) | "smooth_l1" | "mse"
    rec_loss_kind: Literal["cosine", "smooth_l1", "mse"] = "cosine"
    #: Stop-grad on the encoder target inside L_rec, so the encoder cannot
    #: collapse toward an easily-reconstructable representation.
    rec_loss_detach_target: bool = True
    #: Lowest level L_rec scores.  2, not 1: at l == 1 generation emits a real
    #: ``embed(tok)``, so there is no substitution to correct.
    rec_loss_min_level: int = 2
    #: Per-unit probability of feeding the level-l decoder ``chunker(X^(l-2))``
    #: instead of the encoder state ``X^(l-1)`` -- what `chunkgen` does at inference.
    chunk_cond_prob: float = 0.0
    #: Feed the level-l decoder (l >= 2) its own ``X_hat^(l-1)_{<j}`` as content
    #: (Eq. 6) rather than ``X^(l-1)_{<j}``; C_l sequential passes per group.
    recursive_decoder_input: bool = False

    #: Also feed every local decoder ``X^(L)_{g-1}`` directly, beside the state
    #: that reached it through the cascade: wider converter input, no positions.
    global_skip: bool = False

    #: SCALA: share one module set across levels ``2..L-1`` (state_dict keys
    #: ``level_token/level_mid/level_cap``); ``levels`` stays fully expanded.
    tie_mid_levels: bool = False

    #: Output-logit softcap / z-loss for stability.
    z_loss_weight: float = 1e-4
    logit_softcap: float | None = None

    # -- Multi-Token Prediction (DeepSeek-V3) ------------------------------- #
    mtp_depth: int = 0                  # 0 disables MTP
    mtp_loss_weight: float = 0.3

    dtype: str = "bfloat16"

    # ----------------------------------------------------------------- utils #
    def __post_init__(self) -> None:
        self.levels = [
            LevelConfig(**lv) if isinstance(lv, dict) else lv for lv in self.levels
        ]
        if not self.levels:
            raise ValueError("ScalaConfig.levels must not be empty")
        if self.max_seq_len % self.chunk_product != 0:
            raise ValueError(
                f"max_seq_len={self.max_seq_len} must be a multiple of the chunk "
                f"product C_<=L={self.chunk_product}"
            )
        if self.tie_mid_levels:
            self._validate_tie()

    def _validate_tie(self) -> None:
        """SCALA preconditions for one MID module reused at every application:
        identical middle configs, one input width, spans bounded independently
        of depth, and no per-level side channels."""
        if self.n_levels < 3:
            raise ValueError(
                "tie_mid_levels needs >= 3 levels (level 1, >=1 MID, the CAP)")
        mid = self.levels[1]
        for i, lv in enumerate(self.levels[2:-1], start=3):
            if lv != mid:
                raise ValueError(
                    f"tie_mid_levels: level {i} differs from level 2; all "
                    "middle levels must be textually identical")
        if self.levels[0].encoder.d_model != mid.encoder.d_model:
            raise ValueError(
                "tie_mid_levels: the MID's width must equal level 1's "
                f"({mid.encoder.d_model} != {self.levels[0].encoder.d_model}); "
                "its d_below is level-1's width at the first application and "
                "its own width at every deeper one")
        if not (self.levels[0].encoder_window or self.levels[0].encoder_block_local):
            raise ValueError(
                "tie_mid_levels: level 1's encoder must be span-bounded "
                "(encoder_window or encoder_block_local) too, or RecGen's "
                "cache-windowing there would silently diverge from the "
                "trained (globally-attending) receptive field")
        if not mid.encoder_window:
            raise ValueError("tie_mid_levels: the MID encoder must be windowed "
                             "(encoder_window), or its span would depend on "
                             "which level it is applied at")
        if not mid.decoder_stream:
            raise ValueError("tie_mid_levels: the MID decoder must be a "
                             "bounded stream (decoder_stream)")
        if mid.encoder_block_local:
            raise ValueError("tie_mid_levels: encoder_block_local reads the "
                             "chunk size of the level above at build time; "
                             "use encoder_window")
        if mid.decoder_lookback:
            raise ValueError("tie_mid_levels: decoder_lookback is level-1 only")
        if self.global_skip:
            raise ValueError("tie_mid_levels: global_skip is refuted and adds "
                             "a per-level conditioning pathway; keep it off")
        if self.recursive_decoder_input:
            raise ValueError("tie_mid_levels: recursive_decoder_input is "
                             "incompatible with decoder_stream levels")

    # -- derived quantities ------------------------------------------------- #
    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def chunk_product(self) -> int:
        """C_<=L -- number of tokens summarised by one top-level unit."""
        p = 1
        for lv in self.levels:
            p *= lv.chunk_size
        return p

    def cumulative_chunk(self, level: int) -> int:
        """C_<=l for 1-indexed ``level`` (tokens per level-``level`` unit)."""
        p = 1
        for lv in self.levels[:level]:
            p *= lv.chunk_size
        return p

    def width(self, level: int) -> int:
        """D_l -- hidden width at level ``level`` (0 == token level)."""
        if level == 0:
            return self.d_token
        return self.levels[level - 1].encoder.d_model

    def units_at(self, level: int, seq_len: int | None = None) -> int:
        """M_l for a given sequence length."""
        seq_len = seq_len or self.max_seq_len
        return seq_len // self.cumulative_chunk(level)

    # -- (de)serialisation --------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScalaConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "ScalaConfig":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            import yaml  # lazy: only needed for YAML configs

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if "model" in data:  # allow a training config that nests the model
            data = data["model"]
        return cls.from_dict(data)

    def describe(self) -> str:
        lines = [
            f"ScalaConfig(vocab={self.vocab_size}, D0={self.d_token}, "
            f"L={self.n_levels}, C_<=L={self.chunk_product}, ctx={self.max_seq_len})"
        ]
        for i, lv in enumerate(self.levels, start=1):
            e, d = lv.encoder, lv.decoder
            lines.append(
                f"  level {i}: C={lv.chunk_size} R={lv.converter_width} "
                f"| enc d={e.d_model} n={e.n_layers} "
                f"moe={e.moe.n_routed_experts}x{e.moe.expert_inter_size}"
                f"/top{e.moe.top_k}+{e.moe.n_shared_experts}sh "
                f"| dec d={d.d_model} n={d.n_layers} "
                f"moe={d.moe.n_routed_experts}x{d.moe.expert_inter_size}"
                f"/top{d.moe.top_k}+{d.moe.n_shared_experts}sh"
                if d.moe.enabled
                else f"  level {i}: C={lv.chunk_size} R={lv.converter_width}"
            )
        return "\n".join(lines)
