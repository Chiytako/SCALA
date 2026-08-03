"""Pretraining loop for SCALA; single-GPU, DDP and FSDP2 share one path.

bf16 autocast, FSDP2 ``fully_shard`` per transformer block, selective
activation checkpointing, ``torch.compile``, Muon+AdamW on a WSD schedule with
a spike guard, aux-loss-free MoE bias updates (``gamma`` annealed to 0), MTP
weight annealed 0.3 -> 0.1, scheduled sampling of the decoder conditioning, DCP
checkpointing with full resume.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..model.config import ScalaConfig
from ..model.moe import update_expert_biases
from ..model.hierarchy import ScalaForCausalLM
from .optim import (
    GradNormGuard, WSDSchedule, ZClip, build_optimizer, clip_grad_norm_,
)

__all__ = ["TrainConfig", "Trainer"]


# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    # -- what to train ---------------------------------------------------- #
    model_config: str = "configs/base_8b_a1b.yaml"
    data_root: str = "data/tokens"
    out_dir: str = "runs/scala-8b-a1b"
    run_name: str = "scala-8b-a1b"

    # -- token budget ----------------------------------------------------- #
    seq_len: int = 8192
    micro_batch_size: int = 1
    global_batch_tokens: int = 2_097_152      # 2M tokens/step
    total_tokens: float = 120e9
    #: ramp the global batch from this fraction up to 1.0 over bs_warmup_tokens
    bs_warmup_start_frac: float = 0.25
    bs_warmup_tokens: float = 8e9

    # -- optimiser -------------------------------------------------------- #
    lr: float = 3.0e-4
    adamw_lr_mult: float = 1.0
    weight_decay: float = 0.1
    embedding_weight_decay: float = 0.0
    momentum: float = 0.95
    ns_steps: int = 5
    adam_betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    #: "fixed" clips at grad_clip; "zclip" clips at mu + z*sigma of an EMA of
    #: the gradient-norm mean/variance (arXiv:2504.02507).
    grad_clip_mode: str = "fixed"
    zclip_alpha: float = 0.97
    zclip_z: float = 2.5
    warmup_steps: int = 2000
    decay_frac: float = 0.2
    min_lr_ratio: float = 0.03
    decay_shape: str = "1-sqrt"
    spike_guard: bool = True

    # -- objective schedules ---------------------------------------------- #
    #: fraction of training after which the MoE routing bias is frozen (gamma=0)
    moe_bias_freeze_frac: float = 0.95
    mtp_weight_start: float = 0.3
    mtp_weight_end: float = 0.1
    self_cond_prob_end: float = 0.25
    self_cond_ramp_frac: float = 0.5

    # -- two-phase data curriculum --------------------------------------- #
    # The second phase re-weights the sampler over the same shards.
    #: fraction of training (at the end) that uses `midtrain_weights`
    midtrain_frac: float = 0.0
    #: source name -> sampling weight during the mid-training phase
    midtrain_weights: dict = field(default_factory=dict)
    #: source name -> weight during the *first* phase; omitted sources keep their
    #: manifest weight, so listing mid-training-only sources at 0.0 excludes them.
    pretrain_weights: dict = field(default_factory=dict)

    # -- systems ---------------------------------------------------------- #
    parallel: str = "fsdp"              # "none" | "ddp" | "fsdp"
    dtype: str = "bfloat16"
    compile: bool = True
    compile_mode: str = "default"
    activation_checkpointing: str = "selective"   # "none"|"selective"|"full"
    ac_every_n: int = 2
    num_workers: int = 4
    seed: int = 20260726
    #: Data order and shard choice; ``None`` follows ``seed``, which ties data
    #: variance to initialisation variance.
    data_seed: int | None = None
    #: Fraction of every shard reserved for `evaluate()` and never trained on.
    holdout_frac: float = 0.02
    #: "float32" (default) or "bfloat16" for the Muon/AdamW state buffers;
    #: bf16 halves optimiser memory.
    optimizer_state_dtype: str = "float32"
    #: "float32" (default) or "bfloat16" for the *master* weights.  bf16 masters
    #: need stochastic rounding on the update (see optim._add_stochastic_).
    param_dtype: str = "float32"
    #: Hard cap on the fraction of device memory this process may allocate; 0
    #: disables.  Needed where GPU and OS share one pool (unified memory).
    max_memory_fraction: float = 0.0

    # -- io --------------------------------------------------------------- #
    log_every: int = 10
    eval_every: int = 1000
    eval_batches: int = 50
    save_every: int = 2000
    keep_last: int = 3
    wandb_project: Optional[str] = None
    resume: str = "auto"                # "auto" | "none" | <path>

    def __post_init__(self) -> None:
        # PyYAML follows YAML 1.1, where `4.0e6` (no exponent sign) parses as a
        # *string*; coerce every field back to its declared type.
        for name, f in self.__dataclass_fields__.items():
            v = getattr(self, name)
            if not isinstance(v, str):
                continue
            if f.type in ("float", "int") or f.type.startswith(("float", "int")):
                try:
                    setattr(self, name, int(float(v)) if "int" in f.type
                            else float(v))
                except ValueError:
                    pass
            elif f.type == "bool":
                setattr(self, name, v.lower() in ("1", "true", "yes", "on"))
        self.total_tokens = float(self.total_tokens)
        self.bs_warmup_tokens = float(self.bs_warmup_tokens)
        self.global_batch_tokens = int(self.global_batch_tokens)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        import yaml

        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        d = d.get("train", d)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
def _is_master() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log0(*a: Any, **kw: Any) -> None:
    if _is_master():
        print(*a, **kw, flush=True)


# --------------------------------------------------------------------------- #
class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        # FSDP2 needs a process group even on one rank: `fully_shard` builds a
        # DeviceMesh.
        needs_pg = self.world_size > 1 or cfg.parallel == "fsdp"
        if needs_pg and not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29511")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", str(self.world_size))
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                world_size=self.world_size, rank=self.rank,
            )
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            if cfg.max_memory_fraction > 0:
                torch.cuda.set_per_process_memory_fraction(
                    cfg.max_memory_fraction, self.local_rank
                )
                total = torch.cuda.get_device_properties(
                    self.local_rank).total_memory / 2**30
                log0(f"[mem] capped at {cfg.max_memory_fraction:.0%} of "
                     f"{total:.0f} GiB = {total*cfg.max_memory_fraction:.0f} GiB")
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(cfg.seed + self.rank)

        self.model_cfg = ScalaConfig.load(cfg.model_config)
        self.model_cfg.max_seq_len = cfg.seq_len
        self.chunk_product = self.model_cfg.chunk_product

        self.out_dir = Path(cfg.out_dir)
        if _is_master():
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.model_cfg.save(self.out_dir / "model_config.json")
            (self.out_dir / "train_config.json").write_text(
                json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8"
            )

        self.step = 0
        self.tokens_seen = 0
        self._build_model()
        self._build_optimizer()
        self._build_data()
        self._build_logging()
        self._maybe_resume()

    # ------------------------------------------------------------------ #
    # model
    # ------------------------------------------------------------------ #
    def _build_model(self) -> None:
        cfg = self.cfg
        torch.set_float32_matmul_precision("high")
        # Used even on a single GPU: FSDP2's MixedPrecisionPolicy is what makes
        # parameters and gradients bf16 (autocast alone keeps both fp32).
        use_fsdp = cfg.parallel == "fsdp" and torch.cuda.is_available()

        if use_fsdp:
            with torch.device("meta"):
                model = ScalaForCausalLM(self.model_cfg)
            model = self._apply_fsdp(model)
            model.to_empty(device=self.device)
            self._init_weights_after_meta(model)
        else:
            model = ScalaForCausalLM(self.model_cfg).to(self.device)
            if cfg.param_dtype == "bfloat16":
                model = model.to(torch.bfloat16)
                log0("[model] bf16 master weights "
                     "(updates use stochastic rounding)")
            if cfg.parallel == "ddp" and self.world_size > 1:
                model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.local_rank],
                    gradient_as_bucket_view=True,
                )

        self.model = model
        self.raw_model = model.module if hasattr(model, "module") else model

        self._apply_activation_checkpointing()
        if cfg.compile:
            self._apply_compile()

        n = sum(p.numel() for p in self.raw_model.parameters())
        log0(f"[model] {n/1e9:.3f}B parameters (sharded view; "
             f"world_size={self.world_size})")
        log0(self.model_cfg.describe())

    def _apply_fsdp(self, model: ScalaForCausalLM) -> ScalaForCausalLM:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        from ..model.layers import TransformerBlock

        if model.cfg.tie_mid_levels:
            # `modules()` deduplicates the shared MID; cross-application
            # gradient accumulation through a sharded module is unverified.
            raise NotImplementedError(
                "tie_mid_levels + FSDP is unverified; use parallel: none")

        # On one rank there is no reduction to do; fp32 reduce_dtype would only
        # cost a full-size fp32 gradient buffer.
        mp = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=(torch.float32 if self.world_size > 1
                          else torch.bfloat16),
            output_dtype=torch.bfloat16,
        )
        # Per-block sharding gathers parameters just-in-time; the MoE expert
        # weights dominate memory and live here.
        for m in model.modules():
            if isinstance(m, TransformerBlock):
                fully_shard(m, mp_policy=mp)
        # The root (embedding, possibly tied LM head, final norm) is re-run
        # inside the checkpointed cross-entropy slice during backward, so it
        # must stay gathered: a resharded root has already freed those
        # parameters ("setStorage ... storage of size 0").
        fully_shard(model, mp_policy=mp, reshard_after_forward=False)
        return model

    @torch.no_grad()
    def _init_weights_after_meta(self, model: nn.Module) -> None:
        """Re-run the initialisers that meta-device construction skipped.

        Modules exposing ``reset_parameters`` are asked directly; only the
        remainder falls back to a shape-based rule.
        """
        gen = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        handled: set[int] = set()

        for mod in model.modules():
            fn = getattr(mod, "reset_parameters", None)
            if callable(fn) and type(mod).reset_parameters is not \
                    getattr(nn.Module, "reset_parameters", None):
                try:
                    fn()
                except Exception:  # noqa: BLE001 - fall through to the generic rule
                    continue
                handled.update(id(p) for p in mod.parameters(recurse=False))

        for name, p in model.named_parameters():
            if id(p) in handled:
                continue
            local = p.to_local() if hasattr(p, "to_local") else p
            if local.numel() == 0:
                continue
            if local.ndim == 1:
                local.fill_(1.0 if "norm" in name else 0.0)
            elif "conv.weight" in name:
                local.zero_()
            else:
                std = (self.model_cfg.init_std
                       if ("embed" in name or "lm_head" in name)
                       else local.shape[-2] ** -0.5)
                local.normal_(0.0, std, generator=gen)

        for name, b in model.named_buffers():
            if any(k in name for k in ("expert_bias", "load_counter",
                                       "pid_integral", "pid_prev_err")):
                b.zero_()

        self._check_initialisation(model)

    @torch.no_grad()
    def _check_initialisation(self, model: nn.Module) -> None:
        """Raise if any weight initialised to all zeros (non-finite grads)."""
        allowed_zero = ("conv.weight", "w_gate.weight", "sink", "bias")
        bad = []
        for name, p in model.named_parameters():
            local = p.to_local() if hasattr(p, "to_local") else p
            if local.numel() == 0 or any(k in name for k in allowed_zero):
                continue
            if not torch.any(local != 0):
                bad.append(name)
        if bad:
            raise RuntimeError(
                "these parameters initialised to all zeros, which will produce "
                f"non-finite gradients: {bad[:8]}"
                + (f" (+{len(bad)-8} more)" if len(bad) > 8 else "")
            )
        log0(f"[init] {sum(1 for _ in model.parameters())} tensors initialised, "
             "no all-zero weights")

    def _apply_activation_checkpointing(self) -> None:
        if self.cfg.activation_checkpointing == "none":
            return
        from ..model.layers import TransformerStack

        full = self.cfg.activation_checkpointing == "full"
        n = 0
        for stack in self.raw_model.modules():
            if not isinstance(stack, TransformerStack):
                continue
            every = 1 if full else max(self.cfg.ac_every_n, 1)
            stack.ac_layers = set(range(0, len(stack.layers), every))
            n += len(stack.ac_layers)
        self.raw_model.set_gradient_checkpointing(True)
        log0(f"[ac] recomputing {n} blocks "
             f"({self.cfg.activation_checkpointing})")

    def _apply_compile(self) -> None:
        from ..model.layers import TransformerBlock

        os.environ.setdefault("TORCHDYNAMO_CACHE_SIZE_LIMIT", "256")
        # fuses the LM-head GEMM with log-softmax so the (chunk, vocab) fp32
        # logits are never materialised; the compiled region is the whole
        # checkpointed unit, so save and recompute see the same graph
        self.raw_model.compile_loss()

        if self.cfg.activation_checkpointing != "none":
            # Compiling *inside* a checkpointed block is not: recompute takes a
            # different path through the MoE dispatch and autograd rejects it
            # with "recomputed metadata differs from saved metadata".
            log0("[compile] loss only (block compile disabled under "
                 "activation checkpointing)")
            return
        if self.model_cfg.tie_mid_levels:
            # A compiled block re-applied at levels with different sequence
            # shapes would recompile per level or silently specialise.
            log0("[compile] loss only (block compile disabled under "
                 "tie_mid_levels)")
            return
        n = 0
        for m in self.raw_model.modules():
            if isinstance(m, TransformerBlock):
                m.attn.compile(mode=self.cfg.compile_mode, dynamic=False)
                if m.is_moe:
                    # only the grouped GEMMs; routing stays eager
                    m.ffn.experts.compile(mode=self.cfg.compile_mode, dynamic=True)
                else:
                    m.ffn.compile(mode=self.cfg.compile_mode, dynamic=False)
                n += 1
        log0(f"[compile] compiled {n} blocks (mode={self.cfg.compile_mode})")

    # ------------------------------------------------------------------ #
    def _build_optimizer(self) -> None:
        cfg = self.cfg
        self.optimizer = build_optimizer(
            self.raw_model, lr=cfg.lr, adamw_lr_mult=cfg.adamw_lr_mult,
            weight_decay=cfg.weight_decay,
            embedding_weight_decay=cfg.embedding_weight_decay,
            momentum=cfg.momentum, ns_steps=cfg.ns_steps,
            adam_betas=tuple(cfg.adam_betas),
            state_dtype=(torch.bfloat16
                         if cfg.optimizer_state_dtype == "bfloat16" else None),
        )
        self.total_steps = max(int(cfg.total_tokens // cfg.global_batch_tokens), 1)
        self.schedule = WSDSchedule(
            total_steps=self.total_steps, warmup_steps=cfg.warmup_steps,
            decay_frac=cfg.decay_frac, min_lr_ratio=cfg.min_lr_ratio,
            decay_shape=cfg.decay_shape, peak_lr=cfg.lr,
        )
        self.guard = GradNormGuard() if cfg.spike_guard else None
        self.zclip = (ZClip(cfg.zclip_alpha, cfg.zclip_z, max_norm=cfg.grad_clip)
                      if cfg.grad_clip_mode == "zclip" else None)
        log0(f"[optim] total_steps={self.total_steps:,} "
             f"({cfg.total_tokens/1e9:.1f}B tokens @ "
             f"{cfg.global_batch_tokens/1e6:.2f}M tok/step)")

    def _build_data(self, weight_overrides: dict | None = None,
                    midtrain: bool | None = None) -> None:
        from ..data.dataset import MixtureSpec, build_dataloader

        cfg = self.cfg
        if weight_overrides is None and cfg.pretrain_weights:
            weight_overrides = dict(cfg.pretrain_weights)
        if midtrain is None:
            midtrain = bool(weight_overrides) and not cfg.pretrain_weights
        mix = MixtureSpec.from_manifest(cfg.data_root, weight_overrides)
        if _is_master():
            log0(mix.describe())
            ep = mix.epochs_at_budget(int(cfg.total_tokens))
            log0("epochs at budget: " +
                 ", ".join(f"{k}={v:.2f}" for k, v in ep.items()))
        data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
        self.loader = build_dataloader(
            cfg.data_root, cfg.seq_len, cfg.micro_batch_size,
            self.chunk_product, self.rank, self.world_size,
            data_seed + (1 if midtrain else 0), cfg.num_workers,
            weight_overrides=weight_overrides,
            holdout_frac=cfg.holdout_frac,
        )
        self.data_iter = iter(self.loader)
        self.in_midtrain = midtrain

        # a second loader over the reserved holdout tail, for `evaluate()`
        self.eval_iter = None
        if cfg.holdout_frac > 0:
            self.eval_loader = build_dataloader(
                cfg.data_root, cfg.seq_len, cfg.micro_batch_size,
                self.chunk_product, self.rank, self.world_size,
                data_seed, min(cfg.num_workers, 2),
                weight_overrides=weight_overrides,
                holdout_frac=cfg.holdout_frac, split="holdout",
            )
            self.eval_iter = iter(self.eval_loader)

    def _maybe_enter_midtrain(self) -> None:
        """Swap to the high-quality mixture for the final `midtrain_frac`."""
        cfg = self.cfg
        if not cfg.midtrain_frac or not cfg.midtrain_weights:
            return
        if getattr(self, "in_midtrain", False):
            return
        if self.progress < 1.0 - cfg.midtrain_frac:
            return
        log0(f"[data] entering mid-training phase at step {self.step} "
             f"({self.progress:.1%}): re-weighting to "
             f"{cfg.midtrain_weights}")
        if hasattr(self, "loader") and self.loader is not None:
            del self.data_iter, self.loader
        self._build_data(weight_overrides=dict(cfg.midtrain_weights),
                         midtrain=True)

    def _build_logging(self) -> None:
        self.log_path = self.out_dir / "log.jsonl"
        self.wandb = None
        if self.cfg.wandb_project and _is_master():
            try:
                import wandb

                wandb.init(project=self.cfg.wandb_project, name=self.cfg.run_name,
                           config=asdict(self.cfg))
                self.wandb = wandb
            except Exception as e:  # noqa: BLE001
                log0(f"[wandb] disabled: {e}")

    # ------------------------------------------------------------------ #
    # progress-dependent schedules
    # ------------------------------------------------------------------ #
    @property
    def progress(self) -> float:
        return min(self.step / max(self.total_steps, 1), 1.0)

    def current_grad_accum(self) -> int:
        cfg = self.cfg
        tok_per_micro = cfg.micro_batch_size * cfg.seq_len * self.world_size
        full = max(cfg.global_batch_tokens // tok_per_micro, 1)
        if cfg.bs_warmup_tokens <= 0:
            return full
        f = min(self.tokens_seen / cfg.bs_warmup_tokens, 1.0)
        frac = cfg.bs_warmup_start_frac + (1 - cfg.bs_warmup_start_frac) * f
        return max(int(round(full * frac)), 1)

    def current_moe_gamma(self) -> float:
        base = self.model_cfg.levels[0].encoder.moe.bias_update_rate
        return 0.0 if self.progress >= self.cfg.moe_bias_freeze_frac else base

    def current_mtp_weight(self) -> float:
        c = self.cfg
        return c.mtp_weight_start + (c.mtp_weight_end - c.mtp_weight_start) * \
            min(self.progress / 0.7, 1.0)

    def current_self_cond(self) -> float:
        c = self.cfg
        if c.self_cond_ramp_frac <= 0:
            return c.self_cond_prob_end
        return c.self_cond_prob_end * min(self.progress / c.self_cond_ramp_frac, 1.0)

    # ------------------------------------------------------------------ #
    def next_batch(self) -> dict[str, torch.Tensor]:
        try:
            b = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            b = next(self.data_iter)
        return {k: v.to(self.device, non_blocking=True) for k, v in b.items()}

    def _autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def _grad_sync_ctx(self, sync: bool):
        if sync:
            if hasattr(self.model, "set_requires_gradient_sync"):
                self.model.set_requires_gradient_sync(True)
            return nullcontext()
        if hasattr(self.model, "no_sync"):          # DDP
            return self.model.no_sync()
        if hasattr(self.model, "set_requires_gradient_sync"):   # FSDP2
            self.model.set_requires_gradient_sync(False)
        return nullcontext()

    # ------------------------------------------------------------------ #
    def train(self) -> None:
        cfg = self.cfg
        self.model.train()
        t0 = time.time()
        window = {"loss": 0.0, "token": 0.0, "rec": 0.0, "mtp": 0.0,
                  "aux": 0.0, "n": 0.0}

        while self.step < self.total_steps:
            self._maybe_enter_midtrain()
            accum = self.current_grad_accum()
            self.raw_model.cfg.mtp_loss_weight = self.current_mtp_weight()
            self.raw_model.self_cond_prob = self.current_self_cond()

            self.optimizer.zero_grad(set_to_none=True)
            for micro in range(accum):
                batch = self.next_batch()
                with self._grad_sync_ctx(micro == accum - 1), self._autocast():
                    out = self.model(batch["input_ids"], labels=batch["labels"],
                                     return_logits=False)
                    loss = out.loss / accum
                loss.backward()

                window["loss"] += float(out.loss.detach())
                window["token"] += float(out.loss_token.detach())
                if out.loss_rec is not None:
                    window["rec"] += float(out.loss_rec.detach())
                if out.loss_mtp is not None:
                    window["mtp"] += float(out.loss_mtp.detach())
                if out.loss_aux is not None:
                    window["aux"] += float(out.loss_aux.detach())
                window["n"] += 1

            if self.zclip is not None:
                gn, _ = self.zclip(self.raw_model.parameters())
            else:
                gn = float(clip_grad_norm_(self.raw_model.parameters(),
                                           cfg.grad_clip))
            skipped = self.guard.should_skip(gn) if self.guard else False

            lr = self.schedule.apply(self.optimizer, self.step)
            if not skipped:
                self.optimizer.step()
            # reads MaxVio off this step's load, then clears the counters
            self.moe_stats = update_expert_biases(
                self.raw_model, gamma=self.current_moe_gamma()
            )

            self.step += 1
            self.tokens_seen += (accum * cfg.micro_batch_size * cfg.seq_len
                                 * self.world_size)

            if self.step % cfg.log_every == 0:
                self._log(window, lr, gn, accum, skipped, t0)
                window = {k: 0.0 for k in window}
                t0 = time.time()
            if cfg.eval_every and self.step % cfg.eval_every == 0:
                self.evaluate()
            if cfg.save_every and self.step % cfg.save_every == 0:
                self.save_checkpoint()

        self.save_checkpoint(final=True)
        log0("[train] done")

    # ------------------------------------------------------------------ #
    def _log(self, w: dict, lr: float, gnorm: float, accum: int,
             skipped: bool, t0: float) -> None:
        n = max(w["n"], 1)
        dt = time.time() - t0
        toks = (self.cfg.log_every * accum * self.cfg.micro_batch_size
                * self.cfg.seq_len * self.world_size)
        rec = {
            "step": self.step,
            "tokens": self.tokens_seen,
            "loss": w["loss"] / n,
            "loss_token": w["token"] / n,
            "loss_rec": w["rec"] / n,
            "loss_mtp": w["mtp"] / n,
            "loss_aux": w["aux"] / n,
            "ppl": math.exp(min(w["token"] / n, 20)),
            "lr": lr,
            "grad_norm": gnorm,
            "grad_accum": accum,
            "skipped": bool(skipped),
            "tok_per_s": toks / max(dt, 1e-6),
            "mtp_w": self.current_mtp_weight(),
            "self_cond": self.current_self_cond(),
            "moe_gamma": self.current_moe_gamma(),
        }
        rec.update(getattr(self, "moe_stats", {}) or {})
        if _is_master():
            log0(f"step {rec['step']:>7} | {rec['tokens']/1e9:6.2f}B tok | "
                 f"loss {rec['loss']:.4f} | ce {rec['loss_token']:.4f} | "
                 f"ppl {rec['ppl']:8.2f} | rec {rec['loss_rec']:.4f} | "
                 f"lr {lr:.2e} | gn {gnorm:.2f} | "
                 f"vio {rec.get('moe/maxvio_mean', float('nan')):.3f} | "
                 f"{rec['tok_per_s']/1e3:.1f}K tok/s")
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if self.wandb:
                self.wandb.log(rec, step=self.step)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        held_out = self.eval_iter is not None
        tot, n = 0.0, 0
        for _ in range(self.cfg.eval_batches):
            if held_out:
                try:
                    b = next(self.eval_iter)
                except StopIteration:
                    self.eval_iter = iter(self.eval_loader)
                    b = next(self.eval_iter)
                b = {k: v.to(self.device, non_blocking=True) for k, v in b.items()}
                b.setdefault("labels", b["input_ids"])
            else:
                b = self.next_batch()
            with self._autocast():
                out = self.model(b["input_ids"], labels=b["labels"],
                                 return_logits=False)
            tot += float(out.loss_token)
            n += 1
        self.model.train()
        ce = tot / max(n, 1)
        tag = "eval" if held_out else "eval/train-stream"
        res = {f"{tag}/loss": ce, f"{tag}/ppl": math.exp(min(ce, 20))}
        log0(f"  [{tag}] step {self.step}: ce={ce:.4f} "
             f"ppl={math.exp(min(ce, 20)):.2f}")
        if self.wandb:
            self.wandb.log(res, step=self.step)
        return res

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, final: bool = False) -> None:
        path = self.out_dir / ("final" if final else f"step-{self.step:08d}")
        meta = {"step": self.step, "tokens_seen": self.tokens_seen}

        if self.world_size > 1:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict

            msd, osd = get_state_dict(self.raw_model, self.optimizer)
            dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(path))
        else:
            path.mkdir(parents=True, exist_ok=True)
            torch.save({"model": self.raw_model.state_dict(),
                        "optim": self.optimizer.state_dict()}, path / "state.pt")
        if _is_master():
            path.mkdir(parents=True, exist_ok=True)
            (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            self.model_cfg.save(path / "model_config.json")
            self._prune_checkpoints()
            log0(f"  [ckpt] saved {path}")
        if dist.is_initialized():
            dist.barrier()

    def _prune_checkpoints(self) -> None:
        import shutil

        ck = sorted(self.out_dir.glob("step-*"))
        for p in ck[: max(len(ck) - self.cfg.keep_last, 0)]:
            shutil.rmtree(p, ignore_errors=True)

    def _checkpoint_complete(self, path: Path) -> bool:
        """Complete iff ``meta.json`` exists and parses.  ``save_checkpoint``
        writes the state first and ``meta.json`` last, so meta is a completion
        marker: a save killed mid-write leaves a ``step-*`` dir with no meta.
        """
        try:
            json.loads((path / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if self.world_size == 1 and not (path / "state.pt").exists():
            return False
        return True

    def _maybe_resume(self) -> None:
        cfg = self.cfg
        # `--resume none` on the CLI arrives as Python None, not the string
        if cfg.resume in (None, "", "none", "None"):
            return
        if cfg.resume == "auto":
            path = None
            for cand in sorted(self.out_dir.glob("step-*"), reverse=True):
                if self._checkpoint_complete(cand):
                    path = cand
                    break
                log0(f"[resume] skipping incomplete checkpoint {cand.name} "
                     "(killed mid-save)")
            if path is None:
                return
        else:
            path = Path(cfg.resume)
            if not path.exists():
                log0(f"[resume] {path} not found, starting fresh")
                return
            if not self._checkpoint_complete(path):
                raise SystemExit(
                    f"[resume] {path} is incomplete (no readable meta.json) "
                    "-- it was probably killed mid-save; pick another "
                    "checkpoint or pass resume: auto")

        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if self.world_size > 1:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import (
                get_state_dict, set_state_dict,
            )

            msd, osd = get_state_dict(self.raw_model, self.optimizer)
            state = {"model": msd, "optim": osd}
            dcp.load(state, checkpoint_id=str(path))
            set_state_dict(self.raw_model, self.optimizer,
                           model_state_dict=state["model"],
                           optim_state_dict=state["optim"])
        else:
            sd = torch.load(path / "state.pt", map_location=self.device,
                            weights_only=False)
            self.raw_model.load_state_dict(sd["model"])
            self.optimizer.load_state_dict(sd["optim"])
        self.step = meta["step"]
        self.tokens_seen = meta["tokens_seen"]
        log0(f"[resume] from {path} at step {self.step} "
             f"({self.tokens_seen/1e9:.2f}B tokens)")
