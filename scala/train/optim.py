"""Muon + AdamW hybrid optimiser and the WSD learning-rate schedule.

Muon (arXiv:2502.16982) orthogonalises each weight-matrix update by Newton-
Schulz and scales it by ``0.2 * sqrt(max(fan_in, fan_out))`` to match AdamW's
update RMS, so an AdamW-tuned lr and weight decay transfer.  Muon takes every
2-D/3-D hidden matrix; AdamW takes embeddings, LM head, norms, gates and 1-D.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["Muon", "build_optimizer", "WSDSchedule", "GradNormGuard", "ZClip",
           "clip_grad_norm_"]


# --------------------------------------------------------------------------- #
# Newton-Schulz orthogonalisation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _add_stochastic_(p: Tensor, update: Tensor, alpha: float = 1.0) -> None:
    """``p += alpha * update`` with stochastic rounding when ``p`` is bf16.

    bf16 has 8 mantissa bits, so once ``|p| >> |update|`` a round-to-nearest
    add discards the update entirely.  Rounding up with probability equal to
    the position within the interval preserves it in expectation.
    """
    if p.dtype != torch.bfloat16:
        p.add_(update, alpha=alpha)
        return
    # work in fp32, then round the low 16 bits stochastically
    exact = p.float().add_(update.float(), alpha=alpha)
    bits = exact.view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    p.copy_(((bits + noise) & ~0xFFFF).view(torch.float32).to(torch.bfloat16))


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonal polar factor.

    Works on ``(..., m, n)``; leading dims batch, which is what stacked MoE
    expert weights ``(E, d_in, d_out)`` need.
    """
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)

    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mT

    # spectral-norm normalisation so the iteration is in its convergence basin
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


def _muon_scale(p: Tensor) -> float:
    """RMS-matching factor 0.2 * sqrt(max(fan_in, fan_out))."""
    return 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1]))


# --------------------------------------------------------------------------- #
# Muon
# --------------------------------------------------------------------------- #
class Muon(Optimizer):
    """Muon for 2-D/3-D hidden weights, with an AdamW group for everything else.

    Parameter groups carry ``use_muon: bool``.  Groups with ``use_muon=False``
    run a standard decoupled-weight-decay AdamW.
    """

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        state_dtype: torch.dtype | None = None,
    ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, adam_betas=adam_betas, adam_eps=adam_eps,
            use_muon=True,
        )
        #: dtype for the momentum / moment buffers.  ``None`` keeps fp32; bf16
        #: halves optimiser memory.
        self.state_dtype = state_dtype
        super().__init__(param_groups, defaults)

    def _buf_dtype(self, p: Tensor) -> torch.dtype:
        return self.state_dtype or torch.float32

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_local(p: Tensor) -> tuple[Tensor, Any]:
        """FSDP2 shards parameters as DTensors; Newton-Schulz needs the whole
        matrix, so gather it and remember how to put it back."""
        if hasattr(p, "full_tensor") and hasattr(p, "device_mesh"):
            return p.full_tensor(), (p.device_mesh, p.placements)
        return p, None

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            if group.get("use_muon", True):
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss

    # ------------------------------------------------------------------ #
    def _step_muon(self, group) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        mom, nesterov = group["momentum"], group["nesterov"]
        ns = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(
                    g, dtype=self._buf_dtype(p)
                )
            buf = st["momentum_buffer"]
            buf.lerp_(g.to(buf.dtype), 1.0 - mom)
            upd = g.lerp(buf.to(g.dtype), mom) if nesterov else buf.to(g.dtype)

            full, spec = self._to_local(upd)
            if full.ndim == 1:                       # shouldn't happen; be safe
                ortho = full / (full.norm() + 1e-7)
            else:
                ortho = zeropower_via_newtonschulz5(full, ns)
            ortho = ortho * _muon_scale(full)

            if spec is not None:
                from torch.distributed.tensor import distribute_tensor

                ortho = distribute_tensor(ortho, spec[0], spec[1])

            if wd:
                p.mul_(1.0 - lr * wd)
            _add_stochastic_(p, ortho, alpha=-lr)

    # ------------------------------------------------------------------ #
    def _step_adamw(self, group) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        b1, b2 = group["adam_betas"]
        eps = group["adam_eps"]

        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "step" not in st:
                st["step"] = 0
                bd = self._buf_dtype(p)
                st["exp_avg"] = torch.zeros_like(p, dtype=bd)
                st["exp_avg_sq"] = torch.zeros_like(p, dtype=bd)
            st["step"] += 1
            t = st["step"]
            m, v = st["exp_avg"], st["exp_avg_sq"]
            # `Optimizer.load_state_dict` casts float state to the *parameter's*
            # dtype, so after a bf16-master resume these buffers come back bf16
            # whatever they were created as; follow them rather than assume fp32.
            gf = g.to(m.dtype)
            m.lerp_(gf, 1.0 - b1)
            v.mul_(b2).addcmul_(gf, gf, value=1.0 - b2)
            mhat = (m / (1.0 - b1**t)).float()
            vhat = (v / (1.0 - b2**t)).float()
            if wd:
                p.mul_(1.0 - lr * wd)
            _add_stochastic_(p, mhat / (vhat.sqrt() + eps), alpha=-lr)


# --------------------------------------------------------------------------- #
# Parameter grouping
# --------------------------------------------------------------------------- #
#: substrings that force a parameter onto AdamW regardless of its rank
_ADAMW_NAME_HINTS = (
    "embed", "lm_head", "norm", "router.weight", "start_latent",
    "conv.weight", "sink", "query",
)


def classify_parameter(name: str, p: Tensor) -> str:
    """Return ``"muon"`` or ``"adamw"`` for one named parameter."""
    if p.ndim < 2:
        return "adamw"
    if any(h in name for h in _ADAMW_NAME_HINTS):
        return "adamw"
    return "muon"


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    adamw_lr_mult: float = 1.0,
    weight_decay: float = 0.1,
    embedding_weight_decay: float = 0.0,
    momentum: float = 0.95,
    ns_steps: int = 5,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    adam_eps: float = 1e-8,
    verbose: bool = True,
    state_dtype: torch.dtype | None = None,
) -> Muon:
    """Build the Muon/AdamW hybrid with three groups: muon, adamw, adamw-no-wd."""
    muon_p, adam_p, adam_nowd_p = [], [], []
    n_muon = n_adam = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        kind = classify_parameter(name, p)
        if kind == "muon":
            muon_p.append(p)
            n_muon += p.numel()
        elif "embed" in name or "lm_head" in name:
            (adam_p if embedding_weight_decay else adam_nowd_p).append(p)
            n_adam += p.numel()
        else:
            adam_nowd_p.append(p)
            n_adam += p.numel()

    groups = [
        dict(params=muon_p, use_muon=True, lr=lr, weight_decay=weight_decay),
        dict(params=adam_p, use_muon=False, lr=lr * adamw_lr_mult,
             weight_decay=embedding_weight_decay),
        dict(params=adam_nowd_p, use_muon=False, lr=lr * adamw_lr_mult,
             weight_decay=0.0),
    ]
    groups = [g for g in groups if g["params"]]

    if verbose and (not dist.is_initialized() or dist.get_rank() == 0):
        print(f"[optim] Muon params : {n_muon/1e6:9.1f}M "
              f"({len(muon_p)} tensors)")
        print(f"[optim] AdamW params: {n_adam/1e6:9.1f}M "
              f"({len(adam_p)+len(adam_nowd_p)} tensors)")

    return Muon(groups, lr=lr, weight_decay=weight_decay, momentum=momentum,
                ns_steps=ns_steps, adam_betas=adam_betas, adam_eps=adam_eps,
                state_dtype=state_dtype)


# --------------------------------------------------------------------------- #
# WSD schedule
# --------------------------------------------------------------------------- #
@dataclass
class WSDSchedule:
    """Warmup - Stable - Decay (trapezoid) learning-rate schedule.

    The stable plateau means ``total_steps`` need not be fixed up front:
    extending training only re-plans the decay phase.
    """

    total_steps: int
    warmup_steps: int = 2000
    decay_frac: float = 0.2          # fraction of total spent decaying
    min_lr_ratio: float = 0.03       # final LR as a fraction of peak
    decay_shape: str = "1-sqrt"      # "1-sqrt" | "cosine" | "linear"
    peak_lr: float = 3e-4

    def __post_init__(self):
        self.decay_steps = int(self.total_steps * self.decay_frac)
        self.stable_end = self.total_steps - self.decay_steps

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(self.warmup_steps, 1)
        if step < self.stable_end:
            return 1.0
        prog = (step - self.stable_end) / max(self.decay_steps, 1)
        prog = min(max(prog, 0.0), 1.0)
        if self.decay_shape == "cosine":
            shape = 0.5 * (1.0 + math.cos(math.pi * prog))
        elif self.decay_shape == "linear":
            shape = 1.0 - prog
        else:  # 1-sqrt
            shape = 1.0 - math.sqrt(prog)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * shape

    def lr(self, step: int) -> float:
        return self.peak_lr * self.factor(step)

    def apply(self, optimizer: Optimizer, step: int) -> float:
        f = self.factor(step)
        for g in optimizer.param_groups:
            base = g.setdefault("base_lr", g["lr"])
            g["lr"] = base * f
        return self.peak_lr * f


# --------------------------------------------------------------------------- #
# Gradient utilities
# --------------------------------------------------------------------------- #
def clip_grad_norm_(params: Iterable[Tensor], max_norm: float) -> Tensor:
    params = [p for p in params if p.grad is not None]
    if not params:
        return torch.zeros(())
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


class ZClip:
    """Adaptive gradient clipping (arXiv:2504.02507).

    Tracks an EMA of the gradient-norm mean and variance and clips at
    ``mu + z_thresh * sigma``; spikes are rescaled, not dropped.  Complements
    ``GradNormGuard``, which skips the step entirely.
    """

    def __init__(self, alpha: float = 0.97, z_thresh: float = 2.5,
                 warmup: int = 25, max_norm: float | None = None):
        self.alpha = alpha
        self.z_thresh = z_thresh
        self.warmup = warmup
        self.max_norm = max_norm
        self.mu = 0.0
        self.var = 0.0
        self.n = 0
        self.n_clipped = 0

    def threshold(self) -> float:
        sigma = math.sqrt(max(self.var, 0.0))
        return self.mu + self.z_thresh * sigma

    def __call__(self, params: Iterable[Tensor]) -> tuple[float, float]:
        """Clip in place; returns (grad_norm_before, applied_threshold)."""
        params = [p for p in params if p.grad is not None]
        if not params:
            return 0.0, float("inf")
        total = torch.norm(
            torch.stack([p.grad.detach().float().norm(2) for p in params]), 2
        )
        gn = float(total)
        self.n += 1

        if self.n <= self.warmup or not math.isfinite(gn):
            thresh = self.max_norm or float("inf")
        else:
            thresh = self.threshold()
            if self.max_norm:
                thresh = min(thresh, self.max_norm)

        if math.isfinite(gn) and gn > thresh > 0:
            scale = thresh / (gn + 1e-6)
            for p in params:
                p.grad.detach().mul_(scale)
            self.n_clipped += 1
            # a clipped step should not drag the statistics toward the outlier
            observed = thresh
        else:
            observed = gn

        if math.isfinite(observed):
            a = self.alpha
            delta = observed - self.mu
            self.mu = a * self.mu + (1 - a) * observed
            self.var = a * self.var + (1 - a) * delta * delta
        return gn, thresh


class GradNormGuard:
    """Skip a step whose grad norm is non-finite or above ``threshold`` times
    the running median, up to ``max_skip_frac`` of steps seen.
    """

    def __init__(self, window: int = 100, threshold: float = 4.0,
                 warmup: int = 200, max_skip_frac: float = 0.02):
        self.hist: deque[float] = deque(maxlen=window)
        self.threshold = threshold
        self.warmup = warmup
        self.max_skip_frac = max_skip_frac
        self.n_seen = 0
        self.n_skipped = 0

    def should_skip(self, grad_norm: float) -> bool:
        self.n_seen += 1
        if not math.isfinite(grad_norm):
            self.n_skipped += 1
            return True
        if len(self.hist) < self.warmup:
            self.hist.append(grad_norm)
            return False
        med = sorted(self.hist)[len(self.hist) // 2]
        skip = grad_norm > self.threshold * med
        if skip and self.n_skipped < self.max_skip_frac * self.n_seen:
            self.n_skipped += 1
            return True
        self.hist.append(grad_norm)
        return False
