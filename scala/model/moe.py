"""Mixture-of-Experts layer with auxiliary-loss-free load balancing.

Fine-grained routed experts plus always-on shared expert(s); sigmoid gate,
top-k over biased scores with gate values from the unbiased affinities; the
per-expert bias is moved by a sign/PID rule after each optimiser step, not by
gradient.  A small sequence-wise balance loss and a router z-loss are kept.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoEConfig

__all__ = ["MoELayer", "MoEAuxLosses", "moe_aux_losses", "reset_moe_aux_losses",
           "update_expert_biases"]


# --------------------------------------------------------------------------- #
# Aux-loss collection: side losses go into a module-level accumulator the
# training step drains, instead of through every forward signature.
# --------------------------------------------------------------------------- #
@dataclass
class MoEAuxLosses:
    balance: torch.Tensor | float = 0.0
    router_z: torch.Tensor | float = 0.0
    n_layers: int = 0

    def total(self) -> torch.Tensor | float:
        return self.balance + self.router_z


_AUX = MoEAuxLosses()


def reset_moe_aux_losses() -> None:
    global _AUX
    _AUX = MoEAuxLosses()


def moe_aux_losses() -> MoEAuxLosses:
    return _AUX


def _accumulate(balance, router_z) -> None:
    _AUX.balance = _AUX.balance + balance
    _AUX.router_z = _AUX.router_z + router_z
    _AUX.n_layers += 1


# --------------------------------------------------------------------------- #
# Grouped GEMM helpers
# --------------------------------------------------------------------------- #
_GROUPED_MM = getattr(torch, "_grouped_mm", None)

# Ragged offsets make `torch._grouped_mm` raise a device-side assert, which no
# try/except can recover from; opt in with PHOTON_MOE_GROUPED_MM=1 only where
# the kernel accepts unaligned offsets.
_grouped_mm_ok = (_GROUPED_MM is not None
                  and os.environ.get("PHOTON_MOE_GROUPED_MM", "0") == "1")

# Offsets that are multiples of 16 are accepted; PHOTON_MOE_ALIGNED=0 falls back
# to the padded batched GEMM.
def _kernel_supports_grouped_mm() -> bool:
    """Probe the kernel instead of inferring support from the device capability.

    Support is gated on compute capability exactly (9, 0), not ">=".  On ROCm
    the capability is derived from the gfx arch, so gfx90a also reports (9, 0)
    while the kernel raises -- hence HIP builds are rejected outright.  The
    probe uses 16-aligned offsets so it cannot trip the device-side assert.
    """
    if _GROUPED_MM is None or not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None) is not None:
        return False
    try:
        if torch.cuda.get_device_capability(0) != (9, 0):
            return False
        x = torch.zeros(32, 8, dtype=torch.bfloat16, device="cuda")
        w = torch.zeros(2, 8, 8, dtype=torch.bfloat16, device="cuda")
        offs = torch.tensor([16, 32], dtype=torch.int32, device="cuda")
        _GROUPED_MM(x, w, offs=offs)
        return True
    except Exception:  # noqa: BLE001
        return False


_ALIGNED_OK = (os.environ.get("PHOTON_MOE_ALIGNED", "1") == "1"
               and _kernel_supports_grouped_mm())


def _grouped_linear(x: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor,
                    ends: list[int] | None = None):
    """Segment ("jagged batch") matmul.

    x: (N, din), rows already sorted by expert id.
    w: (E, din, dout)
    offsets: (E,) int32 *exclusive-end* offset of each expert's row segment.
    ends: the same offsets already on the host; pass it to avoid a
        device->host sync per call.

    Uses ``torch._grouped_mm`` when available (one fused GEMM, no padding, no
    host sync), otherwise a per-expert loop.
    """
    global _grouped_mm_ok
    # autocast rewrites nn.Linear but not raw-parameter matmuls, and the
    # residual stream stays fp32; without this cast the fused kernel refuses
    # every call and silently drops to the loop
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
        x = x.to(torch.get_autocast_dtype("cuda"))
    w = w.to(x.dtype)

    if _grouped_mm_ok and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
        try:
            return _GROUPED_MM(x, w, offs=offsets)
        except (RuntimeError, NotImplementedError) as e:  # unsupported arch/shape
            _grouped_mm_ok = False
            import warnings

            warnings.warn(f"torch._grouped_mm unavailable ({e}); using the "
                          "per-expert loop", stacklevel=2)

    if ends is None:
        ends = offsets.tolist()
    out = x.new_zeros(x.shape[0], w.shape[-1])
    start = 0
    for e, end in enumerate(ends):
        if end > start:
            out[start:end] = x[start:end] @ w[e]
        start = end
    return out


def _aligned_grouped_mm(x_sorted: torch.Tensor, w: torch.Tensor,
                        dest: torch.Tensor, total: int):
    """Fused grouped GEMM with the minimum padding the kernel will accept.

    Each expert's segment is rounded up to 16 rows (the alignment
    ``torch._grouped_mm`` requires) and scattered into that layout: at most 15
    padding rows per expert, bit-exact against a per-expert matmul.
    """
    buf = x_sorted.new_zeros(total, w.shape[-2])
    buf.index_copy_(0, dest, x_sorted)
    return buf, w


def _padded_bmm(x_sorted: torch.Tensor, w: torch.Tensor, slot: torch.Tensor,
                capacity: int, n_experts: int):
    """Segment matmul as ONE batched GEMM.

    ``slot[i]`` is the destination row of sorted-row ``i`` inside a dense
    ``(E, capacity, ...)`` buffer, or negative for rows that overflowed
    capacity.  Scatter, one ``bmm``, gather back.
    """
    din, dout = w.shape[-2], w.shape[-1]
    buf = x_sorted.new_zeros(n_experts * capacity, din)
    valid = slot >= 0
    if valid.all():
        buf.index_copy_(0, slot, x_sorted)
    else:
        idx = slot[valid]
        buf.index_copy_(0, idx, x_sorted[valid])
    y = torch.bmm(buf.view(n_experts, capacity, din), w.to(x_sorted.dtype))
    y = y.view(n_experts * capacity, dout)
    out = y.index_select(0, slot.clamp_min(0))
    if not valid.all():
        out = out * valid.unsqueeze(-1)
    return out


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
class Router(nn.Module):
    def __init__(self, cfg: MoEConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, d_model))
        nn.init.normal_(self.weight, std=d_model**-0.5)
        # routing bias: a buffer, not a parameter -- updated by the bias
        # controller, never by autograd
        self.register_buffer(
            "expert_bias", torch.zeros(cfg.n_routed_experts), persistent=True
        )
        # Token counts since the last bias update (all-reduced by the trainer).
        self.register_buffer(
            "load_counter", torch.zeros(cfg.n_routed_experts), persistent=False
        )
        # PID controller state (unused when bias_controller == "sign")
        self.register_buffer("pid_integral",
                             torch.zeros(cfg.n_routed_experts), persistent=True)
        self.register_buffer("pid_prev_err",
                             torch.zeros(cfg.n_routed_experts), persistent=True)

    def forward(self, x_flat: torch.Tensor, tokens_per_seq: int | None = None):
        """x_flat: (N, d).  Returns (topk_idx, topk_gate, logits)."""
        cfg = self.cfg
        logits = F.linear(x_flat.float(), self.weight.float())

        if cfg.score_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=-1)

        biased = scores + self.expert_bias

        if cfg.n_groups > 1:
            biased = self._group_limit(biased)

        _, topk_idx = torch.topk(biased, cfg.top_k, dim=-1)
        topk_gate = scores.gather(1, topk_idx)  # unbiased affinities

        if cfg.norm_topk_prob and cfg.top_k > 1:
            topk_gate = topk_gate / topk_gate.sum(-1, keepdim=True).clamp_min(1e-9)
        topk_gate = topk_gate * cfg.routed_scaling_factor

        if self.training:
            self._side_losses(scores, topk_idx, logits, tokens_per_seq)
            with torch.no_grad():
                self.load_counter += torch.bincount(
                    topk_idx.flatten(), minlength=cfg.n_routed_experts
                ).float()

        return topk_idx, topk_gate.to(x_flat.dtype), logits

    # -------------------------------------------------------------- #
    def _group_limit(self, biased: torch.Tensor) -> torch.Tensor:
        """Device/node-limited routing: keep only the best ``topk_groups``."""
        cfg = self.cfg
        N = biased.shape[0]
        g = biased.view(N, cfg.n_groups, -1)
        # group score = sum of its top-2 experts
        gscore = g.topk(min(2, g.shape[-1]), dim=-1).values.sum(-1)
        keep = gscore.topk(cfg.topk_groups, dim=-1).indices
        mask = torch.zeros_like(gscore, dtype=torch.bool).scatter_(1, keep, True)
        return biased.masked_fill(~mask[..., None].expand_as(g).reshape(N, -1),
                                  float("-inf"))

    def _side_losses(self, scores, topk_idx, logits, tokens_per_seq):
        cfg = self.cfg
        N, E = scores.shape

        # -- router z-loss (ST-MoE) ---------------------------------- #
        z = torch.logsumexp(logits, dim=-1)
        z_loss = (z**2).mean() * cfg.router_z_loss_weight

        # -- sequence-wise balance loss ------------------------------ #
        if cfg.seq_aux_loss_weight <= 0:
            _accumulate(logits.new_zeros(()), z_loss)
            return
        S = tokens_per_seq or N
        B = max(N // S, 1)
        p = scores / scores.sum(-1, keepdim=True).clamp_min(1e-9)
        p = p.view(B, S, E).mean(dim=1)                     # (B, E)
        counts = torch.zeros(B, E, device=scores.device, dtype=scores.dtype)
        counts.scatter_add_(
            1,
            topk_idx.view(B, -1),
            torch.ones_like(topk_idx.view(B, -1), dtype=scores.dtype),
        )
        f = counts * (E / (cfg.top_k * S))                  # (B, E)
        bal = (f * p).sum(-1).mean() * cfg.seq_aux_loss_weight
        _accumulate(bal, z_loss)


# --------------------------------------------------------------------------- #
# Experts
# --------------------------------------------------------------------------- #
class GroupedExperts(nn.Module):
    """``n_experts`` SwiGLU FFNs stored as stacked 3-D weights.

    Gate and up share one ``(E, d, 2*inter)`` tensor, so a forward pass issues
    two segment matmuls instead of three; ``w_gate`` / ``w_up`` are views.
    ``w_down`` is ``(E, inter, d)``.
    """

    def __init__(self, n_experts: int, d_model: int, inter: int,
                 n_layers: int = 1, scale_residual_init: bool = True):
        super().__init__()
        self.n_experts = n_experts
        self.inter = inter
        std = d_model**-0.5
        self.w_gate_up = nn.Parameter(
            torch.randn(n_experts, d_model, 2 * inter) * std
        )
        down_std = inter**-0.5
        if scale_residual_init:
            down_std /= math.sqrt(2 * n_layers)
        self.w_down = nn.Parameter(torch.randn(n_experts, inter, d_model) * down_std)

    @property
    def w_gate(self) -> torch.Tensor:
        return self.w_gate_up[:, :, : self.inter]

    @property
    def w_up(self) -> torch.Tensor:
        return self.w_gate_up[:, :, self.inter :]

    def forward(self, x_sorted: torch.Tensor, offsets: torch.Tensor,
                ends: list[int] | None = None):
        gu = _grouped_linear(x_sorted, self.w_gate_up, offsets, ends)
        g, u = gu.split(self.inter, dim=-1)
        return _grouped_linear(F.silu(g) * u, self.w_down, offsets, ends)

    def forward_padded(self, x_sorted: torch.Tensor, slot: torch.Tensor,
                       capacity: int):
        gu = _padded_bmm(x_sorted, self.w_gate_up, slot, capacity, self.n_experts)
        g, u = gu.split(self.inter, dim=-1)
        return _padded_bmm(F.silu(g) * u, self.w_down, slot, capacity,
                           self.n_experts)

    def forward_aligned(self, x_sorted: torch.Tensor, dest: torch.Tensor,
                        offs: torch.Tensor, total: int):
        buf, w = _aligned_grouped_mm(x_sorted, self.w_gate_up.to(x_sorted.dtype),
                                     dest, total)
        gu = _GROUPED_MM(buf, w, offs=offs)
        g, u = gu.split(self.inter, dim=-1)
        h = F.silu(g) * u
        y = _GROUPED_MM(h, self.w_down.to(h.dtype), offs=offs)
        return y.index_select(0, dest)


class MoELayer(nn.Module):
    def __init__(self, cfg: MoEConfig, d_model: int, n_layers: int = 1,
                 scale_residual_init: bool = True):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.router = Router(cfg, d_model)
        self.experts = GroupedExperts(
            cfg.n_routed_experts, d_model, cfg.expert_inter_size,
            n_layers, scale_residual_init,
        )
        if cfg.n_shared_experts > 0:
            from .layers import SwiGLU  # local import avoids a cycle

            self.shared = SwiGLU(
                d_model, cfg.shared_inter_size * cfg.n_shared_experts,
                n_layers, scale_residual_init,
            )
        else:
            self.shared = None

        #: "auto" | "grouped" | "padded" | "loop"; all compute the same thing
        #: (padded drops tokens only when ``capacity_factor`` is set).
        self.dispatch: str = "auto"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        flat = x.reshape(-1, D)
        topk_idx, topk_gate, _ = self.router(flat, tokens_per_seq=S)

        K = self.cfg.top_k
        E = self.cfg.n_routed_experts
        N = flat.shape[0]

        # ---- sort tokens by expert id --------------------------------- #
        flat_expert = topk_idx.reshape(-1)                       # (N*K,)
        order = torch.argsort(flat_expert, stable=True)
        token_of_slot = order // K                               # (N*K,)
        counts = torch.bincount(flat_expert, minlength=E)
        offsets = torch.cumsum(counts, 0).to(torch.int32)

        x_sorted = flat.index_select(0, token_of_slot)

        mode = self.dispatch
        if mode == "auto":
            if flat.is_cuda and _GROUPED_MM is not None and _ALIGNED_OK:
                mode = "aligned"
            elif _grouped_mm_ok and flat.is_cuda:
                mode = "grouped"
            elif flat.is_cuda:
                mode = "padded"
            else:
                mode = "loop"

        if mode == "grouped":
            # fused, dropless, no host sync
            y_sorted = self.experts(x_sorted, offsets, None)
        elif mode == "aligned":
            dest, offs, total = self._aligned_slots(flat_expert, order, counts)
            y_sorted = self.experts.forward_aligned(x_sorted, dest, offs, total)
        elif mode == "padded":
            slot, capacity = self._padded_slots(flat_expert, order, counts, N, K)
            y_sorted = self.experts.forward_padded(x_sorted, slot, capacity)
        else:
            # exact, but 2*n_experts kernels per layer: CPU / reference only
            y_sorted = self.experts(x_sorted, offsets, offsets.tolist())

        # ---- weighted un-permute -------------------------------------- #
        gates = topk_gate.reshape(-1).index_select(0, order).unsqueeze(-1)
        y = torch.zeros_like(flat)
        y.index_add_(0, token_of_slot, (y_sorted * gates).to(flat.dtype))

        out = y.view(B, S, D)
        if self.shared is not None:
            out = out + self.shared(x)
        return out

    # ------------------------------------------------------------------ #
    def _aligned_slots(self, flat_expert, order, counts):
        """Destination row of each sorted slot in a 16-row-aligned layout."""
        # clamp to 16: a zero-length group makes the kernel assert
        aligned = torch.clamp(((counts + 15) // 16) * 16, min=16)
        offs_t = aligned.cumsum(0)
        starts = offs_t - aligned
        sorted_expert = flat_expert.index_select(0, order)
        rank = (torch.arange(flat_expert.numel(), device=flat_expert.device)
                - (torch.cumsum(counts, 0) - counts).index_select(0, sorted_expert))
        dest = starts.index_select(0, sorted_expert) + rank
        return dest, offs_t.to(torch.int32), int(offs_t[-1])

    # ------------------------------------------------------------------ #
    def _padded_slots(self, flat_expert, order, counts, N: int, K: int):
        """Destination row of every sorted slot inside an (E, capacity) buffer.

        ``capacity`` defaults to the largest expert's load, making this path
        dropless at the cost of one host sync per layer; ``capacity_factor``
        caps it and may then drop tokens.
        """
        E = self.cfg.n_routed_experts
        starts = torch.cumsum(counts, 0) - counts
        sorted_expert = flat_expert.index_select(0, order)
        rank = (torch.arange(N * K, device=flat_expert.device)
                - starts.index_select(0, sorted_expert))

        capacity = int(counts.max())
        if self.cfg.capacity_factor:
            capacity = min(capacity,
                           max(1, math.ceil(N * K / E * self.cfg.capacity_factor)))
        slot = sorted_expert * capacity + rank
        return torch.where(rank < capacity, slot, slot.new_full((), -1)), capacity


# --------------------------------------------------------------------------- #
# Aux-loss-free bias update (called by the trainer after optimizer.step)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def update_expert_biases(model: nn.Module, gamma: float | None = None,
                         process_group=None) -> dict[str, float]:
    """Apply the update  b_i <- b_i + gamma * f(mean_load - load_i).

    ``load_counter`` is all-reduced first: the bias is model state, so every
    data-parallel rank must keep an identical copy.  Returns MaxVio measured on
    this step's load, before the counters are cleared.
    """
    import torch.distributed as dist

    routers = [m for m in model.modules() if isinstance(m, Router)]
    if not routers:
        return {}
    if dist.is_available() and dist.is_initialized():
        flat = torch.stack([r.load_counter for r in routers])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=process_group)
        for r, c in zip(routers, flat):
            r.load_counter.copy_(c)

    stats = expert_load_stats(model)
    for r in routers:
        g = r.cfg.bias_update_rate if gamma is None else gamma
        if g > 0:
            load = r.load_counter
            mean = load.mean().clamp_min(1e-6)
            # relative imbalance, +ve = under-loaded; dividing by the mean keeps
            # |err| ~ 1 for a 2x imbalance at any expert count
            err = (mean - load) / mean

            if r.cfg.bias_controller == "pid":
                c = r.cfg
                r.pid_integral.add_(err).clamp_(-10.0, 10.0)   # anti-windup
                deriv = err - r.pid_prev_err
                r.pid_prev_err.copy_(err)
                r.expert_bias.add_(
                    g * (c.pid_kp * err + c.pid_ki * r.pid_integral
                         + c.pid_kd * deriv)
                )
            else:
                r.expert_bias.add_(g * torch.sign(err))

            if r.cfg.bias_clip:
                r.expert_bias.clamp_(-r.cfg.bias_clip, r.cfg.bias_clip)
        r.load_counter.zero_()
    return stats


@torch.no_grad()
def expert_load_stats(model: nn.Module) -> dict[str, float]:
    """MaxVio balance diagnostic, averaged and maxed over routers.

    ``(max_i load_i - mean load) / mean load``: 0 is perfect balance, 1.0 means
    the busiest expert sees twice its fair share.
    """
    routers = [m for m in model.modules() if isinstance(m, Router)]
    vios = []
    for r in routers:
        load = r.load_counter
        if float(load.sum()) == 0.0:
            continue
        vios.append(float((load.max() - load.mean()) / load.mean().clamp_min(1e-9)))
    if not vios:
        return {}
    return {"moe/maxvio_mean": sum(vios) / len(vios), "moe/maxvio_max": max(vios)}
