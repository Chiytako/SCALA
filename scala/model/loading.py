"""Checkpoint loading that tolerates architecture drift.

Rule: parameters must match exactly, buffers may be re-derived.  A missing
weight raises; a missing buffer whose module knows how to initialise it is
filled in and reported.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

__all__ = ["load_state_dict_compat", "REDERIVABLE_BUFFERS"]

#: buffer name suffixes that carry no information a fresh run cannot recreate
REDERIVABLE_BUFFERS: tuple[str, ...] = (
    "pid_integral",     # PID controller accumulator, starts at 0
    "pid_prev_err",     # PID previous error, starts at 0
    "load_counter",     # per-step expert load, non-persistent anyway
    "expert_bias",      # routing bias; zero is the untrained default
    "inv_freq",         # RoPE tables, recomputed from theta
    "cos_cached",
    "sin_cached",
)


def load_state_dict_compat(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    extra_optional: Iterable[str] = (),
    verbose: bool = True,
) -> dict[str, list[str]]:
    """Load ``state_dict`` into ``model``, tolerating re-derivable buffers.

    Raises if any *parameter* is missing or unexpected.  Returns the lists of
    keys that were filled in or ignored, so a caller can log them.
    """
    optional = REDERIVABLE_BUFFERS + tuple(extra_optional)
    param_names = {n for n, _ in model.named_parameters()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    def _is_optional(k: str) -> bool:
        return any(k.endswith(o) or f".{o}" in k for o in optional)

    hard_missing = [k for k in missing
                    if k in param_names or not _is_optional(k)]
    hard_unexpected = [k for k in unexpected if not _is_optional(k)]

    if hard_missing or hard_unexpected:
        raise RuntimeError(
            "checkpoint does not match the model:\n"
            f"  missing    : {hard_missing[:10]}"
            f"{f' (+{len(hard_missing)-10} more)' if len(hard_missing) > 10 else ''}\n"
            f"  unexpected : {hard_unexpected[:10]}"
            f"{f' (+{len(hard_unexpected)-10} more)' if len(hard_unexpected) > 10 else ''}"
        )

    soft_missing = [k for k in missing if k not in hard_missing]
    if soft_missing:
        # they are already at their constructor defaults; just say so
        if verbose:
            kinds = sorted({k.rsplit(".", 1)[-1] for k in soft_missing})
            print(f"[load] checkpoint predates {len(soft_missing)} buffer(s) "
                  f"({', '.join(kinds)}); using their initial values")
    if unexpected and verbose:
        kinds = sorted({k.rsplit(".", 1)[-1] for k in unexpected})
        print(f"[load] ignoring {len(unexpected)} unused buffer(s) "
              f"({', '.join(kinds)})")

    return {"filled": soft_missing, "ignored": list(unexpected)}
