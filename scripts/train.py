#!/usr/bin/env python
"""Launch SCALA pretraining. Any TrainConfig field can be overridden on the
command line, e.g. `--lr 2e-4 --total_tokens 5e9 --compile false`.

    torchrun --standalone --nproc_per_node=8 scripts/train.py --config configs/train_8b_a1b.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.train.trainer import TrainConfig, Trainer  # noqa: E402


def _coerce(value: str, current):
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(value))
    if isinstance(current, float):
        return float(value)
    if current is None or isinstance(current, str):
        return None if value.lower() == "none" else value
    return value


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--config", default="configs/train_8b_a1b.yaml")
    for f in fields(TrainConfig):
        ap.add_argument(f"--{f.name}", default=None)
    args = ap.parse_args()

    cfg = TrainConfig.load(args.config) if Path(args.config).exists() \
        else TrainConfig()
    for f in fields(TrainConfig):
        v = getattr(args, f.name)
        if v is not None:
            setattr(cfg, f.name, _coerce(v, getattr(cfg, f.name)))

    Trainer(cfg).train()


if __name__ == "__main__":
    main()
