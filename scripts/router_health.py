#!/usr/bin/env python
"""Per-stack router diagnosis from an exported checkpoint: breaks MaxVio down
by hierarchy stack instead of one aggregate. Reads `expert_bias` (the
auxiliary-loss-free balancing term); experts pinned at +/-`--clip` indicate
the controller has exhausted its correction authority for that router.

    python scripts/router_health.py export/scala-8b-a1b-v3
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch

STACK = re.compile(r"levels\.(\d+)\.(encoder|decoder)")


def stack_name(key: str) -> str:
    m = STACK.search(key)
    if not m:
        return "other"
    return f"L{int(m.group(1)) + 1}.{m.group(2)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="export directory or a state.pt")
    ap.add_argument("--clip", type=float, default=2.0,
                    help="bias_clip from the model config")
    args = ap.parse_args()

    p = Path(args.src)
    if (p / "model.safetensors").exists():
        from safetensors.torch import load_file

        sd = load_file(str(p / "model.safetensors"))
    else:
        f = p / "state.pt" if (p / "state.pt").exists() else p
        # mmap avoids loading the full checkpoint into RAM
        blob = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
        sd = blob.get("model", blob)

    per = defaultdict(list)
    for k, v in sd.items():
        if k.endswith("expert_bias"):
            per[stack_name(k)].append((k, v.float()))

    if not per:
        print("no expert_bias buffers found -- was this exported from a run "
              "with an auxiliary-loss-free controller?")
        return

    print(f"{'stack':<14}{'layers':>7}{'experts':>9}{'bias sd':>10}"
          f"{'at -clip':>10}{'at +clip':>10}{'range':>18}")
    print("-" * 78)
    for stack in sorted(per):
        entries = per[stack]
        E = entries[0][1].numel()
        lo = hi = 0
        sds, mins, maxs = [], [], []
        for _, b in entries:
            sds.append(float(b.std()))
            mins.append(float(b.min()))
            maxs.append(float(b.max()))
            lo += int((b <= -args.clip + 1e-4).sum())
            hi += int((b >= args.clip - 1e-4).sum())
        n = len(entries) * E
        print(f"{stack:<14}{len(entries):>7}{E:>9}"
              f"{sum(sds)/len(sds):>10.3f}"
              f"{100*lo/n:>9.1f}%{100*hi/n:>9.1f}%"
              f"{min(mins):>9.2f}..{max(maxs):<8.2f}")

    print("\nHow to read this: experts pinned at the clamp are experts the "
          "controller\nwanted to move further and could not. A few percent is "
          "ordinary. Tens of\npercent on one stack means that stack's router "
          "has lost the argument, and\nthe capacity sitting behind it is not "
          "being used.")


if __name__ == "__main__":
    main()
