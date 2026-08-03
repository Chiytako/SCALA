#!/usr/bin/env python
"""Turn several `compare_arms.py --json-out` files into one delta table.
Each arm is a delta against the two-seed mean of its slice; the two-seed
spread is printed underneath (a delta smaller than it is unresolved, never
equal).  The first two runs passed to `compare_arms.py` must be the seed pair.

    python scripts/arm_table.py runs/cel_ja48.json runs/cel_ja60.json --labels ja@48M ja@60M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--metric", default="eval_ce")
    args = ap.parse_args()

    slices = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.jsons]
    labels = args.labels or [f"{s['data_source']}@{s['data_offset']//10**6}M"
                             for s in slices]
    names = [a["run"] for a in slices[0]["arms"]]
    for s in slices:
        if [a["run"] for a in s["arms"]] != names:
            raise SystemExit("the json files do not hold the same arms in the "
                             "same order; the delta table would be nonsense")

    # baseline of each slice = the mean of the first two arms (the seed pair)
    base = [sum(a[args.metric] for a in s["arms"][:2]) / 2 for s in slices]
    spread = [abs(s["arms"][0][args.metric] - s["arms"][1][args.metric])
              for s in slices]

    w = max(len(n) for n in names) + 2
    print(f"\n{args.metric}, delta against the two-seed mean of each slice "
          f"({slices[0]['tokens_per_arm']:,} tokens/arm)")
    print(f"{'arm':<{w}}" + "".join(f"{l:>12}" for l in labels)
          + f"{'ctx nats':>26}")
    print("-" * (w + 12 * len(labels) + 26))
    for i, n in enumerate(names):
        d = "".join(f"{s['arms'][i][args.metric] - base[k]:>+12.4f}"
                    for k, s in enumerate(slices))
        ctx = " / ".join(f"{s['arms'][i]['ctx_nats']:.3f}" for s in slices)
        print(f"{n:<{w}}{d}{ctx:>26}")
    print("-" * (w + 12 * len(labels) + 26))
    print(f"{'two-seed spread':<{w}}" + "".join(f"{v:>12.4f}" for v in spread))
    print("\nOne realisation of |X1 - X2|, with no confidence attached: a delta "
          "smaller than\nit is UNRESOLVED, never equal.  An effect that changes "
          "sign between slices is\nnot an effect (docs/findings.md 4m).")


if __name__ == "__main__":
    main()
