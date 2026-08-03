#!/usr/bin/env python
"""Assemble the `decoder_stream` window sweep into one table: eval CE (delta
vs the two-seed mean), len@16x (CE at 8192 vs 512-token training length),
KL/agree between the two lengths, and ctx nats for the 256-512 band only the
hierarchy can carry.

    python scripts/window_sweep.py --runs 8:runs/probe-cel-w8 64:runs/probe-cel --quality runs/sweep_ja48.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _band(ablation: list, lo: int, hi: int) -> float | None:
    """Nats attributable to context between `lo` and `hi` tokens."""
    d = {int(k): v for k, v in ablation}
    if lo not in d or hi not in d:
        return None
    return d[lo] - d[hi]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="WINDOW:PATH pairs, e.g. 64:runs/probe-cel")
    ap.add_argument("--quality", nargs="*", default=[],
                    help="compare_arms --json-out files, one per held-out slice")
    ap.add_argument("--baseline", default="probe-p2full",
                    help="arm to report the quality delta against as well")
    args = ap.parse_args()

    arms = []
    for spec in args.runs:
        w, path = spec.split(":", 1)
        arms.append((int(w), Path(path)))
    arms.sort()

    # quality: delta against the two-seed mean of each slice
    qual: dict[str, list[float]] = {}
    base_delta: list[float] = []
    labels = []
    for jf in args.quality:
        s = json.loads(Path(jf).read_text(encoding="utf-8"))
        labels.append(f"{s['data_source'].split('_')[0]}@"
                      f"{s['data_offset'] // 10**6}M")
        mean2 = sum(a["eval_ce"] for a in s["arms"][:2]) / 2
        for a in s["arms"]:
            qual.setdefault(a["run"], []).append(a["eval_ce"] - mean2)
        base = next((a for a in s["arms"] if a["run"] == args.baseline), None)
        if base:
            base_delta.append(base["eval_ce"] - mean2)

    print(f"\n{'w':>5}{'tokens':>8}" + "".join(f"{l:>10}" for l in labels)
          + f"{'len@16x':>10}{'KL':>10}{'agree':>8}"
          + f"{'ctx>16':>9}{'ctx 256-512':>13}")
    print("-" * (13 + 10 * len(labels) + 50))
    for w, path in arms:
        name = path.name
        missing = [f for f in ("length.json", "sensitivity.json", "context.json")
                   if not (path / f).exists()]
        if missing:
            print(f"{w:>5}  {name}: missing {', '.join(missing)} -- skipped")
            continue
        # w == 0 is the block layout (`decoder_lookback`), the thing the stream
        # replaces; it is not a window of zero and must not be read as one.
        row = f"{'block' if w == 0 else w:>5}"
        # stream positions -> tokens, at R=1 C=4 the ratio is 4/5
        row += f"{'4' if w == 0 else int(w * 4 / 5):>8}"
        row += "".join(f"{v:>+10.4f}" for v in qual.get(name, [])) or " " * (10 * len(labels))
        ln = json.loads((path / "length.json").read_text(encoding="utf-8"))
        base = next(r["ce"] for r in ln["rows"] if r["length"] == ln["trained_len"])
        worst = max(r["ce"] - base for r in ln["rows"])
        row += f"{worst:>+10.4f}"
        se = json.loads((path / "sensitivity.json").read_text(encoding="utf-8"))
        row += f"{se['kl']:>10.5f}{100 * se['argmax_agreement']:>7.1f}%"
        cx = json.loads((path / "context.json").read_text(encoding="utf-8"))
        ab = cx["ablation"]
        full = ab[0][1]
        row += f"{ab[-1][1] - full:>+9.4f}"
        b = _band(ab, 256, 512)
        row += f"{b:>+13.4f}" if b is not None else f"{'-':>13}"
        print(row)

    if base_delta:
        print("-" * (13 + 10 * len(labels) + 50))
        print(f"{args.baseline:>13}" + "".join(f"{v:>+10.4f}" for v in base_delta))
    print("""
w              `decoder_stream`, in stream positions; `tokens` is what that is
               worth in level-1 token history at R=1, C=4.
eval CE        delta against the two-seed mean of that slice.  A delta smaller
               than the two-seed spread is UNRESOLVED, never equal.
len@16x        worst CE degradation against the 512-token training length.
KL / agree     between the 512- and 8192-token predictions on identical tokens.
               **A flat len@16x with KL ~ 0 is indifference, not robustness.**
ctx 256-512    nats the model draws from context only the hierarchy can carry --
               the decoder window cannot reach there at any width in this sweep.""")


if __name__ == "__main__":
    main()
