#!/usr/bin/env python
"""Ultra-long-context probes (131K..1M+) via the tiled exact scorer. --mode
length: length_diag semantics (fixed final --win tokens, varying history).
--mode copy: copy_diag semantics at ultra distances (gains are
distribution-matched, not per-sequence-paired).

    python scripts/longctx_probe.py --ckpt runs/probe-scala-k2/final --depth 2 --mode length --lengths 131072,262144,524288 --win 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scala.infer.scoring import TiledScorer  # noqa: E402
from scala.model.layers import RotaryEmbedding  # noqa: E402
from copy_diag import bounded_reach_tokens, build_batch, load_pool  # noqa: E402
from protocol_diag import load_model  # noqa: E402


def _pool_tokens(cfg, args, need: int) -> np.ndarray:
    class _A:  # load_pool reads these fields
        data_root = args.data_root
        data_source = args.data_source
        data_offset = args.data_offset
        pool = need
    return load_pool(cfg, _A)


def mode_length(model, cfg, args, scorer) -> dict:
    lengths = sorted(int(x) for x in args.lengths.split(","))
    cp = cfg.chunk_product
    assert all(t % cp == 0 for t in lengths), f"lengths must be multiples of {cp}"
    pool = _pool_tokens(cfg, args, max(lengths) + args.batches * args.stride)

    rows = []
    for T in lengths:
        ces, t0 = [], time.time()
        for b in range(args.batches):
            end = max(lengths) + b * args.stride
            seq = torch.from_numpy(pool[end - T : end]).unsqueeze(0)
            ce, _ = scorer.score_span(seq, args.win)
            ces.append(ce)
        ce = sum(ces) / len(ces)
        rows.append({"context": T, "ce": ce, "sec_per_seq":
                     (time.time() - t0) / args.batches})
        base = rows[0]["ce"]
        print(f"  {T:>9,}  CE {ce:.4f}  vs shortest {ce - base:+.4f}  "
              f"({rows[-1]['sec_per_seq']:.1f} s/seq)")
    return {"mode": "length", "win": args.win, "batches": args.batches,
            "rows": rows}


def mode_copy(model, cfg, args, scorer) -> dict:
    cp = cfg.chunk_product
    T = args.seq_len
    assert T % cp == 0
    reach = bounded_reach_tokens(cfg)
    dists = [int(x) for x in args.distances.split(",")]
    pool = _pool_tokens(cfg, args, args.pool)

    class _B:  # build_batch reads these fields
        seq_len = T
        batch = args.batch
        span = args.span

    print(f"C_<=L={cp}  bounded reach ~{reach:,} tok  span={args.span}  "
          f"seq_len={T:,}")
    print(f"{'d':>10}{'d%C':>6}{'grid':>9}{'CE rep':>9}{'CE ctrl':>9}"
          f"{'gain':>9}")
    rows = []
    for d in dists:
        for grid, dd in (("aligned", d), ("offset", d + cp // 2)):
            if dd > T - args.span:
                continue
            ce = {}
            for repeat in (True, False):
                # copy_diag discipline: same seed for both arms, rng advances
                # across repeats within an arm
                rng = np.random.default_rng(args.seed + d)
                tot = 0.0
                for _ in range(args.repeats):
                    b, p2 = build_batch(pool, rng, _B, dd, repeat)
                    assert p2 == T - args.span
                    mean_ce, _ = scorer.score_span(b, args.span)
                    tot += mean_ce
                ce[repeat] = tot / args.repeats
            gain = ce[False] - ce[True]
            rows.append({"d": dd, "base_d": d, "grid": grid,
                         "ce_repeat": ce[True], "ce_control": ce[False],
                         "gain": gain, "beyond_window": d > reach})
            print(f"{dd:>10,}{dd % cp:>6}{grid:>9}{ce[True]:>9.4f}"
                  f"{ce[False]:>9.4f}{gain:>+9.4f}")

    # document the pairing structure instead of assuming it
    rng_a = np.random.default_rng(args.seed + dists[0])
    rng_b = np.random.default_rng(args.seed + dists[0])
    a, p2 = build_batch(pool, rng_a, _B, dists[0], True)
    c, _ = build_batch(pool, rng_b, _B, dists[0], False)
    outside = int((a != c)[:, : p2].sum())
    print(f"pairing note: {outside:,} positions differ between arms OUTSIDE "
          f"the scored span (expected nonzero for i>=1; gains are "
          f"distribution-matched, not per-sequence-paired)")

    by_base: dict[int, dict[str, float]] = {}
    for r in rows:
        if r["beyond_window"]:
            by_base.setdefault(r["base_d"], {})[r["grid"]] = r["gain"]
    pairs = [(v["aligned"], v["offset"]) for v in by_base.values()
             if "aligned" in v and "offset" in v]
    summary = {}
    if pairs:
        al = sum(p[0] for p in pairs) / len(pairs)
        of = sum(p[1] for p in pairs) / len(pairs)
        summary = {"mean_gain_aligned": al, "mean_gain_offset": of,
                   "grid_penalty": al - of,
                   "worst_pair": max(abs(p[0] - p[1]) for p in pairs)}
        print(f"\nbeyond reach ({len(pairs)} pairs): aligned {al:+.4f}  "
              f"offset {of:+.4f}  grid penalty {al - of:+.4f}")
    return {"mode": "copy", "span": args.span, "seq_len": T,
            "chunk_product": cp, "reach": reach, "rows": rows,
            "pairing_positions_differ_outside_span": outside, **summary}


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--mode", choices=["length", "copy"], required=True)
    ap.add_argument("--data-root", default="data/tokens_ja")
    ap.add_argument("--data-source", default="ja_wikipedia")
    ap.add_argument("--data-offset", type=int, default=48_000_000)
    ap.add_argument("--dtype", default="float32",
                    help="published copy/length rows are float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--tile-tokens", type=int, default=65_536)
    ap.add_argument("--enc-block-units", type=int, default=1_024)
    # length mode
    ap.add_argument("--lengths", default="131072,262144,524288,1048576")
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--stride", type=int, default=65_536)
    # copy mode
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--span", type=int, default=16)
    ap.add_argument("--distances", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--pool", type=int, default=8_000_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, cfg = load_model(args.ckpt, args.config, args.device, dtype,
                            depth=args.depth)
    scorer = TiledScorer(model, args.device, dtype,
                         tile_tokens=args.tile_tokens,
                         enc_block_units=args.enc_block_units)

    out = (mode_length if args.mode == "length" else mode_copy)(
        model, cfg, args, scorer)
    out["rope_max_table_len"] = RotaryEmbedding.MAX_TABLE_LEN
    out["rope_convention"] = ("rows whose stream positions exceed "
                              "max_table_len use fp64-phase rotary rows")
    out["depth"] = args.depth
    out["dtype"] = args.dtype
    if args.device == "cuda":
        out["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        print(f"peak GPU memory: {out['peak_gib']:.2f} GiB")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
