#!/usr/bin/env python
"""Token-retrieval probe: places a span, repeats it `d` tokens later, scores
only the repeat. copy gain(d) = CE(unrelated) - CE(repeat). Aligned vs offset
(mod C_<=L) sweep isolates the fixed-grid penalty beyond the bounded reach.

    python scripts/copy_diag.py --ckpt runs/probe-cel/final --data-source ja_wikipedia --data-offset 48000000
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol_diag import load_model  # noqa: E402


def bounded_reach_tokens(cfg) -> int:
    """Max distance a token can see WITHOUT the global stream, in tokens.

    The maximum over every span-bounded stack: a level-l decoder windows over
    level-(l-1) units, an encoder window over level-l units, so reach grows
    with depth.  Only distances past this number test the hierarchy.
    """
    reach = 0.0
    c_below = 1                       # tokens per level-(l-1) unit
    for i, lv in enumerate(cfg.levels, start=1):
        top = i == len(cfg.levels)
        if lv.decoder_stream:
            units = (lv.decoder_stream * lv.chunk_size
                     / (lv.converter_width + lv.chunk_size))
        else:
            units = lv.chunk_size * (1 + lv.decoder_lookback)
        reach = max(reach, units * c_below)
        c_here = c_below * lv.chunk_size
        if not top:
            if lv.encoder_window:
                reach = max(reach, lv.encoder_window * c_here)
            elif lv.encoder_block_local:
                reach = max(reach, cfg.levels[i].chunk_size * c_here)
        c_below = c_here
    return int(reach)


def load_pool(cfg, args) -> np.ndarray:
    """One contiguous block of held-out tokens to cut spans and filler from."""
    root = Path(args.data_root)
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    dtype = {"uint16": np.uint16, "uint32": np.uint32}[man.get("dtype", "uint32")]
    want = args.data_source
    srcs = [s for s in man["sources"] if not want or s["name"] == want]
    if not srcs:
        raise SystemExit(f"--data-source {want!r} not in {root/'manifest.json'}")
    shard = root / srcs[0]["shards"][0]["path"]
    arr = np.memmap(shard, dtype=dtype, mode="r")
    need = args.data_offset + args.pool
    if arr.size < need:
        raise SystemExit(f"{shard.name} holds {arr.size} tokens, need {need}")
    pool = np.asarray(arr[args.data_offset : args.data_offset + args.pool]).astype(np.int64)
    if pool.max() >= cfg.vocab_size:
        raise SystemExit(f"{shard.name}: id {pool.max()} >= vocab {cfg.vocab_size}")
    print(f"tokens: {shard.name} @{args.data_offset:,} ({args.pool:,} held out)")
    return pool


def build_batch(pool: np.ndarray, rng: np.random.Generator, args,
                d: int, repeat: bool) -> tuple[torch.Tensor, int]:
    """`n` sequences of length `T`, each with a span at `p1` and at `p1 + d`.

    Returns the batch and the start index of the second occurrence, which is
    the only region scored.  ``repeat=False`` puts an unrelated span there, so
    the two batches differ *only* in whether retrieval is possible.
    """
    T, n, span = args.seq_len, args.batch, args.span
    p2 = T - span                      # second occurrence at the very end
    p1 = p2 - d
    if p1 < 0:
        raise SystemExit(f"d={d} with span={span} does not fit in seq_len={T}")

    out = np.empty((n, T), dtype=np.int64)
    for i in range(n):
        base = rng.integers(0, len(pool) - T - span)
        out[i] = pool[base : base + T]                        # natural filler
        s = pool[base + T : base + T + span]                  # a span from elsewhere
        out[i, p1 : p1 + span] = s
        if repeat:
            out[i, p2 : p2 + span] = s
        else:
            alt = rng.integers(0, len(pool) - span)
            out[i, p2 : p2 + span] = pool[alt : alt + span]
    return torch.from_numpy(out), p2


@torch.no_grad()
def score(model, batch: torch.Tensor, p2: int, span: int, device) -> float:
    """Mean CE over the second occurrence only."""
    x = batch.to(device)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, p2 : p2 + span] = True
    out = model(x, labels=x, return_logits=False, loss_mask=mask)
    return float(out.loss_token)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--depth", type=int, default=None,
                    help="re-express a tied (SCALA) checkpoint at this many "
                         "MID applications before scoring")
    ap.add_argument("--data-root", default="data/tokens_ja")
    ap.add_argument("--data-source", default="ja_wikipedia")
    ap.add_argument("--data-offset", type=int, default=48_000_000)
    ap.add_argument("--pool", type=int, default=4_000_000,
                    help="held-out tokens to draw spans and filler from")
    ap.add_argument("--seq-len", type=int, default=None,
                    help="defaults to the checkpoint's max_seq_len")
    ap.add_argument("--span", type=int, default=16,
                    help="tokens in the repeated span")
    ap.add_argument("--distances", default=None,
                    help="comma-separated; default is a log sweep that fits")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=4,
                    help="independent batches averaged per cell")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, args.config, args.device,
                            getattr(torch, args.dtype), depth=args.depth)
    cp = cfg.chunk_product
    args.seq_len = args.seq_len or cfg.max_seq_len
    win_tok = bounded_reach_tokens(cfg)      # the control boundary

    if args.distances:
        dists = [int(x) for x in args.distances.split(",")]
    else:
        dists, d = [], cp
        while d <= args.seq_len - args.span - cp:
            dists.append(d)
            d *= 2
    if not any(d > win_tok for d in dists):
        raise SystemExit(
            f"every distance is inside the {win_tok}-token bounded reach of "
            f"this model, so none of them test the hierarchy.  Raise --seq-len "
            f"(the checkpoint's max_seq_len is a training shape, not a limit) "
            f"or pass --distances explicitly.")
    pool = load_pool(cfg, args)

    print(f"\nC_<=L = {cp}   bounded (non-global) reach ~= {win_tok} tokens   "
          f"span = {args.span}   seq_len = {args.seq_len}")
    print(f"{'d':>7}{'d%C':>5}{'grid':>9}{'CE repeat':>11}{'CE ctrl':>10}"
          f"{'copy gain':>11}")
    print("-" * 53)

    # A silently-inert offset would also produce a null result, so check once
    # that the offset construction moves the span off the grid and changes
    # the tokens.
    _rng = np.random.default_rng(args.seed)
    _d = next(d for d in dists if d > win_tok)
    _a, _p2 = build_batch(pool, np.random.default_rng(args.seed + _d), args, _d, True)
    _o, _ = build_batch(pool, np.random.default_rng(args.seed + _d), args,
                        _d + cp // 2, True)
    # Alignment is `p1 == p2 (mod C)` -- the two occurrences cut the same way
    # relative to each other.  NOT "p1 on a chunk boundary": p2 = seq_len - span
    # need not sit on one.
    assert (_p2 - _d) % cp == _p2 % cp, "the aligned arm is not aligned"
    assert (_p2 - _d - cp // 2) % cp != _p2 % cp, "the offset arm is not offset"
    n_diff = int((_a != _o).sum())
    assert n_diff > 0, "the offset arm produced identical token sequences"
    print(f"offset check: {n_diff} token positions differ between the grids")

    rows = []
    for d in dists:
        for grid, dd in (("aligned", d), ("offset", d + cp // 2)):
            if dd > args.seq_len - args.span:
                continue
            ce = {}
            for repeat in (True, False):
                # the SAME seed for both arms: identical filler and identical
                # span placement, so the only difference is retrievability
                rng = np.random.default_rng(args.seed + d)
                tot = 0.0
                for r in range(args.repeats):
                    b, p2 = build_batch(pool, rng, args, dd, repeat)
                    tot += score(model, b, p2, args.span, args.device)
                ce[repeat] = tot / args.repeats
            gain = ce[False] - ce[True]
            # `base_d` pairs the two grids: judging each row's own distance
            # against the reach would admit the offset row (d + C/2) while
            # excluding its aligned partner (d) at the boundary.
            rows.append({"d": dd, "base_d": d, "grid": grid,
                         "ce_repeat": ce[True], "ce_control": ce[False],
                         "gain": gain, "beyond_window": d > win_tok})
            print(f"{dd:>7}{dd % cp:>5}{grid:>9}{ce[True]:>11.4f}"
                  f"{ce[False]:>10.4f}{gain:>+11.4f}")

    # only complete aligned/offset pairs, and only past the bounded reach
    by_base: dict[int, dict[str, float]] = {}
    for r in rows:
        if r["beyond_window"]:
            by_base.setdefault(r["base_d"], {})[r["grid"]] = r["gain"]
    pairs = [(v["aligned"], v["offset"]) for v in by_base.values()
             if "aligned" in v and "offset" in v]
    if pairs:
        a = sum(p[0] for p in pairs) / len(pairs)
        o = sum(p[1] for p in pairs) / len(pairs)
        worst = max(abs(p[0] - p[1]) for p in pairs)
        print(f"\nbeyond the {win_tok}-token bounded reach "
              f"({len(pairs)} matched pairs), mean copy gain:")
        print(f"  chunk-aligned repeat : {a:+.4f} nats")
        print(f"  off-grid repeat      : {o:+.4f} nats")
        print(f"  --> fixed-grid penalty: {a - o:+.4f} nats "
              f"(worst single pair {worst:.4f})")
        print(f"      ^ this is what dynamic chunking would buy, before "
              f"building it")
    print("\nCopy gain is CE(unrelated span) - CE(repeated span) on the same "
          "positions with\nidentical filler, so it is retrieval and nothing "
          "else.  Inside the decoder window\nthe local stream can copy raw "
          "tokens, so only rows beyond it test the hierarchy.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"ckpt": args.ckpt, "chunk_product": cp, "window_tokens": win_tok,
             "span": args.span, "seq_len": args.seq_len, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
