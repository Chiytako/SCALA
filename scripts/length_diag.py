#!/usr/bin/env python
"""CE against context length, on identical scored tokens: only the final
`--score-window` positions are scored, so only the history changes with `T`.
Reports CE(T) - CE(T_train), signed: positions past the training range should
not raise CE if the model generalizes there.

    python scripts/length_diag.py --ckpt runs/probe-cel/final --data-source ja_wikipedia --data-offset 48000000 --lengths 512,1024,2048,4096
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol_diag import load_model, load_tokens  # noqa: E402


@torch.no_grad()
def ce_at_length(model, cfg, args, T: int, quiet: bool) -> tuple[float, int]:
    """CE over the last ``--score-window`` positions, given ``T`` of context.

    One sequence at a time: ``load_tokens`` returns *contiguous* tokens
    reshaped to ``(batch, seq_len)``, so a batched read would move every
    sequence's end-offset with ``T`` -- the scored text must not move.
    """
    import contextlib
    import io

    win = args.score_window
    sub = argparse.Namespace(**vars(args))
    sub.seq_len, sub.batch = T, 1
    tot, n = 0.0, 0
    for b in range(args.batches):
        seqs = []
        for i in range(args.batch):
            s = b * args.batch + i
            # this sequence's last token is fixed; only how far back it starts
            # depends on T
            sub.offset = args.data_offset + (s + 1) * args.max_len - T
            sink = io.StringIO() if (quiet or s) else sys.stdout
            with contextlib.redirect_stdout(sink):
                seqs.append(load_tokens(cfg, sub, args.device))
        tokens = torch.cat(seqs, dim=0)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask[:, -win:] = True
        out = model(tokens, labels=tokens, return_logits=False, loss_mask=mask)
        tot += float(out.loss_token) * tokens.shape[0]
        n += tokens.shape[0]
    return tot / n, n


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--depth", type=int, default=None,
                    help="re-express a tied (SCALA) checkpoint at this many "
                         "MID applications before scoring")
    ap.add_argument("--data-root", default="data/tokens_ja")
    ap.add_argument("--data-source", default=None)
    ap.add_argument("--data-offset", type=int, default=48_000_000,
                    help="held-out offset; the default is past what any probe "
                         "run consumed of a ja shard")
    ap.add_argument("--lengths", default="512,1024,2048,4096",
                    help="context lengths to score at, comma-separated")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--score-window", type=int, default=256,
                    help="positions at the end of the sequence that are scored; "
                         "the same text at every length")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    args.max_len = max(lengths)
    model, cfg = load_model(args.ckpt, args.config, args.device,
                            getattr(torch, args.dtype), depth=args.depth)
    trained = cfg.max_seq_len
    for T in lengths:
        if T % cfg.chunk_product:
            raise SystemExit(f"length {T} is not a multiple of "
                             f"C_<=L={cfg.chunk_product}")

    rows = []
    for k, T in enumerate(lengths):
        ce, n = ce_at_length(model, cfg, args, T, quiet=bool(k))
        rows.append({"length": T, "ce": ce, "sequences": n})

    base = next(r for r in rows if r["length"] == trained)["ce"] \
        if any(r["length"] == trained for r in rows) else rows[0]["ce"]

    print(f"\ntrained at {trained} tokens; scoring the same "
          f"{args.score_window} tokens with varying history")
    print(f"{'context':>9}{'x train':>9}{'CE':>10}{'vs train len':>14}")
    print("-" * 42)
    for r in rows:
        print(f"{r['length']:>9}{r['length']/trained:>8.1f}x{r['ce']:>10.4f}"
              f"{r['ce'] - base:>+14.4f}")
    worst = max(r["ce"] - base for r in rows)
    print(f"\nlargest degradation against the training length: {worst:+.4f} nats")
    print("A model whose positions all stay inside their trained range should "
          "improve or hold;\nrising CE past the training length is "
          "extrapolation failure, not a harder slice --\nthe scored tokens are "
          "identical in every row.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"ckpt": args.ckpt, "trained_len": trained,
             "score_window": args.score_window,
             "data_source": args.data_source, "data_offset": args.data_offset,
             "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
