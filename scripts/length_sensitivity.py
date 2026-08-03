#!/usr/bin/env python
"""Separate robust from indifferent behind a flat length curve: score identical
tokens at two context lengths and compare the logits.  KL ~ 0 with near-100%
argmax agreement means the extra context never reached the prediction
(indifference); a distribution that moves while CE holds is robustness.

    python scripts/length_sensitivity.py --ckpt runs/probe-cel/final --data-source ja_wikipedia --data-offset 48000000
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol_diag import load_model, load_tokens  # noqa: E402


@torch.no_grad()
def logits_at_length(model, cfg, args, T: int, quiet: bool) -> torch.Tensor:
    """Logits over the last ``--score-window`` positions, given ``T`` of context.

    One sequence at a time; each sequence's last token is pinned, so the
    scored text is byte-identical at every ``T``.
    """
    sub = argparse.Namespace(**vars(args))
    sub.seq_len, sub.batch = T, 1
    out = []
    for s in range(args.sequences):
        sub.offset = args.data_offset + (s + 1) * args.max_len - T
        sink = io.StringIO() if (quiet or s) else sys.stdout
        with contextlib.redirect_stdout(sink):
            tokens = load_tokens(cfg, sub, args.device)
        out.append(model(tokens).logits[0, -args.score_window :].float())
    return torch.stack(out)                     # (S, win, V)


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
    ap.add_argument("--data-offset", type=int, default=48_000_000)
    ap.add_argument("--short", type=int, default=None,
                    help="reference context; defaults to the training length")
    ap.add_argument("--long", type=int, default=8192)
    ap.add_argument("--sequences", type=int, default=4)
    ap.add_argument("--score-window", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, args.config, args.device,
                            getattr(torch, args.dtype), depth=args.depth)
    short = args.short or cfg.max_seq_len
    args.max_len = max(short, args.long)
    for T in (short, args.long):
        if T % cfg.chunk_product:
            raise SystemExit(f"length {T} is not a multiple of "
                             f"C_<=L={cfg.chunk_product}")

    a = logits_at_length(model, cfg, args, short, quiet=False)
    b = logits_at_length(model, cfg, args, args.long, quiet=True)

    d = (a - b).abs()
    p, q = a.softmax(-1), b.softmax(-1)
    kl = (p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum(-1).mean()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean()

    res = {"ckpt": args.ckpt, "short": short, "long": args.long,
           "sequences": args.sequences, "score_window": args.score_window,
           "max_abs_dlogit": d.max().item(), "mean_abs_dlogit": d.mean().item(),
           "kl": kl.item(), "argmax_agreement": agree.item()}

    print(f"\n{args.ckpt}: context {short} vs {args.long}, "
          f"{args.sequences * args.score_window:,} identical scored tokens")
    print(f"  max |delta logit|   {res['max_abs_dlogit']:10.4f}")
    print(f"  mean |delta logit|  {res['mean_abs_dlogit']:10.4f}")
    print(f"  KL(short || long)   {res['kl']:10.5f}")
    print(f"  argmax agreement    {100 * res['argmax_agreement']:9.1f}%")
    print("\nKL near zero with a flat CE curve means the extra context never "
          "reached the\nprediction -- indifference, not robustness.  A moving "
          "distribution with a flat\nCE curve is the good case.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
