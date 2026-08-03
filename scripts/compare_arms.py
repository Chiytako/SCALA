#!/usr/bin/env python
"""Several probe arms on one table, next to the two-seed spread. Per arm:
eval CE (full-sequence forward), train CE (spread over several tail windows),
ctx (nats lost when context past one meta-context is noised), params.

    python scripts/compare_arms.py runs/probe runs/probe-seed2 runs/probe-skip --data-source ja_wikipedia --data-offset 48000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from context_diag import ablate_far_context  # noqa: E402
from protocol_diag import load_model, load_tokens  # noqa: E402

#: tail windows to average `loss_token` over; report the spread, not one value.
TAILS = (3, 5, 8, 10, 15)


def tail_ce(run: Path, tail: int) -> float:
    rows = [json.loads(l) for l in (run / "log.jsonl").open(encoding="utf-8")]
    # the trainer re-logs a step on resume; keep the last entry per step
    by_step = {r["step"]: r for r in rows}
    steps = sorted(by_step)[-tail:]
    return sum(by_step[s]["loss_token"] for s in steps) / max(len(steps), 1)


@torch.no_grad()
def eval_ce(model, cfg, args) -> tuple[float, int]:
    """Mean CE over `--batches` full sequences.  Returns (nats, n_tokens).

    ``return_logits=False`` routes through the model's chunked cross-entropy;
    materialising logits costs ``batch * seq_len * vocab * 4`` bytes.
    """
    tot, n = 0.0, 0
    for b in range(args.batches):
        args.offset = args.data_offset + b * args.batch * args.seq_len
        tokens = load_tokens(cfg, args, args.device)
        out = model(tokens, labels=tokens, return_logits=False)
        tot += float(out.loss_token) * tokens.numel()
        n += tokens.numel()
    return tot / n, n


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--depth", type=int, default=None,
                    help="re-express tied (SCALA) arms at this many MID "
                         "applications; untied arms in the same table are "
                         "scored as-is, with a note")
    ap.add_argument("--data-root", default="data/tokens_ja")
    ap.add_argument("--data-source", default=None,
                    help="manifest source to score on; default is the first, "
                         "which for data/tokens_ja is English")
    ap.add_argument("--data-offset", type=int, default=1024,
                    help="token offset into the shard.  The default is inside "
                         "the prefix every training run reads, i.e. training "
                         "data; pass a value past total_tokens*weight instead")
    ap.add_argument("--tail", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--score-window", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rows = []
    n_scored = 0
    for r in args.runs:
        run = Path(r)
        depth = args.depth
        if depth is not None:
            from scala.model.config import ScalaConfig
            if not ScalaConfig.load(run / "model_config.json").tie_mid_levels:
                print(f"{run.name}: untied -- scoring at its own depth")
                depth = None
        model, cfg = load_model(str(run / "final"), str(run / "model_config.json"),
                                args.device, getattr(torch, args.dtype),
                                depth=depth)
        ce, n_scored = eval_ce(model, cfg, args)
        abl = ablate_far_context(model, cfg, args)
        rows.append({
            "run": run.name,
            "eval_ce": ce,
            "tails": {k: tail_ce(run, k) for k in TAILS},
            "ce": tail_ce(run, args.tail),
            "ctx_nats": abl[-1][1] - abl[0][1],
            "ctx_from": abl[-1][0],
            "params": model.num_parameters(),
        })
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    src = args.data_source or "<first source in manifest>"
    print(f"\nscored on {src} @ token {args.data_offset:,}  "
          f"({n_scored:,} tokens/arm, {args.seq_len}-token sequences)")
    print(f"{'arm':<18}{'eval CE':>10}{'d eval':>9}{'train CE':>10}"
          f"{'ctx nats':>10}{'params':>12}")
    print("-" * 71)
    base = rows[0]
    for r in rows:
        print(f"{r['run']:<18}{r['eval_ce']:>10.4f}"
              f"{r['eval_ce'] - base['eval_ce']:>+9.4f}{r['ce']:>10.4f}"
              f"{r['ctx_nats']:>10.4f}{r['params']/1e6:>11.2f}M")

    print(f"\ntrain CE by tail window -- one column is not a result, the spread is")
    print(f"{'arm':<18}" + "".join(f"{'k=' + str(k):>10}" for k in TAILS))
    print("-" * (18 + 10 * len(TAILS)))
    for r in rows:
        print(f"{r['run']:<18}" + "".join(f"{r['tails'][k]:>10.4f}" for k in TAILS))

    print(f"\n'ctx nats' = CE lost when context is cut to {base['ctx_from']} "
          f"tokens; higher means the model leans on the hierarchy more.")
    if len(rows) >= 2:
        spread = {k: abs(rows[1]['tails'][k] - rows[0]['tails'][k]) for k in TAILS}
        print(f"\ntwo-seed spread (first two arms), train CE: "
              + "  ".join(f"k={k}:{v:.4f}" for k, v in spread.items()))
        print(f"two-seed spread, eval CE: "
              f"{abs(rows[1]['eval_ce'] - rows[0]['eval_ce']):.4f}")
        print("This is ONE realisation of |X1 - X2|, not a dispersion estimate: "
              "it has no confidence attached and cannot license an equivalence\n"
              "claim.  Treat a delta smaller than it as unresolved, never as "
              "equal.  n>=8 runs of the unchanged config would give a real sigma.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"data_source": args.data_source, "data_offset": args.data_offset,
             "tokens_per_arm": n_scored, "arms": rows}, indent=2))


if __name__ == "__main__":
    main()
