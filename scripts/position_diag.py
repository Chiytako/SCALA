#!/usr/bin/env python
"""CE profile over (chunk slot, group slot): readout of the one-vector
bottleneck's cost. Slot j=0 of a level-1 chunk is conditioned only through
`U^(1)(X_hat^(1)_{m-1})`; slot j>0 also reads j raw token embeddings, same
split one level up. Flat = funnel wide enough; sawtooth = which level to widen.

    python scripts/position_diag.py --ckpt runs/probe/final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol_diag import load_model, load_tokens  # noqa: E402


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
    ap.add_argument("--data-offset", type=int, default=1024,
                    help="the default is training data; pass a held-out offset")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, cfg = load_model(args.ckpt, args.config, args.device, dtype,
                            depth=args.depth)

    C1 = cfg.levels[0].chunk_size
    C2 = cfg.levels[1].chunk_size if cfg.n_levels > 1 else 1
    cp = cfg.chunk_product

    tot = torch.zeros(cp, dtype=torch.float64)
    n = torch.zeros(cp, dtype=torch.float64)

    for b in range(args.batches):
        args.offset = args.data_offset + b * args.batch * args.seq_len
        tokens = load_tokens(cfg, args, args.device)
        out = model(tokens, return_logits=True)
        ce = F.cross_entropy(out.logits.float().flatten(0, 1),
                             tokens.reshape(-1), reduction="none")
        ce = ce.view(tokens.shape)
        # drop the first meta-group: it has no history and would dominate
        ce = ce[:, cp:]
        m = ce.shape[1] // cp * cp
        per = ce[:, :m].reshape(-1, cp).double().cpu()
        tot += per.sum(0)
        n += per.shape[0]

    mean = (tot / n)
    print(f"\nC_1={C1}  C_2={C2}  meta-context={cp} tokens   "
          f"{int(n[0])} samples/position")
    print(f"overall CE {mean.mean():.4f}\n")

    print("position within the meta-context (row = level-1 chunk, "
          "col = slot in chunk)")
    header = "        " + "".join(f"{j:>9}" for j in range(C1))
    print(header)
    grid = mean.view(C2, C1)
    for g in range(C2):
        row = "".join(f"{grid[g, j]:>9.4f}" for j in range(C1))
        print(f"chunk {g}{row}")

    slot = grid.mean(0)
    chunk = grid.mean(1)
    print(f"\nmean by slot-in-chunk : "
          + "  ".join(f"{j}={slot[j]:.4f}" for j in range(C1)))
    print(f"mean by chunk-in-group: "
          + "  ".join(f"{g}={chunk[g]:.4f}" for g in range(C2)))

    print(f"\ncost of the level-1 bottleneck  (slot 0 - slot {C1-1}): "
          f"{slot[0] - slot[-1]:+.4f} nats")
    print(f"cost of the level-2 bottleneck (chunk 0 - chunk {C2-1}): "
          f"{chunk[0] - chunk[-1]:+.4f} nats")
    print(f"first token of a meta-group vs last: "
          f"{grid[0, 0] - grid[-1, -1]:+.4f} nats")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"per_position": mean.tolist(), "C1": C1, "C2": C2}, indent=2))


if __name__ == "__main__":
    main()
