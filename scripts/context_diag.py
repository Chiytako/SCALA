#!/usr/bin/env python
"""Long-range context probes: ce_by_position (CE vs absolute position; shards
concatenate documents, so later positions cross more boundaries),
influence_by_distance (perturb token i, measure |dlogit| at i+d), and
ablate_far_context (replace context beyond d with noise; measures use, not
potential).

    python scripts/context_diag.py --ckpt runs/probe/final
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
def ce_by_position(model, cfg, args) -> list[tuple[int, float]]:
    T = args.seq_len
    tot = torch.zeros(T, dtype=torch.float64)
    n = 0
    for b in range(args.batches):
        args.offset = args.data_offset + b * args.batch * T
        tokens = load_tokens(cfg, args, args.device)
        logits = model(tokens, return_logits=True).logits
        ce = F.cross_entropy(logits.float().flatten(0, 1), tokens.reshape(-1),
                             reduction="none").view(tokens.shape)
        tot += ce.double().sum(0).cpu()
        n += tokens.shape[0]
    mean = tot / n

    cp = cfg.chunk_product
    out = []
    lo = cp
    while lo < T:
        hi = min(lo * 2, T)
        out.append((lo, mean[lo:hi].mean().item()))
        lo = hi
    return out


@torch.no_grad()
def influence_by_distance(model, cfg, args) -> list[tuple[int, float]]:
    """Move one token; see how far the change travels.

    The probe token sits at the start of a meta-context so that every distance
    is measured from the same phase of the hierarchy -- otherwise the
    within-chunk sawtooth (`position_diag.py`) shows up as distance structure.
    """
    cp = cfg.chunk_product
    args.offset = args.data_offset + 3072
    tokens = load_tokens(cfg, args, args.device)
    i = 4 * cp                                   # a meta-context boundary
    base = model(tokens, return_logits=True).logits.float()

    alt = tokens.clone()
    alt[:, i] = (alt[:, i] + 7919) % cfg.vocab_size
    pert = model(alt, return_logits=True).logits.float()

    delta = (base - pert).abs().amax(dim=-1).mean(0)      # (T,)
    out = []
    d = 1
    while i + d < tokens.shape[1]:
        hi = min(i + 2 * d, tokens.shape[1])
        out.append((d, delta[i + d : hi].mean().item()))
        d *= 2
    return out


@torch.no_grad()
def ablate_far_context(model, cfg, args) -> list[tuple[int, float]]:
    """Corrupt everything further back than `d` and see what the loss does.

    Needs no reference model: measures how many nats the model's own
    predictions lose when far context is replaced with noise.  Scoring is
    confined to the last `--score-window` positions so every `d` is graded
    on the same tokens.
    """
    T, cp = args.seq_len, cfg.chunk_product
    win = args.score_window
    if win >= T - cp:
        # `keep = T - win - d` would be <= 0 for every d: nothing corrupted,
        # every arm reports 0.0 -- fail loudly instead.
        raise SystemExit(
            f"--score-window {win} leaves no context to ablate at seq_len {T} "
            f"(need win < T - {cp}).  Use a small window here; widen --batches "
            f"instead if you want a lower-variance number.")
    g = torch.Generator(device="cpu").manual_seed(1234)

    out = []
    for d in [T] + [cp * 2**k for k in range(0, 16) if cp * 2**k < T - win][::-1]:
        tot, n = 0.0, 0
        for b in range(args.batches):
            args.offset = args.data_offset + b * args.batch * T
            tokens = load_tokens(cfg, args, args.device)
            x = tokens.clone()
            keep = T - win - d                 # corrupt positions [0, keep)
            if keep > 0:
                noise = torch.randint(0, cfg.vocab_size, (x.shape[0], keep),
                                      generator=g).to(x.device)
                x[:, :keep] = noise
            # score only the last `win` positions, without materialising a
            # (batch, seq, vocab) logit tensor -- see `compare_arms.eval_ce`
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:, -win:] = True
            res = model(x, labels=tokens, return_logits=False, loss_mask=mask)
            tot += float(res.loss_token) * x.shape[0]
            n += x.shape[0]
        out.append((min(d, T), tot / n))
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--depth", type=int, default=None,
                    help="re-express a tied (SCALA) checkpoint at this many "
                         "MID applications before scoring")
    ap.add_argument("--data-root", default="data/tokens_ja")
    ap.add_argument("--data-source", default=None,
                    help="manifest source to score on; default is the first, "
                         "which for data/tokens_ja is English")
    ap.add_argument("--data-offset", type=int, default=1024,
                    help="token offset into the shard.  The default sits in "
                         "the prefix every training run reads first, so it "
                         "scores training data; pass a value past "
                         "total_tokens*weight for a real holdout")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--score-window", type=int, default=64,
                    help="positions at the end of the sequence that the "
                         "context-ablation sweep is scored on")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, args.config, args.device,
                            getattr(torch, args.dtype), depth=args.depth)
    cp = cfg.chunk_product

    ce = ce_by_position(model, cfg, args)
    print(f"\nCE by position (meta-context = {cp} tokens)")
    print(f"{'positions':>14}{'CE':>10}{'vs first':>10}")
    for lo, v in ce:
        print(f"{lo:>7}-{min(lo*2, args.seq_len):<6}{v:>10.4f}"
              f"{v - ce[0][1]:>+10.4f}")
    print(f"\ncontext gain over the measured range: {ce[0][1] - ce[-1][1]:+.4f} nats")

    inf = influence_by_distance(model, cfg, args)
    print(f"\nmax |delta logit| at distance d from a perturbed token")
    print(f"{'d':>8}{'|dlogit|':>14}{'vs d=1':>12}")
    for d, v in inf:
        print(f"{d:>8}{v:>14.4e}{v / max(inf[0][1], 1e-30):>12.3e}")
    beyond = [v for d, v in inf if d >= cp]
    within = [v for d, v in inf if d < cp]
    if beyond and within:
        print(f"\nmean within one meta-context: {sum(within)/len(within):.4e}")
        print(f"mean beyond                 : {sum(beyond)/len(beyond):.4e}")

    abl = ablate_far_context(model, cfg, args)
    full = abl[0][1]
    print(f"\nCE on the last {args.score_window} positions, with everything "
          f"further back than d replaced by noise")
    print(f"{'d':>8}{'CE':>10}{'nats lost':>12}")
    for d, v in abl:
        print(f"{d:>8}{v:>10.4f}{v - full:>+12.4f}")
    used = abl[-1][1] - full
    print(f"\ncontext the model actually uses beyond {abl[-1][0]} tokens: "
          f"{used:+.4f} nats")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"ce_by_position": ce, "influence_by_distance": inf,
             "ablation": abl}, indent=2))


if __name__ == "__main__":
    main()
