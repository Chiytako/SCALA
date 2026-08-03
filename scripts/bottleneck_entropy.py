#!/usr/bin/env python
"""Measures how much of the top-level summary is decided only after the
meta-context is sampled (Definition A.4's unreachable part): within-condition
spread, centroid ceiling (best any pre-committed point can score), and
measured A_hat (RecGen's deterministic estimate).

    python scripts/bottleneck_entropy.py --ckpt runs/probe/final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.infer.generate import (  # noqa: E402
    PROTOCOLS, GenerationConfig, ScalaGenerator, _GenContext,
)
from protocol_diag import load_model, load_tokens  # noqa: E402


@torch.no_grad()
def summaries_from_one_state(gen: ScalaGenerator, x_top: torch.Tensor,
                             samples: int, temperature: float, seed: int):
    """Sample `samples` meta-contexts from the same top state and return, for
    each, the real ``A^(L)`` its tokens produce plus the single ``A_hat^(L)``
    RecGen would have used."""
    L, model = gen.L, gen.model
    hier, paper = PROTOCOLS["hiergen"], PROTOCOLS["recgen_paper"]
    B = x_top.shape[0]
    cp = gen.cfg.chunk_product

    reals = []
    for i in range(samples):
        # HierGen keeps the lower encoders, so the summary it folds is the real
        # bottom-up one -- the quantity Definition A.4 compares against.
        gen._alloc_caches(B, 4 * cp, hier)
        cfg = GenerationConfig(max_new_tokens=cp, temperature=temperature,
                               top_p=1.0, seed=seed + i)
        ctx = _GenContext(cfg, B, gen.device, x_top.new_zeros(B, 0).long(),
                          torch.Generator(device=gen.device).manual_seed(seed + i))
        sub = gen._emit_group(L, x_top, hier, ctx, x_top)
        reals.append(model.levels[-1].chunker(sub)[:, 0])

    gen._alloc_caches(B, 4 * cp, paper)
    cfg = GenerationConfig(max_new_tokens=cp, temperature=temperature, top_p=1.0)
    ctx = _GenContext(cfg, B, gen.device, x_top.new_zeros(B, 0).long(),
                      torch.Generator(device=gen.device).manual_seed(seed))
    sub = gen._emit_group(L, x_top, paper, ctx, x_top)
    a_hat = model.levels[-1].chunker(sub)[:, 0]

    return torch.stack(reals, dim=1).float(), a_hat.float()   # (B,S,D), (B,D)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default="data/tokens_ja")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, cfg = load_model(args.ckpt, args.config, args.device, dtype)
    tokens = load_tokens(cfg, args, args.device)
    gen = ScalaGenerator(model, args.device, dtype)

    gen._alloc_caches(tokens.shape[0], tokens.shape[1] + 4 * cfg.chunk_product,
                      PROTOCOLS["hiergen"])
    x_top, _ = gen._prefill(tokens, PROTOCOLS["hiergen"])
    reals, a_hat = summaries_from_one_state(gen, x_top, args.samples,
                                            args.temperature, args.seed)

    # spread of the real summary across draws of the same meta-context
    n = reals.shape[1]
    pair = F.cosine_similarity(reals[:, :, None], reals[:, None, :], dim=-1)
    off = ~torch.eye(n, dtype=torch.bool, device=pair.device)
    spread = pair[:, off].mean().item()

    # best a single pre-committed point can do: the centroid of the cloud
    centroid = reals.mean(dim=1)
    ceiling = F.cosine_similarity(centroid[:, None], reals, dim=-1).mean().item()
    measured = F.cosine_similarity(a_hat[:, None], reals, dim=-1).mean().item()

    print(f"\nmeta-context = {cfg.chunk_product} tokens, "
          f"{args.samples} draws at T={args.temperature}, batch {reals.shape[0]}")
    print(f"  within-condition spread   cos = {spread:.4f}   "
          f"(1.0 would mean the summary ignores what was sampled)")
    print(f"  ceiling for any A_hat     cos = {ceiling:.4f}   "
          f"(the centroid -- nothing chosen in advance beats it)")
    print(f"  measured A_hat            cos = {measured:.4f}")
    print(f"\n  reachable by training : {ceiling - measured:+.4f}")
    print(f"  structurally out of reach: {1.0 - ceiling:+.4f}  "
          f"<- Definition A.4's residual")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"spread": spread, "ceiling": ceiling, "measured": measured,
             "samples": args.samples, "temperature": args.temperature},
            indent=2))


if __name__ == "__main__":
    main()
