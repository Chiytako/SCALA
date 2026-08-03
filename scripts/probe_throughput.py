#!/usr/bin/env python
"""Find the micro-batch that trains fastest without OOM. Sweeps micro-batch
sizes, reports tok/s and peak allocation, stops at the first OOM.
`--max-memory-fraction` caps the allocator, needed on unified-memory systems
where the GPU and OS share one pool.

    python scripts/probe_throughput.py --config configs/base_small.yaml --seq-len 1024 --batches 8 16 32 --max-memory-fraction 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.model.config import ScalaConfig  # noqa: E402
from scala.model.hierarchy import ScalaForCausalLM  # noqa: E402
from scala.train.optim import build_optimizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_small.yaml")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batches", type=int, nargs="*", default=[8, 16, 32, 64])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-memory-fraction", type=float, default=0.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--activation-checkpointing", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"device: {props.name}  sm_{props.major}{props.minor}  "
              f"{props.total_memory/2**30:.0f} GiB")
        if args.max_memory_fraction > 0:
            torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction)
            print(f"allocator capped at {args.max_memory_fraction:.0%} "
                  f"({props.total_memory/2**30*args.max_memory_fraction:.0f} GiB)")
    torch.set_float32_matmul_precision("high")

    cfg = ScalaConfig.load(args.config)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    model = ScalaForCausalLM(cfg).to(dev)
    model.train()
    if args.activation_checkpointing:
        model.set_gradient_checkpointing(True)
    opt = build_optimizer(model, lr=1e-4, verbose=False)

    import scala.model.moe as moe

    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f}M  "
          f"loss_chunk={model.loss_chunk_tokens}  "
          f"grouped_mm={moe._grouped_mm_ok}")

    if args.compile:
        from scala.model.layers import TransformerBlock

        model.compile_loss()
        for m in model.modules():
            if isinstance(m, TransformerBlock):
                m.attn.compile(dynamic=False)
        print("compiled: loss slice + attention blocks")

    results = []
    for mb in args.batches:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randint(0, cfg.vocab_size, (mb, args.seq_len), device=dev)
            for i in range(args.warmup + args.steps):
                if i == args.warmup:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=dev.type == "cuda"):
                    out = model(x, labels=x, return_logits=False)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / args.steps
            peak = torch.cuda.max_memory_allocated() / 2**30
            tps = mb * args.seq_len / dt
            results.append({"micro_batch": mb, "tok_per_s": tps,
                            "peak_gib": peak, "ms_per_step": dt * 1000})
            print(f"mb={mb:4d}  {tps/1000:8.1f}K tok/s  peak {peak:6.2f} GiB  "
                  f"{dt*1000:7.0f} ms/step")
        except torch.OutOfMemoryError:
            print(f"mb={mb:4d}  OOM (allocator cap held -- machine is fine)")
            torch.cuda.empty_cache()
            break

    if results:
        best = max(results, key=lambda r: r["tok_per_s"])
        print(f"\nbest: micro_batch={best['micro_batch']} "
              f"{best['tok_per_s']/1000:.1f}K tok/s @ {best['peak_gib']:.1f} GiB")
        for budget in (200e6, 400e6):
            print(f"  {budget/1e6:.0f}M tokens -> "
                  f"{budget/best['tok_per_s']/3600:.2f} h")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
