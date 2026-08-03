#!/usr/bin/env python
"""Break a training step into its parts (encode / decode / LM-loss / MTP /
backward / optimizer) and time each one.

    python scripts/profile_components.py --config configs/base_small.yaml --batch 16 --seq-len 1024
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.model.config import ScalaConfig  # noqa: E402
from scala.model.hierarchy import ScalaForCausalLM  # noqa: E402
from scala.train.optim import build_optimizer  # noqa: E402


def bench(fn, n=3, warmup=2):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_small.yaml")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-memory-fraction", type=float, default=0.0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda" and args.max_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction)
    torch.set_float32_matmul_precision("high")

    cfg = ScalaConfig.load(args.config)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    m = ScalaForCausalLM(cfg).to(dev).train()
    opt = build_optimizer(m, lr=1e-4, verbose=False)
    x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len), device=dev)
    ntok = x.numel()

    def ac():
        return torch.autocast("cuda", dtype=torch.bfloat16,
                              enabled=dev.type == "cuda")

    print(f"batch={args.batch} seq={args.seq_len} -> {ntok:,} tokens/step")
    print(f"vocab={cfg.vocab_size:,}  d_token={cfg.d_token}  "
          f"loss_chunk={m.loss_chunk_tokens}  mtp={cfg.mtp_depth}")
    lm_flops = 2 * ntok * cfg.d_token * cfg.vocab_size / 1e12
    print(f"LM-head GEMM alone: {lm_flops:.2f} TFLOP per pass\n")

    rows = []

    with torch.no_grad(), ac():
        rows.append(("encode_all (no grad)", bench(lambda: m.encode_all(x))))
        enc = m.encode_all(x)
        rows.append(("decode_all (no grad)", bench(lambda: m.decode_all(enc))))
        dec = m.decode_all(enc)
        h0 = m.final_norm(dec[0])
        rows.append(("lm_loss (no grad)", bench(lambda: m._lm_loss(h0, x))))
        if cfg.mtp_depth:
            rows.append(("mtp_loss (no grad)", bench(lambda: m._mtp_loss(h0, x, x, None))))

    def fwd():
        with ac():
            return m(x, labels=x, return_logits=False).loss

    rows.append(("full forward (grad)", bench(fwd)))

    def fwd_bwd():
        with ac():
            loss = m(x, labels=x, return_logits=False).loss
        loss.backward()
        m.zero_grad(set_to_none=True)

    rows.append(("forward + backward", bench(fwd_bwd)))

    with ac():
        m(x, labels=x, return_logits=False).loss.backward()
    rows.append(("optimizer.step", bench(lambda: opt.step())))

    # what does it cost with MTP off?
    if cfg.mtp_depth:
        saved, m.mtp = m.mtp, None
        rows.append(("forward + backward, MTP off", bench(fwd_bwd)))
        m.mtp = saved

    print(f"{'component':<32}{'ms':>10}{'% of fwd+bwd':>15}")
    print("-" * 57)
    fb = next(v for k, v in rows if k == "forward + backward")
    for k, v in rows:
        print(f"{k:<32}{v:>10.1f}{100*v/fb:>14.1f}%")
    print("-" * 57)
    print(f"{'throughput':<32}{ntok/(fb/1000):>10,.0f} tok/s")
    if dev.type == "cuda":
        print(f"{'peak memory':<32}"
              f"{torch.cuda.max_memory_allocated()/2**30:>10.2f} GiB")


if __name__ == "__main__":
    main()
