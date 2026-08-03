#!/usr/bin/env python
"""Print the analytic parameter / FLOP / KV-cache report for a config.

    python scripts/count_params.py configs/base_8b_a1b.yaml
    python scripts/count_params.py configs/base_8b_a1b.yaml --compare-dense
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.model.accounting import (  # noqa: E402
    count_model, flops_per_token, format_report, kv_cache_bytes_per_token,
    scala_state_bytes,
)
from scala.model.config import ScalaConfig  # noqa: E402


def dense_baseline(cfg: ScalaConfig, n_layers: int, d: int, inter: int,
                   n_kv_heads: int = 8, head_dim: int = 128) -> dict:
    """A modern dense transformer (GQA, same vocab) for a FLOP/KV contrast."""
    n_heads = d // head_dim
    attn = 2 * d * n_heads * head_dim + 2 * d * n_kv_heads * head_dim
    ffn = 3 * d * inter
    total = n_layers * (attn + ffn + 2 * d) + 2 * cfg.vocab_size * d
    kv = n_layers * 2 * n_kv_heads * head_dim * 2   # bytes/token @ bf16
    return {"params": total, "kv_bytes_per_token": kv,
            "flops_per_token": 2 * total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/base_8b_a1b.yaml")
    ap.add_argument("--compare-dense", action="store_true")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--u-max", type=int, default=None,
                    help="SCALA bounded-top policy: max CAP units before the "
                         "checkpoint is re-expressed one level deeper")
    ap.add_argument("--context", type=int, action="append", default=None,
                    help="context length(s) to price the policy at "
                         "(repeatable; needs --u-max and a tied config)")
    args = ap.parse_args()

    cfg = ScalaConfig.load(args.config)
    if args.seq_len:
        cfg.max_seq_len = args.seq_len
    print(format_report(cfg))

    if args.u_max:
        if not cfg.tie_mid_levels:
            ap.error("--u-max needs a tie_mid_levels config")
        contexts = args.context or [8192, 32768, 131072]
        print(f"\nSCALA bounded-top policy (u_max={args.u_max}; absolute "
              "resident state per sequence -- the claim is that this column "
              "is O(log T), so it is not divided by T):")
        print(f"{'context':>10}{'depth':>7}{'L':>4}{'windows':>12}"
              f"{'cap':>10}{'decoders':>11}{'total':>12}")
        print("-" * 66)
        for t in sorted(contexts):
            s = scala_state_bytes(cfg, t, u_max=args.u_max)
            print(f"{t:>10,}{s['depth']:>7}{s['levels']:>4}"
                  f"{s['bytes_windows']/1024:>10.1f}Ki"
                  f"{s['bytes_cap']/1024:>8.1f}Ki"
                  f"{s['bytes_decoders']/1024:>9.1f}Ki"
                  f"{s['bytes_total']/1024:>10.1f}Ki")

    if args.compare_dense:
        mc = count_model(cfg)
        fl = flops_per_token(cfg)
        print("\nContrast with dense GQA transformers (same 99.6K vocab):")
        print(f"{'model':<28}{'params':>10}{'FLOPs/tok':>12}{'KV B/tok':>11}")
        print("-" * 61)
        print(f"{'SCALA  (HierGen)':<28}{mc.total/1e9:>8.2f}B"
              f"{fl['total']/1e9:>11.2f}G"
              f"{kv_cache_bytes_per_token(cfg):>11.0f}")
        print(f"{'SCALA  (RecGen)':<28}{'':>9}{'':>11}"
              f"{kv_cache_bytes_per_token(cfg, recgen=True):>11.0f}")
        for name, nl, d, i in [
            ("dense 1.5B (24L d2048)", 24, 2048, 5632),
            ("dense 8B   (32L d4096)", 32, 4096, 14336),
        ]:
            b = dense_baseline(cfg, nl, d, i)
            print(f"{name:<28}{b['params']/1e9:>8.2f}B"
                  f"{b['flops_per_token']/1e9:>11.2f}G"
                  f"{b['kv_bytes_per_token']:>11.0f}")


if __name__ == "__main__":
    main()
