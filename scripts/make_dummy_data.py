#!/usr/bin/env python
"""Generate a small synthetic corpus in the shard format, for smoke tests.

The corpus is deliberately *learnable*: a bank of random motifs is sampled and
repeated, so a working model's loss must fall well below ln(vocab).  If the
loss sits flat at ln(vocab) something is wired wrong.

    python scripts/make_dummy_data.py --out data/tokens_dummy --vocab 2048
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tokens_dummy")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=8_000_000)
    ap.add_argument("--motifs", type=int, default=512)
    ap.add_argument("--motif-len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bank = rng.integers(1, args.vocab, size=(args.motifs, args.motif_len),
                        dtype=np.uint32)
    # Zipf-ish motif frequencies, so the model has something to compress.
    p = 1.0 / np.arange(1, args.motifs + 1) ** 1.1
    p /= p.sum()

    n_draw = args.tokens // args.motif_len + 1
    idx = rng.choice(args.motifs, size=n_draw, p=p)
    stream = bank[idx].reshape(-1)[: args.tokens].copy()
    # sprinkle an EOS-like id so packing boundaries are exercised
    stream[rng.integers(0, args.tokens, size=args.tokens // 400)] = 0

    out = Path(args.out)
    (out / "dummy").mkdir(parents=True, exist_ok=True)
    path = out / "dummy" / "dummy-00000.bin"
    stream.tofile(path)

    manifest = {
        "tokenizer": "synthetic",
        "vocab_size": args.vocab,
        "dtype": "uint32",
        "sources": [{
            "name": "dummy", "weight": 1.0, "epochs_cap": 1e9,
            "shards": [{"path": "dummy/dummy-00000.bin",
                        "n_tokens": int(len(stream))}],
        }],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    print(f"wrote {len(stream):,} tokens to {path}")
    print(f"entropy floor is well below ln({args.vocab}) = "
          f"{float(np.log(args.vocab)):.3f}")


if __name__ == "__main__":
    main()
