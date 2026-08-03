#!/usr/bin/env python
"""BPC = nll / (ln2 * chars_per_token): the tokenizer-invariant quality figure.
Per-token perplexity cannot compare models with different vocabularies
(llm-jp v4 = 196,608 vs v3 = 99,574).  chars_per_token is measured on the
exact held-out slice the evaluation used.

    python scripts/bpc_compare.py
"""
import math
from datasets import load_dataset
from transformers import AutoTokenizer

SKIP, N = 700_000, 400
ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train",
                  streaming=True)
texts = []
for i, r in enumerate(ds):
    if i < SKIP:
        continue
    if i >= SKIP + N:
        break
    texts.append(r["text"])
chars = sum(len(t) for t in texts)
print(f"held-out slice: {len(texts)} articles, {chars:,} chars")

res = {}
for name in ("llm-jp/llm-jp-4-8b-base", "llm-jp/llm-jp-3-1.8b"):
    tk = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    n = sum(len(tk(t, add_special_tokens=False)["input_ids"]) for t in texts)
    res[name] = chars / n
    print(f"  {name:<28} {n:>10,} tokens   {chars/n:.4f} chars/token")

# nll from the runs; ppl values are what the model cards would otherwise quote
RUNS = [
    ("8B v3        (v4 tok, 344M tok)", 69.00, "llm-jp/llm-jp-4-8b-base"),
    ("250M v3-900m (v3 tok, 900M tok)", 67.60, "llm-jp/llm-jp-3-1.8b"),
    ("250M v4      (v3 tok, 200M tok)", 51.61, "llm-jp/llm-jp-3-1.8b"),
]
print(f"\n{'run':<34}{'ja-wiki ppl':>12}{'chars/tok':>11}{'BPC':>8}")
print("-" * 65)
for label, ppl, tok in RUNS:
    nll = math.log(ppl)
    bpc = nll / (math.log(2) * res[tok])
    print(f"{label:<34}{ppl:>12.2f}{res[tok]:>11.4f}{bpc:>8.4f}")
print("\nLower BPC is better.  This is the only one of these three columns that "
      "compares\nacross tokenizers.")
