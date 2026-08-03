#!/usr/bin/env python
"""Evaluate a SCALA checkpoint on Japanese benchmarks (multiple choice and
held-out perplexity/BPC; --ppl-only skips the tasks).

    python scripts/eval_ja.py --ckpt runs/scala-8b-a1b/final --tasks jcommonsenseqa jnli marc_ja jmmlu --limit 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.eval.harness import (  # noqa: E402
    JA_TASKS, build_task, evaluate_multiple_choice, evaluate_perplexity,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import load as load_model  # noqa: E402


def wiki_texts(config: str, n: int, skip: int = 0):
    """`n` articles after skipping the first `skip`; `skip` must exceed the
    articles training consumed or this measures training data."""
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", config, split="train",
                      streaming=True)
    for i, rec in enumerate(ds):
        if i < skip:
            continue
        if i >= skip + n:
            break
        yield rec["text"]


def local_jsonl_texts(pattern: str, n: int, text_key: str = "text"):
    """`n` documents from local .jsonl(.gz) files matching `pattern`; the
    held-out guarantee is `skip_files` in the data config -- point `pattern`
    at files training excluded."""
    import glob as _glob
    import gzip

    files = sorted(_glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"no file matched {pattern}")
    seen = 0
    for path in files:
        op = (gzip.open(path, "rt", encoding="utf-8", errors="replace")
              if path.endswith(".gz")
              else open(path, "rt", encoding="utf-8", errors="replace"))
        with op as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = rec.get(text_key)
                if isinstance(v, str) and v.strip():
                    yield v
                    seen += 1
                    if seen >= n:
                        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tokenizer", default="llm-jp/llm-jp-3-1.8b")
    ap.add_argument("--tasks", nargs="*", default=list(JA_TASKS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-shot", type=int, default=None)
    ap.add_argument("--ppl-only", action="store_true")
    ap.add_argument("--ppl-sequences", type=int, default=100)
    # skip defaults must exceed the articles any run consumed
    ap.add_argument("--ppl-skip-ja", type=int, default=700_000,
                    help="articles to skip so the sample is genuinely unseen")
    ap.add_argument("--ppl-skip-en", type=int, default=400_000)
    ap.add_argument("--ppl-extra", nargs="*", default=[],
                    help="extra HF text datasets as repo[:text_key[:skip]]. "
                         "`skip` matters as much here as it does for wikipedia: "
                         "globis-university/aozorabunko-clean was the neutral "
                         "corpus for the v1-v4 comparison, but data_ja_v3.yaml "
                         "puts it *in* the training mixture, so from v3 onward "
                         "it is only held out beyond the records the run "
                         "actually consumed.")
    ap.add_argument("--local-jsonl", nargs="*", default=[],
                    help="held-out local corpora as label:glob[:text_key]. "
                         "For offline machines (the ACRi room has no route to "
                         "the Hub). The glob must name files the training "
                         "mixture excluded -- see `skip_files` in "
                         "configs/data_ja_acri.yaml.")
    ap.add_argument("--skip-hf-wiki", action="store_true",
                    help="do not touch wikimedia/wikipedia (it needs the Hub)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model, _ = load_model(args.ckpt, args.config, args.device,
                          getattr(torch, args.dtype))
    dtype = getattr(torch, args.dtype)
    results: dict[str, float] = {}

    def report(label: str, r: dict) -> None:
        results[f"ppl/{label}"] = r["ppl"]
        results[f"bpc/{label}"] = r["bpc"]
        results[f"chars_per_token/{label}"] = r["chars_per_token"]
        # BPC first: the only figure that survives a change of tokenizer
        print(f"  {label}: BPC={r['bpc']:.4f}  ppl={r['ppl']:.3f} "
              f"nll={r['nll']:.4f}  {r['chars_per_token']:.4f} chars/token "
              f"over {int(r['tokens']):,} tokens")

    print("\n== perplexity ==")
    if not args.skip_hf_wiki:
        for label, cfg in (("ja", "20231101.ja"), ("en", "20231101.en")):
            skip = args.ppl_skip_ja if label == "ja" else args.ppl_skip_en
            # 2x the sequence count in articles: short articles are common, and
            # running out of text silently shrinks the sample instead of failing.
            r = evaluate_perplexity(model, tok,
                                    wiki_texts(cfg, 2 * args.ppl_sequences, skip),
                                    args.seq_len, args.device, dtype,
                                    args.ppl_sequences)
            report(f"wiki_{label}", r)

    for spec in args.local_jsonl:
        label, pattern, *rest = spec.split(":")
        key = rest[0] if rest and rest[0] else "text"
        r = evaluate_perplexity(model, tok,
                                local_jsonl_texts(pattern,
                                                  4 * args.ppl_sequences, key),
                                args.seq_len, args.device, dtype,
                                args.ppl_sequences)
        report(label, r)

    # --ppl-extra takes any HF text dataset; for cross-mixture comparisons
    # pass one that neither model has seen.
    for spec in args.ppl_extra:
        from datasets import load_dataset
        parts = spec.split(":")
        repo = parts[0]
        key = parts[1] if len(parts) > 1 and parts[1] else "text"
        skip = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        ds = load_dataset(repo, split="train", streaming=True)

        def _texts(ds=ds, key=key, skip=skip):
            for i, rec in enumerate(ds):
                if i < skip:
                    continue
                if i >= skip + 2 * args.ppl_sequences:
                    break
                v = rec.get(key)
                if isinstance(v, str) and v.strip():
                    yield v

        r = evaluate_perplexity(model, tok, _texts(), args.seq_len,
                                args.device, dtype, args.ppl_sequences)
        report(repo, r)
        results[f"ppl/{repo}/skip"] = skip

    if not args.ppl_only:
        print("\n== multiple choice ==")
        for name in args.tasks:
            if name not in JA_TASKS:
                print(f"  [skip] unknown task {name}")
                continue
            ov = {}
            if args.limit is not None:
                ov["limit"] = args.limit
            if args.n_shot is not None:
                ov["n_shot"] = args.n_shot
            try:
                r = evaluate_multiple_choice(model, tok, build_task(name, **ov),
                                             args.device, dtype)
                results.update(r)
                print(f"  {name}: acc={r[f'{name}/acc']:.4f} "
                      f"(n={int(r[f'{name}/n'])})")
            except Exception as e:  # noqa: BLE001
                print(f"  [fail] {name}: {type(e).__name__}: {e}")

    print("\n" + json.dumps(results, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
