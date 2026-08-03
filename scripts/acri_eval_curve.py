#!/usr/bin/env python
"""Evaluate each checkpoint a run produces; appends one JSON line per
checkpoint (BPC per held-out corpus, x-axis = training tokens) to
`<run>/curve.jsonl`. Pass the same held-out slices earlier numbers were
measured on.

    python scripts/acri_eval_curve.py --run <run> --tokenizer <dir> --heldout ja_wiki:eval_data/wiki_ja_heldout.jsonl.gz --watch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_ja import local_jsonl_texts  # noqa: E402

from scala.eval.harness import evaluate_perplexity  # noqa: E402
from generate import load as load_model  # noqa: E402


def steps_done(run: Path) -> dict[int, int]:
    """step -> tokens, from the training log, so the curve's x-axis is tokens."""
    out: dict[int, int] = {}
    log = run / "log.jsonl"
    if not log.exists():
        return out
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in d and "tokens" in d:
            out[int(d["step"])] = int(d["tokens"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--heldout", required=True, nargs="+",
                    help="held-out corpora as label:glob -- local jsonl(.gz) "
                         "files the training mixture never opened.  Repeat the "
                         "flag's values to track several at once; each gets its "
                         "own BPC column in curve.jsonl.")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--sequences", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--stop-after-final", action="store_true", default=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    dtype = getattr(torch, args.dtype)
    run = Path(args.run)
    curve = run / "curve.jsonl"

    seen: set[str] = set()
    if curve.exists():
        for line in curve.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["ckpt"])
            except (json.JSONDecodeError, KeyError):
                pass

    while True:
        tokens_at = steps_done(run)
        cands = sorted([p for p in run.glob("step-*") if p.is_dir()],
                       key=lambda p: p.name)
        final = run / "final"
        if final.is_dir():
            cands.append(final)

        for ck in cands:
            if ck.name in seen:
                continue
            # A checkpoint directory appears before it is fully written.  Wait
            # for it to stop growing rather than reading a half-flushed shard.
            size = -1
            for _ in range(30):
                cur = sum(f.stat().st_size for f in ck.rglob("*") if f.is_file())
                if cur == size and cur > 0:
                    break
                size, _ = cur, time.sleep(2)
            try:
                model, _ = load_model(str(ck), None, args.device, dtype)
            except Exception as e:  # noqa: BLE001 -- a torn checkpoint is retried next poll
                print(f"[{ck.name}] not loadable yet: {type(e).__name__}: {e}",
                      flush=True)
                continue

            step = int(ck.name.split("-")[-1]) if ck.name != "final" else max(
                tokens_at or {0: 0})
            row = {"ckpt": ck.name, "step": step, "tokens": tokens_at.get(step)}
            parts = []
            for spec in args.heldout:
                label, pattern = spec.split(":", 1)
                r = evaluate_perplexity(
                    model, tok,
                    local_jsonl_texts(pattern, 4 * args.sequences, args.text_key),
                    args.seq_len, args.device, dtype, args.sequences)
                row[f"bpc/{label}"] = r["bpc"]
                row[f"ppl/{label}"] = r["ppl"]
                row[f"nll/{label}"] = r["nll"]
                row[f"cpt/{label}"] = r["chars_per_token"]
                row[f"eval_tokens/{label}"] = r["tokens"]
                parts.append(f"{label} BPC={r['bpc']:.4f} ppl={r['ppl']:.2f}")
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

            with curve.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            seen.add(ck.name)
            print(f"[{ck.name}] tokens={row['tokens']}  " + "  ".join(parts),
                  flush=True)

            if ck.name == "final" and args.stop_after_final:
                print("final evaluated; done", flush=True)
                return

        if not args.watch:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
