#!/usr/bin/env python
"""Gather a run's artifacts (eval*.json, protocol_diag.json, log.jsonl) across
the bf16 export and quantised siblings into one markdown table. RecGen
agreement can degrade under quantisation independent of perplexity.

    python scripts/collect_results.py /workspace/export/scala-8b-a1b-v3 --run /workspace/runs/scala-8b-a1b-v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing stage should not stop the table
        return None


def first(d: dict | None, *keys, default=None):
    if not d:
        return default
    for k in keys:
        if k in d:
            return d[k]
    # allow prefix matches, e.g. "ppl/globis-university/aozorabunko-clean"
    for k in keys:
        for kk, v in d.items():
            if kk.startswith(k):
                return v
    return default


def fmt(v, spec=".2f"):
    return format(v, spec) if isinstance(v, (int, float)) else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="the bf16 export directory")
    ap.add_argument("--run", default=None, help="training run dir (log.jsonl)")
    ap.add_argument("--tags", nargs="*",
                    default=["fp8", "nvfp4", "nvfp4-hier", "mxfp4"])
    args = ap.parse_args()

    exp = Path(args.export)

    # ---- training curve --------------------------------------------------
    if args.run:
        rows = []
        lg = Path(args.run) / "log.jsonl"
        if lg.exists():
            rows = [json.loads(l) for l in lg.read_text().splitlines() if l.strip()]
        if rows:
            print("## Training\n")
            print(f"{len(rows)} logged steps, "
                  f"final step {rows[-1].get('step')}, "
                  f"{rows[-1].get('tokens', 0)/1e6:.0f}M tokens\n")
            print("| step | loss | token CE | ppl | MaxVio mean | MaxVio max |")
            print("|---:|---:|---:|---:|---:|---:|")
            step = max(len(rows) // 12, 1)
            # rows[::step] rarely lands on the last row, and the final numbers
            # are the ones anyone reads -- append it unless it is already there
            shown = rows[::step]
            if shown[-1] is not rows[-1]:
                shown.append(rows[-1])
            for r in shown:
                print(f"| {r.get('step')} | {fmt(r.get('loss'), '.4f')} | "
                      f"{fmt(r.get('loss_token'), '.4f')} | "
                      f"{fmt(r.get('ppl'), '.1f')} | "
                      f"{fmt(r.get('moe/maxvio_mean'), '.2f')} | "
                      f"{fmt(r.get('moe/maxvio_max'), '.2f')} |")
            print()

    # ---- precision comparison -------------------------------------------
    print("## Precision\n")
    print("| variant | ja-wiki ppl | en-wiki ppl | neutral ppl | "
          "RecGen agreement | KL | size |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    def row(label: str, d: Path, ev: str, rg: str):
        e = load(d / ev)
        g = load(d / rg)
        st = d / "model.safetensors"
        size = f"{st.stat().st_size / 2**30:.2f} GiB" if st.exists() else "-"
        agree, kl = first(g, "agreement"), first(g, "kl")
        # prefer protocol_diag.json: it scores against the teacher-forced
        # training forward; agreement against HierGen is only as good as
        # HierGen.
        pd = load(d / "protocol_diag.json")
        if pd:
            hier = next((r for r in pd if r["protocol"] == "hiergen"), None)
            best = next((r for r in pd if r["protocol"] == "chunkgen"), None)
            if hier and hier["agree_vs_train"] < 0.90:
                print(f"| {label} | **HierGen scores "
                      f"{hier['agree_vs_train']:.1%} against the training "
                      f"forward -- the cached inference path does not match "
                      f"training; fix that before reading anything else** |")
                return
            if best:
                agree, kl = best["agree_vs_train"], best["kl_vs_train"]
        print(f"| {label} | {fmt(first(e, 'ppl/wiki_ja'))} | "
              f"{fmt(first(e, 'ppl/wiki_en'))} | "
              f"{fmt(first(e, 'ppl/globis'))} | "
              f"{fmt(agree, '.1%')} | "
              f"{fmt(kl, '.3f')} | {size} |")

    row("bf16", exp, "eval_bf16.json", "recgen_bf16.json")
    for tag in args.tags:
        d = Path(str(exp) + f"-{tag}")
        if d.exists():
            row(tag, d, "eval.json", "recgen.json")

    # ---- what each quantiser actually did --------------------------------
    print("\n## Quantisation detail\n")
    for tag in args.tags:
        q = load(Path(str(exp) + f"-{tag}") / "quantization_config.json")
        if not q:
            continue
        by = q.get("parameters_by_format", {})
        share = ", ".join(f"{k} {v/1e9:.2f}B" for k, v in by.items())
        print(f"* **{tag}** — {q.get('quant_method')} / "
              f"policy `{q.get('policy')}`, block {q.get('block_size')}; "
              f"{share}; kept {q.get('kept_parameters', 0)/1e6:.0f}M in bf16")


if __name__ == "__main__":
    main()
