#!/usr/bin/env bash
# Runs post-training steps once an ACRi run writes `final`: mirrors the
# checkpoint, then computes held-out BPC. Usage: scripts/acri_finish.sh [run_dir]
set -uo pipefail

RUN="${1:-/scratch/$USER/runs/scala-small-v4-acri}"
REPO="/scratch/$USER/SCALA"
PY="/scratch/$USER/unsloth-ft/.venv/bin/python"
TOK="$REPO/tokenizers/llm-jp-tok-v3"
E="$REPO/eval_data"
OUT="$RUN/final_report"

export HF_HOME="/scratch/$USER/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd "$REPO"
echo "$(date +%T) waiting for $RUN/final"
while [ ! -d "$RUN/final" ]; do sleep 120; done
sleep 60                                  # let the last shard finish flushing
mkdir -p "$OUT"
echo "$(date +%T) final/ appeared"

bash scripts/acri_sync_ckpt.sh "$RUN" | tee "$OUT/mirror.txt"

echo "$(date +%T) == held-out BPC =="
"$PY" scripts/eval_ja.py --ckpt "$RUN/final" --tokenizer "$TOK" \
    --ppl-only --skip-hf-wiki --ppl-sequences 200 --seq-len 2048 \
    --local-jsonl "ja_wiki:$E/wiki_ja_heldout.jsonl.gz" \
                  "en_wiki:$E/wiki_en_heldout.jsonl.gz" \
                  "aozora:$E/aozora_heldout.jsonl.gz" \
    --out "$OUT/eval_final.json" 2>&1 | tee "$OUT/eval_final.txt"

cat > "$OUT/NEXT_STEPS.md" << 'MD'
# Left to do, with the corrected code

`final/` is safe and its held-out BPC is measured. What is **not** done, and
why, is the generation side: this run trained on the pre-§4g code, where the
weight-absorbed MLA path dropped the attention output gate. That path is
inference-only, so nothing above is affected -- but anything that generates is.

Sync the corrected `scala/model/{layers,photon,config}.py` and
`scripts/{generate,protocol_diag}.py` to this machine **now that no training
job can restart**, then:

    python scripts/protocol_diag.py --ckpt <run>/final \
        --tokenizer tokenizers/llm-jp-tok-v3
    python scripts/generate.py --ckpt <run>/final \
        --tokenizer tokenizers/llm-jp-tok-v3 --prompt 日本の首都は

`protocol_diag.py` scores each protocol against the teacher-forced training
forward, not against HierGen -- §4g is the record of why the old comparison
was measuring against a reference that was itself 39% wrong.

The question worth answering with this checkpoint: §4b's negative result held
at four model *sizes*. This is the first point on the *token* axis -- 10x the
budget of the 250M v4, scheduled sampling at its full 0.25 the whole second
half. If the protocol gaps look the same here, the token axis is closed too.
MD

echo "$(date +%T) done -- everything under $OUT"
echo "generation deliberately skipped; see $OUT/NEXT_STEPS.md"
ls -la "$OUT"
