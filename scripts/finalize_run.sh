#!/usr/bin/env bash
# Post-training pipeline: evaluate, sample, benchmark, export, push. Each stage
# skips if its output already exists. Usage: RUN=<dir> bash scripts/finalize_run.sh
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/SCALA}"
RUN="${RUN:?set RUN to the run directory}"
CKPT="${CKPT:-$RUN/final}"
NAME="${NAME:-SCALA}"
REPO="${REPO:-}"
OUT="${OUT:-/workspace/export/$(basename "$RUN")}"
TOKENIZER="${TOKENIZER:-llm-jp/llm-jp-3-1.8b}"
DEVICE="${DEVICE:-cuda}"
PPL_SEQS="${PPL_SEQS:-60}"
MC_LIMIT="${MC_LIMIT:-300}"

log() { echo -e "\n\033[1;36m==> $*\033[0m"; }
cd "$WORKDIR"
mkdir -p "$OUT" "$(dirname "$OUT")"

[ -d "$CKPT" ] || { echo "no checkpoint at $CKPT"; ls "$RUN"; exit 1; }

# --------------------------------------------------------------------------- #
log "1/5 evaluate"
if [ ! -f "$RUN/eval.json" ]; then
    python scripts/eval_ja.py --ckpt "$CKPT" --tokenizer "$TOKENIZER" \
        --device "$DEVICE" --ppl-sequences "$PPL_SEQS" --seq-len 1024 \
        --limit "$MC_LIMIT" --out "$RUN/eval.json" 2>&1 | tail -20
else
    echo "eval.json present, skipping"
fi

# --------------------------------------------------------------------------- #
log "2/5 samples"
if [ ! -f "$RUN/samples.txt" ]; then
    : > "$RUN/samples.txt"
    for p in "日本の首都は" "富士山は" "人工知能とは" "夏目漱石の代表作は" "水の沸点は"; do
        python scripts/generate.py --ckpt "$CKPT" --tokenizer "$TOKENIZER" \
            --prompt "$p" --max-new-tokens 60 --seed 0 --device "$DEVICE" \
            --mode hiergen 2>/dev/null | tail -n +2 | tail -3 \
            >> "$RUN/samples.txt" || true
        echo "" >> "$RUN/samples.txt"
    done
    cat "$RUN/samples.txt"
else
    echo "samples.txt present, skipping"
fi

# --------------------------------------------------------------------------- #
log "3/5 HierGen vs RecGen"
python scripts/benchmark_generation.py \
    --config "$RUN/model_config.json" --ckpt "$CKPT" \
    --prompt-len 1024 --new-tokens 128 --repeats 2 --warmup 1 \
    --device "$DEVICE" --check-agreement \
    --json-out "$RUN/genbench.json" 2>&1 | tail -22 || true

# --------------------------------------------------------------------------- #
log "4/5 export"
python scripts/export_checkpoint.py "$CKPT" --out "$OUT" --tokenizer "$TOKENIZER"
python scripts/make_model_card.py --export "$OUT" --run "$RUN" --name "$NAME" \
    --hardware "${HARDWARE:-1x NVIDIA H100 SXM 80GB (vast.ai)}" \
    --eval-json "$RUN/eval.json" --samples-file "$RUN/samples.txt" \
    --data-desc "${DATA_DESC:-Japanese-majority mixture: fineweb-2-edu-japanese, \
FineWeb-2 ja, Japanese Wikipedia, Zyda-2, FinePDFs-Edu, FineWeb-Edu and \
SwallowMath-v2, tokenised with llm-jp-tokenizer v3.  The last 15% of training \
switches to a high-quality math / code / Japanese-edu blend (Nemotron-3, OLMo 3 \
and llm-jp-4 all do this).}"
cp -f "$RUN/log.jsonl" "$OUT/training_log.jsonl" 2>/dev/null || true
cp -f "$RUN/eval.json" "$RUN/genbench.json" "$OUT/" 2>/dev/null || true
ls -la "$OUT"

# --------------------------------------------------------------------------- #
log "5/5 push"
if [ -n "$REPO" ]; then
    python - "$OUT" "$REPO" "$NAME" <<'PY'
import sys
from huggingface_hub import HfApi
folder, repo, name = sys.argv[1:4]
api = HfApi()                      # reads HF_HOME/token
print("as:", api.whoami()["name"])
api.create_repo(repo, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=folder, repo_id=repo,
                  commit_message=f"{name}: trained checkpoint, eval and logs")
print("pushed https://huggingface.co/" + repo)
PY
else
    echo "REPO unset -- skipping upload"
fi

log "done"
