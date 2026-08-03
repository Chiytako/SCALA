#!/usr/bin/env bash
# Takes a fresh vast.ai H100 instance from bare image to a running SCALA
# 8B-A1B v2 job (deps, sanity, tokenize, train). Usage: HF_TOKEN=... bash vastai/run_8b_v2.sh
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/SCALA}"
DATA_DIR="${DATA_DIR:-/data/tokens}"
CONFIG="${CONFIG:-configs/train_8b_a1b_v2_vast.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data_ja_v2.yaml}"
SESSION="${SESSION:-photon8b}"
# Tokenise ~15% more than the training budget so the sampler never has to wrap
# a source mid-run.
TOKEN_BUDGET="${TOKEN_BUDGET:-6.5e9}"
SKIP_TESTS="${SKIP_TESTS:-0}"

log() { echo -e "\n\033[1;36m==> $*\033[0m"; }
cd "$WORKDIR"

# --------------------------------------------------------------------------- #
log "Stage 1: dependencies"
export DEBIAN_FRONTEND=noninteractive
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux jq pigz >/dev/null)
python -m pip install -q -U pip
# Pin numpy to the container's version: an ABI mismatch breaks torch.from_numpy
# only at the first training batch, after expensive setup has already run.
NUMPY_PIN="numpy==$(python -c 'import numpy; print(numpy.__version__)')"
python -m pip install -q "$NUMPY_PIN" datasets transformers zstandard pyyaml \
    safetensors huggingface_hub pytest
python -c "import numpy, torch; torch.from_numpy(numpy.zeros(3)); print('numpy/torch ABI ok:', numpy.__version__)"

export HF_HOME="${HF_HOME:-/data/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=true
export TORCHINDUCTOR_CACHE_DIR=/data/inductor-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_DATASETS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$DATA_DIR"

if [ -n "${HF_TOKEN:-}" ]; then
    mkdir -p "$HF_HOME"
    printf '%s' "$HF_TOKEN" > "$HF_HOME/token"
    chmod 600 "$HF_HOME/token"
    echo "HF token installed"
fi

python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}  {p.name}  sm_{p.major}{p.minor}  "
      f"{p.total_memory/2**30:.0f} GiB")
# the fused MoE GEMM is gated on compute capability; H100 (9.0) has it,
# consumer Blackwell and GB10 do not and fall back to the padded batched GEMM
print("grouped_mm available:", p.major >= 9)
PY

# --------------------------------------------------------------------------- #
log "Stage 2: sanity"
python scripts/count_params.py "$(grep -oP 'model_config:\s*\K\S+' "$CONFIG")"
if [ "$SKIP_TESTS" != "1" ]; then
    python -m pytest tests -q
fi

if [ ! -f .memprobe_ok ]; then
    log "memory probe (finds the largest micro-batch that fits)"
    python scripts/probe_throughput.py \
        --config "$(grep -oP 'model_config:\s*\K\S+' "$CONFIG")" \
        --seq-len 4096 --batches 2 4 8 --steps 3 --warmup 1 \
        --json-out /data/probe.json || true
    touch .memprobe_ok
fi

# --------------------------------------------------------------------------- #
log "Stage 3: tokenise (budget $TOKEN_BUDGET)"
if [ ! -f "$DATA_DIR/manifest.json" ] || [ "${FORCE_PREP:-0}" = "1" ]; then
    python scripts/prepare_data.py --config "$DATA_CONFIG" --dry-run || {
        echo "one or more sources are unreachable -- fix before spending GPU time"
        exit 1
    }
    python scripts/prepare_data.py --config "$DATA_CONFIG" \
        --out "$DATA_DIR" --budget-tokens "$TOKEN_BUDGET" --workers 16
else
    echo "manifest already present, skipping (FORCE_PREP=1 to redo)"
fi
python - "$DATA_DIR" <<'PY'
import json, sys
m = json.load(open(f"{sys.argv[1]}/manifest.json"))
tot = 0
for s in m["sources"]:
    n = sum(x["n_tokens"] for x in s["shards"]); tot += n
    print(f"  {s['name']:<22}{n/1e9:8.2f}B  w={s['weight']}")
print(f"  {'TOTAL':<22}{tot/1e9:8.2f}B tokens")
PY

# --------------------------------------------------------------------------- #
log "Stage 4: train"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already running; attach with: tmux attach -t $SESSION"
    exit 0
fi
tmux new-session -d -s "$SESSION" \
    "cd $WORKDIR && python scripts/train.py --config $CONFIG 2>&1 | tee -a train8b.out"
echo
echo "training started."
echo "  attach : tmux attach -t $SESSION"
echo "  logs   : tail -f $WORKDIR/train8b.out"
echo "  metrics: tail -f $(grep -oP 'out_dir:\s*\K\S+' "$CONFIG")/log.jsonl | jq -c '{step,ppl,grad_norm}'"
