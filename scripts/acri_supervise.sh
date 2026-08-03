#!/usr/bin/env bash
# Restarts an ACRi training run if it dies, until `final` appears or
# MAX_RESTARTS consecutive failures with no progress. Usage: scripts/acri_supervise.sh [config] [logfile]
set -uo pipefail

CFG="${1:-configs/train_small_v4_acri.yaml}"
LOG="${2:-/scratch/$USER/train_v4_acri.log}"
REPO="/scratch/$USER/SCALA"
PY="/scratch/$USER/unsloth-ft/.venv/bin/python"
MAX_RESTARTS=8

cd "$REPO"
# Use the venv interpreter: the system python3 lacks PyYAML, and a blank
# OUT_DIR would make every later check silently wrong.
OUT_DIR=$("$PY" - "$CFG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["train"]["out_dir"])
PY
)
if [ -z "$OUT_DIR" ] || [ ! -d "$OUT_DIR" ]; then
    echo "could not read out_dir from $CFG (got '$OUT_DIR'); refusing to supervise blind"
    exit 1
fi
echo "$(date +%T) supervising $OUT_DIR (config $CFG)"

export HF_HOME="/scratch/$USER/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Wait for ROCm to be ready before launching: right after boot,
# torch.cuda.is_available() can return False and training silently falls back to CPU.
wait_for_gpu() {
    for _ in $(seq 1 60); do
        if "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
                2>/dev/null; then
            return 0
        fi
        echo "$(date +%T) GPU not ready yet; waiting"
        sleep 30
    done
    echo "$(date +%T) GPU never became available; refusing to train on CPU"
    return 1
}

fails=0
last_tokens=-1
while true; do
    if [ -d "$OUT_DIR/final" ]; then
        echo "$(date +%T) run finished (final/ exists); supervisor exiting"
        exit 0
    fi

    if pgrep -f "train\.py --config $CFG" > /dev/null; then
        sleep 120
        continue
    fi

    # Progress since the previous restart tells a crash-loop apart from a
    # run that is advancing but keeps hitting something transient.
    tokens=$(tail -1 "$OUT_DIR/log.jsonl" 2>/dev/null \
             | python3 -c 'import sys,json; print(json.loads(sys.stdin.read() or "{}").get("tokens",0))' 2>/dev/null || echo 0)
    if [ "$tokens" -le "$last_tokens" ]; then
        fails=$((fails + 1))
    else
        fails=0
    fi
    last_tokens=$tokens

    if [ "$fails" -ge "$MAX_RESTARTS" ]; then
        echo "$(date +%T) $fails restarts with no progress past $tokens tokens;"
        echo "  giving up rather than looping.  Last 40 lines of $LOG:"
        tail -40 "$LOG"
        exit 1
    fi

    if ! wait_for_gpu; then
        exit 1
    fi
    echo "$(date +%T) training not running at $tokens tokens; restarting (#$((fails + 1)))"
    setsid nohup "$PY" scripts/train.py --config "$CFG" >> "$LOG" 2>&1 < /dev/null &
    sleep 180        # let it get past model build + compile before judging it
done
