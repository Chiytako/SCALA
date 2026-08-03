#!/usr/bin/env bash
# Mirrors the newest SCALA checkpoint off the (rented, disposable) instance,
# optionally pushing to the HF Hub or S3. Usage: HF_REPO=... bash vastai/sync.sh [WATCH=seconds]
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/SCALA}"
RUN_DIR="${RUN_DIR:-/workspace/runs/scala-8b-a1b}"
EXPORT_DIR="${EXPORT_DIR:-/data/export}"
HF_REPO="${HF_REPO:-}"
S3_URI="${S3_URI:-}"
WATCH="${WATCH:-0}"
KEEP_LOCAL="${KEEP_LOCAL:-2}"

cd "$WORKDIR"

sync_once() {
    local latest
    latest=$(ls -d "$RUN_DIR"/step-* 2>/dev/null | sort | tail -1 || true)
    [ -z "$latest" ] && { echo "no checkpoint yet in $RUN_DIR"; return 0; }

    local step name out
    step=$(basename "$latest")
    name="scala-8b-a1b-$step"
    out="$EXPORT_DIR/$name"

    if [ -f "$out/.synced" ]; then
        echo "$step already synced"
        return 0
    fi

    echo "==> exporting $latest"
    mkdir -p "$out"
    python scripts/export_checkpoint.py "$latest" --out "$out"

    # the training log is small and the most useful thing to keep
    cp -f "$RUN_DIR/log.jsonl" "$out/log.jsonl" 2>/dev/null || true

    if [ -n "$HF_REPO" ]; then
        echo "==> pushing to https://huggingface.co/$HF_REPO (revision $step)"
        python - "$out" "$HF_REPO" "$step" <<'PY'
import sys
from huggingface_hub import HfApi
folder, repo, step = sys.argv[1:4]
api = HfApi()
api.create_repo(repo, exist_ok=True)
try:
    api.create_branch(repo, branch=step, exist_ok=True)
except Exception:
    pass
api.upload_folder(folder_path=folder, repo_id=repo, revision=step,
                  commit_message=f"checkpoint {step}")
print("uploaded")
PY
    fi

    if [ -n "$S3_URI" ]; then
        echo "==> pushing to $S3_URI/$name"
        if command -v rclone >/dev/null; then
            rclone copy "$out" "$S3_URI/$name" --progress
        else
            aws s3 sync "$out" "$S3_URI/$name"
        fi
    fi

    touch "$out/.synced"

    # keep the box from filling up
    ls -d "$EXPORT_DIR"/scala-8b-a1b-step-* 2>/dev/null | sort | \
        head -n -"$KEEP_LOCAL" | xargs -r rm -rf
    echo "==> $step synced"
}

if [ "$WATCH" -gt 0 ]; then
    echo "watching $RUN_DIR every ${WATCH}s (Ctrl-C to stop)"
    while true; do
        sync_once || echo "sync failed, will retry"
        sleep "$WATCH"
    done
else
    sync_once
fi
