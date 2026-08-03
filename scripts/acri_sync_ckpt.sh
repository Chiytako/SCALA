#!/usr/bin/env bash
# Mirrors the newest checkpoint of an ACRi run from /scratch (server-local,
# reset between reservations) to the shared NFS home. Usage: scripts/acri_sync_ckpt.sh [run_dir] [dest_root]
set -euo pipefail

RUN="${1:-/scratch/$USER/runs/scala-small-v4-acri}"
DEST_ROOT="${2:-$HOME/photonjp-ckpt}"
DEST="$DEST_ROOT/$(basename "$RUN")"

mkdir -p "$DEST"

# Config and log make an orphaned checkpoint readable; curve.jsonl carries the
# held-out BPC curve, which is not reproducible by re-running (the weights are).
for f in model_config.json train_config.json log.jsonl curve.jsonl \
         curve_llmjp_wiki_slice.jsonl; do
    [ -f "$RUN/$f" ] && cp -f "$RUN/$f" "$DEST/"
done

# `final` is checked first rather than relying on sort order: "final" sorts
# alphabetically before "step-*", so naive sorting would pick the wrong checkpoint.
if [ -d "$RUN/final" ]; then
    latest="$RUN/final"
else
    latest=$(ls -d "$RUN"/step-* 2>/dev/null | sort -V | tail -1 || true)
fi
if [ -z "$latest" ]; then
    echo "no checkpoint under $RUN yet"
    exit 0
fi

name=$(basename "$latest")
if [ -d "$DEST/$name" ]; then
    echo "$name already mirrored"
else
    avail=$(df -Pk "$DEST" | awk 'NR==2{print $4}')
    need=$(du -sk "$latest" | awk '{print $1}')
    if [ "$avail" -lt $((need + 2097152)) ]; then   # keep 2 GB of headroom
        echo "refusing to copy $name: ${need}K needed, only ${avail}K free on the"
        echo "shared home pool.  Free space there or pass another destination."
        exit 1
    fi
    cp -r "$latest" "$DEST/$name.partial"
    mv "$DEST/$name.partial" "$DEST/$name"   # never leave a half-copy that
    echo "mirrored $name -> $DEST/$name"     # looks resumable
fi

# Prune: keep the two newest step-* plus final.
ls -d "$DEST"/step-* 2>/dev/null | head -n -2 | while read -r old; do
    echo "pruning $(basename "$old")"
    rm -rf "$old"
done

df -h "$DEST" | tail -1
