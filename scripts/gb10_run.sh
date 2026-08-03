#!/usr/bin/env bash
# Runs a command inside the NGC PyTorch container on GB10 (unified memory: hard
# --memory limit here plus max_memory_fraction in the trainer). Usage: scripts/gb10_run.sh "<command>"
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
WORKDIR="${WORKDIR:-$HOME/SCALA}"
NAME="${NAME:-photon}"
MEM_LIMIT="${MEM_LIMIT:-96g}"      # of 121 GiB total -- leaves the OS 25 GiB
SHM="${SHM:-16g}"
DETACH="${DETACH:-0}"
CMD="${1:?usage: gb10_run.sh \"<command>\"}"

# Refuse if NAME is already running: callers that don't set NAME share the
# default, so removing a running container here would kill another launch mid-run.
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "gb10_run.sh: container '$NAME' is already running." >&2
    echo "  Set NAME=<something-else> to run alongside it, or" >&2
    echo "  docker rm -f $NAME   to take it down deliberately." >&2
    exit 1
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true

ARGS=(
  --gpus all
  --name "$NAME"
  --memory "$MEM_LIMIT"
  --shm-size "$SHM"
  --ulimit memlock=-1 --ulimit stack=67108864
  -v "$WORKDIR":/work
  -w /work
  -e PYTHONUNBUFFERED=1
  # extra pip installs live on the mounted volume so they survive the container
  -e PYTHONPATH=/work/.pylibs
  -e HF_HOME=/work/.hf
  -e TOKENIZERS_PARALLELISM=true
  -e OMP_NUM_THREADS=8
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
[ "$DETACH" = "1" ] && ARGS+=(-d) || ARGS+=(--rm -i)

exec docker run "${ARGS[@]}" "$IMAGE" bash -lc "$CMD"
