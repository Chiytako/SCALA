#!/usr/bin/env bash
# Starts (or resumes) SCALA pretraining on a vast.ai instance, in tmux so an
# SSH drop won't kill it. Usage: bash vastai/launch.sh (NPROC=n, CONFIG=path)
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/SCALA}"
CONFIG="${CONFIG:-configs/train_8b_a1b.yaml}"
SESSION="${SESSION:-photon}"
NPROC="${NPROC:-$(nvidia-smi --list-gpus | wc -l)}"
MASTER_PORT="${MASTER_PORT:-29500}"

cd "$WORKDIR"

# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR before calling.
NNODES="${NNODES:-1}"
if [ "$NNODES" -gt 1 ]; then
    RDZV=(--nnodes="$NNODES" --node_rank="${NODE_RANK:?set NODE_RANK}"
          --master_addr="${MASTER_ADDR:?set MASTER_ADDR}"
          --master_port="$MASTER_PORT")
else
    RDZV=(--standalone)
fi

CMD="torchrun ${RDZV[*]} --nproc_per_node=$NPROC scripts/train.py --config $CONFIG"

echo "GPUs      : $NPROC"
echo "nodes     : $NNODES"
echo "config    : $CONFIG"
echo "command   : $CMD"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" \
    "source ${VENV:-/workspace/venv}/bin/activate; $CMD 2>&1 | tee -a train.out"
tmux new-window -t "$SESSION" -n gpu 'nvtop || watch -n2 nvidia-smi'

echo
echo "started in tmux session '$SESSION'"
echo "  attach : tmux attach -t $SESSION"
echo "  detach : Ctrl-b d"
echo "  logs   : tail -f $WORKDIR/train.out"
