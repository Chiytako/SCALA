#!/usr/bin/env bash
# vast.ai provisioning for a SCALA training instance. Idempotent: safe to re-run.
# Expects a CUDA-capable PyTorch base image already installed on the instance.
set -euo pipefail

REPO_URL="${REPO_URL:-}"                    # optional: git clone source
WORKDIR="${WORKDIR:-/workspace/SCALA}"
DATA_DIR="${DATA_DIR:-/data/tokens}"
VENV="${VENV:-/workspace/venv}"

log() { echo -e "\n\033[1;36m==> $*\033[0m"; }

# --------------------------------------------------------------------------- #
log "System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    git git-lfs curl wget tmux htop nvtop jq rsync build-essential \
    python3-dev python3-venv pigz aria2 >/dev/null
git lfs install --skip-repo || true

# --------------------------------------------------------------------------- #
log "Python environment"
if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -qU pip wheel setuptools

# Keep whatever torch the base image ships (it is matched to the driver).
python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"devices {torch.cuda.device_count()}")
assert torch.cuda.is_available(), "no CUDA device visible"
PY

# --------------------------------------------------------------------------- #
log "Project"
if [ -n "$REPO_URL" ] && [ ! -d "$WORKDIR/.git" ]; then
    git clone "$REPO_URL" "$WORKDIR"
fi
mkdir -p "$WORKDIR" "$DATA_DIR"
cd "$WORKDIR"
python -m pip install -q -r requirements.txt

# --------------------------------------------------------------------------- #
log "Caches on the big volume"
export HF_HOME="${HF_HOME:-/data/hf}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRITON_CACHE_DIR="/data/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="/data/inductor-cache"
mkdir -p "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cat >> ~/.bashrc <<EOF
source $VENV/bin/activate
export HF_HOME=$HF_HOME
export HF_DATASETS_CACHE=$HF_DATASETS_CACHE
export TRITON_CACHE_DIR=$TRITON_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
cd $WORKDIR
EOF

# --------------------------------------------------------------------------- #
log "NCCL / network tuning"
# vast.ai instances are usually single-node; these help when they are not.
cat > /etc/nccl.conf <<'EOF'
NCCL_IB_DISABLE=0
NCCL_P2P_DISABLE=0
NCCL_SOCKET_IFNAME=^lo,docker
NCCL_ASYNC_ERROR_HANDLING=1
EOF

# --------------------------------------------------------------------------- #
log "Sanity checks"
cd "$WORKDIR"
python scripts/count_params.py configs/base_8b_a1b.yaml
python -m pytest tests -q -x

log "Provisioning complete."
cat <<EOF

Next:
  1. tokenize the corpus (hours; run under tmux):
       python scripts/prepare_data.py --config configs/data_ja_mix.yaml \\
           --out $DATA_DIR --budget-tokens 120e9 --workers 32
  2. start training:
       bash vastai/launch.sh
  3. watch:
       tail -f /workspace/runs/scala-8b-a1b/log.jsonl | jq -c '{step,loss,ppl}'
EOF
