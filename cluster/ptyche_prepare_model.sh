#!/usr/bin/env bash
# Convert the downloaded nvidia/Cosmos3-Nano (cosmos3_omni) checkpoint into a Qwen3-VL HF checkpoint that the
# TAO cosmos-rl container can train. Runs the conversion as a short SLURM step inside the TAO container.
#
#   export ACCOUNT=general_sa PARTITION=gb300 LUSTRE_DIR=/lustre/fsw/general_sa/$USER
#   bash cluster/ptyche_prepare_model.sh
#
# Output: $LUSTRE_DIR/cosmos3tao/models/Cosmos3-Nano-qwen3vl   (train_in_container.sh picks it up automatically)
set -euo pipefail
ACCOUNT="${ACCOUNT:-general_sa}"
PARTITION="${PARTITION:-batch}"
LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/$ACCOUNT/$USER}"
WORK="${WORK:-$LUSTRE_DIR/cosmos3tao}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${TAO_IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-rl}"
SRC="$WORK/models/Cosmos3-Nano"
DONOR="$WORK/models/Qwen3-VL-8B-Instruct-donor"
DST="$WORK/models/Cosmos3-Nano-qwen3vl"

[ -f "$SRC/config.json" ] || { echo "FATAL: $SRC missing; run cluster/ptyche_setup.sh first"; exit 1; }

# 1. donor config files (tiny; login node has internet)
if [ ! -f "$DONOR/config.json" ]; then
  echo ">> downloading Qwen/Qwen3-VL-8B-Instruct config files -> $DONOR"
  # shellcheck disable=SC1091
  source "$WORK/venv/bin/activate"
  # filenames are positional for `hf download` (not --include)
  HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-VL-8B-Instruct \
    config.json generation_config.json model.safetensors.index.json --local-dir "$DONOR"
fi
for f in config.json model.safetensors.index.json; do
  [ -f "$DONOR/$f" ] || { echo "FATAL: donor file $DONOR/$f missing after download"; exit 1; }
done

# 2. pick the sqsh for this cluster's compute arch
ARCH="$(srun -A "$ACCOUNT" -p "$PARTITION" -N1 -n1 --time=00:03:00 --job-name="$ACCOUNT-cosmos3.arch-probe" uname -m 2>/dev/null | tail -1)"
SQSH="$WORK/sqsh/$(echo "$IMAGE" | tr '/:' '__')-${ARCH:-x86_64}.sqsh"
[ -f "$SQSH" ] && CONTAINER="$SQSH" || CONTAINER="${IMAGE/\//#}"
echo ">> arch=$ARCH container=$CONTAINER"

# 3. convert inside the container (torch + safetensors available), CPU only, ~17 GB written
echo ">> converting $SRC -> $DST"
srun -A "$ACCOUNT" -p "$PARTITION" -N1 -n1 --time=01:00:00 --job-name="$ACCOUNT-cosmos3.convert" \
  --container-image="$CONTAINER" --container-mounts="$WORK:/tao-workspace,$REPO_DIR:/tao-repo:ro" \
  --no-container-remap-root \
  bash -c 'set -e; PY=/opt/venv/cosmos_rl/bin/python; [ -x "$PY" ] || PY=python
    echo "-- using $PY ($($PY --version 2>&1))"
    $PY -c "import torch, safetensors; print(\"torch\", torch.__version__, \"safetensors\", safetensors.__version__)"
    $PY /tao-repo/cluster/convert_omni_to_qwen3vl.py --src /tao-workspace/models/Cosmos3-Nano \
      --donor /tao-workspace/models/Qwen3-VL-8B-Instruct-donor --dst /tao-workspace/models/Cosmos3-Nano-qwen3vl
    echo "-- verifying load with transformers AutoConfig:"
    $PY -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained(\"/tao-workspace/models/Cosmos3-Nano-qwen3vl\"); print(c.model_type, c.architectures)"'

echo
echo "Prepared model: $DST"
ls -la "$DST" | head -20
echo "Now submit training; train_in_container.sh uses Cosmos3-Nano-qwen3vl automatically."
