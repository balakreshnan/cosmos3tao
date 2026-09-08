#!/usr/bin/env bash
# One-time setup on the Pre-Tyche (ptyche) SLURM login node.
#   1. creates a Python venv on Lustre (dataset generation + HF download helpers only; NO training here)
#   2. generates the synthetic tube-inspection dataset on Lustre
#   3. downloads nvidia/Cosmos3-Nano to Lustre so compute nodes never need internet
#   4. pre-imports the TAO cosmos-rl container to a .sqsh so the GPU job starts instantly
#
# Usage (on login-ptyche):
#   export ACCOUNT=general_sa                # your SLURM account
#   export HF_TOKEN=hf_xxx                   # optional, only if the model repo is gated for you
#   bash cluster/ptyche_setup.sh
set -euo pipefail

ACCOUNT="${ACCOUNT:-general_sa}"
LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/$ACCOUNT/$USER}"
WORK="${WORK:-$LUSTRE_DIR/cosmos3tao}"
IMAGE="${TAO_IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-rl}"
MODEL_ID="${MODEL_ID:-nvidia/Cosmos3-Nano}"
N_TRAIN="${N_TRAIN:-800}"
N_VAL="${N_VAL:-200}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$WORK"/{data,models,results,sqsh,logs}
echo ">> work dir: $WORK"

# ---------------------------------------------------------------- 1. venv on Lustre
if [ ! -x "$WORK/venv/bin/python" ]; then
  echo ">> creating venv ($(python3 --version))"
  python3 -m venv "$WORK/venv"
fi
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
pip install -q --upgrade pip
pip install -q pillow numpy pyyaml huggingface_hub tqdm

# ---------------------------------------------------------------- 2. dataset
if [ ! -f "$WORK/data/tube_inspection/train/annotations.json" ]; then
  echo ">> generating dataset ($N_TRAIN train / $N_VAL val images)"
  python "$REPO_DIR/dataset/generate_tube_dataset.py" --out "$WORK/data/tube_inspection" --train "$N_TRAIN" --val "$N_VAL"
else
  echo ">> dataset already present, skipping"
fi

# ---------------------------------------------------------------- 3. model
MODEL_DIR="$WORK/models/$(basename "$MODEL_ID")"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo ">> downloading $MODEL_ID to $MODEL_DIR (~33 GB, weights only) inside a SLURM allocation"
  # Login nodes OOM-kill large downloads (Xet backend is memory hungry), so run it as a short job.
  # The download is resumable; re-run this script if it is interrupted.
  export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME="$WORK/hf_cache"
  srun -A "$ACCOUNT" -p "${PARTITION:-batch}" -N1 -n1 --cpus-per-task=16 --mem=64G --time=02:00:00 \
    --job-name="$ACCOUNT-cosmos3.hf-download" --export=ALL \
    "$WORK/venv/bin/hf" download "$MODEL_ID" --local-dir "$MODEL_DIR" \
      --exclude 'assets/*' --exclude 'images/*' --max-workers 8
else
  echo ">> model already present, skipping"
fi

# ---------------------------------------------------------------- 4. container -> sqsh
# enroot pulls the image for the CPU arch of the COMPUTE node doing the import (x86_64 on ptyche,
# aarch64 on GB200/GB300 nodes as on hecate/lyris). Login nodes may differ from compute nodes, so ask
# the partition for its arch and key the sqsh file on that.
ARCH="${ARCH:-$(srun -A "$ACCOUNT" -p "${PARTITION:-batch}" -N1 -n1 --time=00:03:00 \
        --job-name="$ACCOUNT-cosmos3.arch-probe" uname -m 2>/dev/null | tail -1)}"
ARCH="${ARCH:-$(uname -m)}"
echo ">> compute node arch: $ARCH"
SQSH="$WORK/sqsh/$(echo "$IMAGE" | tr '/:' '__')-$ARCH.sqsh"
if [ ! -f "$SQSH" ]; then
  echo ">> importing $IMAGE -> $SQSH (uses ~/.config/enroot/.credentials for nvcr.io)"
  srun -A "$ACCOUNT" -p "${PARTITION:-batch}" -N1 -n1 --cpus-per-task=4 --time=01:00:00 --job-name="$ACCOUNT-cosmos3.enroot-import" \
    enroot import -o "$SQSH" "docker://${IMAGE/\//#}"
else
  echo ">> sqsh already present, skipping"
fi

cat <<EOF

Setup complete.
  dataset : $WORK/data/tube_inspection/{train,val}
  model   : $MODEL_DIR
  sqsh    : $SQSH

Submit the 4-GPU LoRA fine-tune with:
  sbatch --account=$ACCOUNT --partition=${PARTITION:-batch} --export=ALL,ACCOUNT=$ACCOUNT,LUSTRE_DIR=$LUSTRE_DIR \
         --output=$WORK/logs/%x-%j.out --error=$WORK/logs/%x-%j.err cluster/ptyche_train.sbatch
EOF
