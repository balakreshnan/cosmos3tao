#!/usr/bin/env bash
# Interactive 4-GPU shell INSIDE the TAO cosmos-rl container on Pre-Tyche (same flags you use today,
# but with the TAO image and the Lustre/repo mounts). Once the prompt appears run:
#
#     bash /tao-repo/cluster/train_in_container.sh
#
# Usage (on login-ptyche, from the repo checkout):  bash cluster/ptyche_interactive.sh
set -euo pipefail
ACCOUNT="${ACCOUNT:-general_sa}"
LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/$ACCOUNT/$USER}"
WORK="${WORK:-$LUSTRE_DIR/cosmos3tao}"
IMAGE="${TAO_IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-rl}"
ARCH="${ARCH:-$(uname -m)}"
SQSH="${TAO_SQSH:-$WORK/sqsh/$(echo "$IMAGE" | tr '/:' '__')-$ARCH.sqsh}"
[ -f "$SQSH" ] || SQSH="$WORK/sqsh/$(echo "$IMAGE" | tr '/:' '__').sqsh"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GPUS="${GPUS:-4}"
[ -f "$SQSH" ] && CONTAINER="$SQSH" || CONTAINER="${IMAGE/\//#}"

echo "container=$CONTAINER  mounts=$WORK:/tao-workspace,$REPO_DIR:/tao-repo  (GPUS=$GPUS, exported)"
export GPUS
exec srun --account="$ACCOUNT" \
     --partition="${PARTITION:-36x2-a01r}" \
     --nodes=1 --ntasks-per-node=1 \
     --time=5:00:00 \
     --job-name=general_sa-finetune:tao \
     --container-image="$CONTAINER" \
     --container-mounts="$WORK:/tao-workspace,$REPO_DIR:/tao-repo" \
     --container-mount-home \
     --no-container-remap-root \
     --mpi=pmix \
     --pty bash
