#!/usr/bin/env bash
# Batch LoRA SFT of nvidia/Cosmos3-Nano with the TAO cosmos-rl container on Pre-Tyche: 1 node x 4 GPUs.
#
#   cd $LUSTRE_DIR/cosmos3tao/repo          # your git checkout
#   sbatch cluster/ptyche_train.sbatch
#   squeue -u $USER
#   tail -f $LUSTRE_DIR/cosmos3tao/logs/general_sa-finetune:tao-<jobid>.out
#
# Run cluster/ptyche_setup.sh once first (dataset, model, sqsh on Lustre).
#SBATCH --account=general_sa
#SBATCH --partition=36x2-a01r
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
# NOTE: Pre-Tyche has no "gpu" GRES; nodes come whole. GPUS below limits the job to N devices.
#SBATCH --job-name=general_sa-finetune:tao
#SBATCH --no-requeue
#SBATCH --output=/lustre/fsw/general_sa/%u/cosmos3tao/logs/%x-%j.out
#SBATCH --error=/lustre/fsw/general_sa/%u/cosmos3tao/logs/%x-%j.err
set -euo pipefail

ACCOUNT="${ACCOUNT:-${SLURM_JOB_ACCOUNT:-general_sa}}"
LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/$ACCOUNT/$USER}"
WORK="${WORK:-$LUSTRE_DIR/cosmos3tao}"
IMAGE="${TAO_IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-rl}"
SQSH="${TAO_SQSH:-$WORK/sqsh/$(echo "$IMAGE" | tr '/:' '__').sqsh}"
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
export GPUS="${GPUS:-4}" EPOCHS="${EPOCHS:-5}" RUN="cosmos3_nano_tube_lora_${SLURM_JOB_ID}"
mkdir -p "$WORK/logs" "$WORK/results"

[ -f "$SQSH" ] && CONTAINER="$SQSH" || CONTAINER="${IMAGE/\//#}"
# Never mount over /workspace: the image keeps the cosmos_rl package there.
MOUNTS="$WORK:/tao-workspace,$REPO_DIR:/tao-repo:ro"

echo "== job $SLURM_JOB_ID node=$(hostname) image=$CONTAINER gpus=$GPUS run=$RUN"
srun --ntasks=1 --mpi=pmix \
     --container-image="$CONTAINER" \
     --container-mounts="$MOUNTS" \
     --container-mount-home \
     --no-container-remap-root \
     --container-env=GPUS,EPOCHS,RUN,TAO_TRAIN_CMD,HF_TOKEN,WANDB_MODE \
     bash /tao-repo/cluster/train_in_container.sh
