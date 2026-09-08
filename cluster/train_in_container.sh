#!/usr/bin/env bash
# Runs INSIDE the TAO cosmos-rl container (launched by ptyche_train.sbatch or ptyche_interactive.sh).
# Renders the spec, runs a preflight, then launches LoRA SFT of Cosmos3-Nano on the visible GPUs.
#
# Expects (all container paths):
#   /tao-workspace  -> $LUSTRE_DIR/cosmos3tao  (data/, models/, results/)
#   /tao-repo       -> this git checkout
# Env: GPUS (default: visible GPU count), EPOCHS (5), RUN (name), TAO_TRAIN_CMD (override launcher)
set -euo pipefail

C_WORK=/tao-workspace
REPO=/tao-repo
GPUS="${GPUS:-$(nvidia-smi -L | wc -l)}"
EPOCHS="${EPOCHS:-5}"
RUN="${RUN:-cosmos3_nano_tube_lora_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
RESULTS="$C_WORK/results/$RUN"
MODEL="$C_WORK/models/Cosmos3-Nano"
TRAIN_ROOT="$C_WORK/data/tube_inspection/train"
VAL_ROOT="$C_WORK/data/tube_inspection/val"
mkdir -p "$RESULTS"

echo "== run=$RUN gpus=$GPUS epochs=$EPOCHS"
echo "== host=$(hostname) python=$(python --version 2>&1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ---- preflight
echo "-- entrypoints on PATH:"
for e in cosmos_rl cosmos-rl tao; do
  if command -v "$e" >/dev/null 2>&1; then echo "   $e -> $(command -v $e)"; else echo "   ($e not found)"; fi
done
[ -f "$MODEL/config.json" ] || { echo "FATAL: model not found at $MODEL (run cluster/ptyche_setup.sh)"; exit 2; }
echo "-- model_type: $(python -c "import json;print(json.load(open('$MODEL/config.json')).get('model_type'))")"
[ -f "$TRAIN_ROOT/annotations.json" ] || { echo "FATAL: dataset not found at $TRAIN_ROOT"; exit 2; }
python -c "import json;print('-- train samples:',len(json.load(open('$TRAIN_ROOT/annotations.json'))))"
python -c "import json;print('-- val samples:',len(json.load(open('$VAL_ROOT/annotations.json'))))"

# ---- render spec (PyYAML ships in the TAO image)
python "$REPO/cluster/render_spec.py" \
  --template "$REPO/specs/cosmos3_nano_lora_sft.yaml" --out "$RESULTS/train_spec.yaml" \
  --model "$MODEL" --train-root "$TRAIN_ROOT" --val-root "$VAL_ROOT" \
  --results "$RESULTS" --gpus "$GPUS" --epochs "$EPOCHS" --experiment-name "$RUN"

# ---- offline HF so compute nodes never reach the internet; caches on Lustre
export HF_HOME="${HF_HOME:-$C_WORK/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NUM_GPU_PER_NODE="$GPUS" WORLD_SIZE=1 NODE_RANK=0 MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" MASTER_PORT="${MASTER_PORT:-29500}"

# ---- train. TAO container convention: "<network> <task> -e <spec> -r <results_dir>".
TRAIN_CMD="${TAO_TRAIN_CMD:-cosmos_rl train -e $RESULTS/train_spec.yaml -r $RESULTS}"
echo "== launching: $TRAIN_CMD"
cd "$C_WORK"
set +e
eval "$TRAIN_CMD" 2>&1 | tee "$RESULTS/train.log"
rc=${PIPESTATUS[0]}
set -e
echo "== training exit code: $rc"
[ -f "$RESULTS/status.json" ] && { echo "== status.json (tail):"; tail -c 2000 "$RESULTS/status.json"; echo; }
echo "== results: $RESULTS"; ls -la "$RESULTS" | head -40
exit "$rc"
