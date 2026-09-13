#!/usr/bin/env bash
# Evaluate task accuracy of the fine-tuned adapter AND the zero-shot base model on the validation split,
# in parallel on two GPUs of one node, inside the TAO cosmos-rl container.
#
#   export ACCOUNT=general_sa PARTITION=gb200 LUSTRE_DIR=/lustre/fsw/general_sa/$USER
#   RUN=cosmos3_nano_tube_lora_3044273 sbatch --account=$ACCOUNT --partition=$PARTITION \
#       --export=ALL,ACCOUNT=$ACCOUNT,LUSTRE_DIR=$LUSTRE_DIR,RUN=$RUN,LIMIT=300 \
#       --output=$LUSTRE_DIR/cosmos3tao/logs/%x-%j.out --error=$LUSTRE_DIR/cosmos3tao/logs/%x-%j.err cluster/ptyche_eval.sbatch
#
# Outputs: $LUSTRE_DIR/cosmos3tao/results/$RUN/eval_finetuned/{metrics.json,predictions.jsonl}
#          $LUSTRE_DIR/cosmos3tao/results/$RUN/eval_baseline/{metrics.json,predictions.jsonl}
#SBATCH --account=general_sa
#SBATCH --partition=gb200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --job-name=general_sa-cosmos3.tao-eval
#SBATCH --no-requeue
set -euo pipefail
ACCOUNT="${ACCOUNT:-${SLURM_JOB_ACCOUNT:-general_sa}}"
LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/$ACCOUNT/$USER}"
WORK="${WORK:-$LUSTRE_DIR/cosmos3tao}"
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
IMAGE="${TAO_IMAGE:-nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-rl}"
ARCH="${ARCH:-$(uname -m)}"
SQSH="${TAO_SQSH:-$WORK/sqsh/$(echo "$IMAGE" | tr '/:' '__')-$ARCH.sqsh}"
[ -f "$SQSH" ] && CONTAINER="$SQSH" || CONTAINER="${IMAGE/\//#}"
# RUN = results folder name; defaults to the most recent training run that produced an output/ dir
if [ -z "${RUN:-}" ]; then
  RUN=$(ls -dt "$WORK"/results/cosmos3_nano_tube_lora_*/output 2>/dev/null | head -1 | xargs -r dirname | xargs -r basename)
fi
[ -n "${RUN:-}" ] && [ -d "$WORK/results/$RUN" ] || { echo "FATAL: no training run found under $WORK/results; set RUN=<folder>"; exit 1; }
export RUN LIMIT="${LIMIT:-300}" MAXTOK="${MAXTOK:-768}"

echo "== eval job $SLURM_JOB_ID node=$(hostname) run=$RUN limit=$LIMIT image=$CONTAINER"
srun --ntasks=1 --mpi=pmix --container-image="$CONTAINER" \
     --container-mounts="$WORK:/tao-workspace,$REPO_DIR:/tao-repo:ro" --container-mount-home --no-container-remap-root \
     --container-env=RUN,LIMIT,MAXTOK \
     bash -c '
set -euo pipefail
source /opt/venv/cosmos_rl/bin/activate
export HF_HOME=/tao-workspace/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
R=/tao-workspace/results/$RUN
MODEL=/tao-workspace/models/Cosmos3-Nano-qwen3vl
VAL=/tao-workspace/data/tube_inspection/val
# pick the adapter export: prefer output/best/*, else highest epoch under output/*/safetensors/
ADAPTER=$(ls -d $R/output/best/*safetensors* 2>/dev/null | head -1 || true)
[ -n "$ADAPTER" ] || ADAPTER=$(ls -d $R/output/*/safetensors/epoch_* 2>/dev/null | sort -t_ -k2 -n | tail -1)
echo "-- adapter: $ADAPTER"; ls -la "$ADAPTER" | head; [ -f "$ADAPTER/adapter_config.json" ] && cat "$ADAPTER/adapter_config.json" || true
python -c "import transformers,torch;print(\"transformers\",transformers.__version__,\"torch\",torch.__version__)"; python -c "import peft;print(\"peft\",peft.__version__)" 2>/dev/null || echo "peft not installed (manual LoRA merge will be used)"
CUDA_VISIBLE_DEVICES=0 python /tao-repo/cluster/eval_in_container.py --model $MODEL --adapter "$ADAPTER" --annotations $VAL/annotations.json --media $VAL --limit $LIMIT --max-new-tokens $MAXTOK --out $R/eval_finetuned > $R/eval_finetuned.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python /tao-repo/cluster/eval_in_container.py --model $MODEL --annotations $VAL/annotations.json --media $VAL --limit $LIMIT --max-new-tokens $MAXTOK --out $R/eval_baseline > $R/eval_baseline.log 2>&1 &
P2=$!
set +e
wait $P1; rc1=$?; wait $P2; rc2=$?
echo "== finetuned exit $rc1, baseline exit $rc2"
echo "== FINETUNED"; tail -n 40 $R/eval_finetuned.log
echo "== BASELINE";  tail -n 40 $R/eval_baseline.log
exit $(( rc1 | rc2 ))'
