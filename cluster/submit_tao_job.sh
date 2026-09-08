#!/usr/bin/env bash
# End-to-end remote fine-tune of nvidia/Cosmos3-Nano via TAO Fine-Tuning Microservices (FTMS)
# dispatched to a SLURM cluster. NOTHING trains locally; this script only talks to the TAO API.
#
# Prereqs:
#   1. .env filled in (see .env.example): TAO_BASE_URL, NGC_ORG_NAME, NGC_KEY, HF_TOKEN, SLURM_*, *_DATASET_URI
#   2. Dataset generated (python dataset/generate_tube_dataset.py) and each split folder
#      (images.tar.gz + annotations.json) uploaded to the location in TRAIN_DATASET_URI / EVAL_DATASET_URI.
#   3. venv activated (tao CLI on PATH)  -> Windows: .\.venv\Scripts\Activate.ps1 then run under Git Bash,
#      or run this script from WSL/Linux with the same venv layout.
#
# Usage:  bash cluster/submit_tao_job.sh [train|evaluate|status|logs] [JOB_ID]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a

ACTION="${1:-train}"
JOB_ID="${2:-}"
EXP_NAME="${EXP_NAME:-cosmos3-nano-tube-inspection}"
ENC_KEY="${TAO_ENCRYPTION_KEY:-tlt_encode}"
STATE_FILE=".tao_state.json"

req() { for v in "$@"; do [ -n "${!v:-}" ] || { echo "Missing env var: $v" >&2; exit 1; }; done; }
save() { python - "$1" "$2" <<'EOF'
import json,sys,pathlib
p=pathlib.Path(".tao_state.json"); d=json.loads(p.read_text()) if p.exists() else {}
d[sys.argv[1]]=sys.argv[2]; p.write_text(json.dumps(d,indent=1))
EOF
}
load() { python -c "import json,sys;print(json.load(open('.tao_state.json')).get(sys.argv[1],''))" "$1" 2>/dev/null || true; }
id_of() { python -c "import json,sys;d=json.load(sys.stdin);print(d.get('id') or d.get('job_id') or d)"; }

req TAO_BASE_URL NGC_ORG_NAME NGC_KEY
tao login --tao-base-url "$TAO_BASE_URL" --ngc-org-name "$NGC_ORG_NAME" --ngc-key "$NGC_KEY" >/dev/null
tao whoami

case "$ACTION" in
  train)
    req SLURM_USER SLURM_HOSTNAME SLURM_PARTITION SLURM_CLUSTER_NAME TRAIN_DATASET_URI HF_TOKEN

    WS_ID="$(load workspace_id)"
    if [ -z "$WS_ID" ]; then
      echo ">> creating SLURM workspace"
      WS_ID=$(tao cosmos-rl create-workspace-slurm --name "$EXP_NAME-ws" \
        --slurm-user "$SLURM_USER" --slurm-hostname "$SLURM_HOSTNAME" \
        ${SLURM_BASE_RESULTS_DIR:+--base-results-dir "$SLURM_BASE_RESULTS_DIR"} --output json | id_of)
      save workspace_id "$WS_ID"
    fi
    echo "workspace: $WS_ID"

    TRAIN_DS="$(load train_dataset_id)"
    if [ -z "$TRAIN_DS" ]; then
      echo ">> registering train dataset $TRAIN_DATASET_URI"
      TRAIN_DS=$(tao cosmos-rl create-dataset --dataset-type vlm --dataset-format llava \
        --workspace-id "$WS_ID" --cloud-file-path "$TRAIN_DATASET_URI" --use-for training --output json | id_of)
      save train_dataset_id "$TRAIN_DS"
    fi
    EVAL_DS="$(load eval_dataset_id)"
    if [ -z "$EVAL_DS" ] && [ -n "${EVAL_DATASET_URI:-}" ]; then
      echo ">> registering eval dataset $EVAL_DATASET_URI"
      EVAL_DS=$(tao cosmos-rl create-dataset --dataset-type vlm --dataset-format llava \
        --workspace-id "$WS_ID" --cloud-file-path "$EVAL_DATASET_URI" --use-for evaluation --output json | id_of)
      save eval_dataset_id "$EVAL_DS"
    fi
    echo "datasets: train=$TRAIN_DS eval=${EVAL_DS:-none}"

    echo ">> base experiments available for cosmos-rl:"
    tao cosmos-rl list-base-experiments --filter-param network_arch=cosmos-rl || true

    echo ">> submitting LoRA SFT train job to SLURM partition $SLURM_PARTITION"
    JOB=$(tao cosmos-rl create-job --kind experiment --action train \
      --name "$EXP_NAME" --encryption-key "$ENC_KEY" --workspace-id "$WS_ID" \
      --train-dataset "$TRAIN_DS" ${EVAL_DS:+--eval-dataset "$EVAL_DS"} \
      --specs @specs/cosmos3_nano_lora_sft.yaml \
      --backend-type slurm --partition "$SLURM_PARTITION" --cluster-name "$SLURM_CLUSTER_NAME" \
      --env HF_TOKEN="$HF_TOKEN" ${WANDB_API_KEY:+--env WANDB_API_KEY="$WANDB_API_KEY"} \
      --tag tube-inspection --tag cosmos3-nano --output json | id_of)
    save train_job_id "$JOB"
    echo "train job: $JOB   (poll with: bash cluster/submit_tao_job.sh status $JOB)"
    ;;

  evaluate)
    req SLURM_PARTITION SLURM_CLUSTER_NAME HF_TOKEN
    PARENT="${JOB_ID:-$(load train_job_id)}"; [ -n "$PARENT" ] || { echo "no train job id"; exit 1; }
    WS_ID="$(load workspace_id)"; EVAL_DS="$(load eval_dataset_id)"
    JOB=$(tao cosmos-rl create-job --kind experiment --action evaluate \
      --name "$EXP_NAME-eval" --encryption-key "$ENC_KEY" --workspace-id "$WS_ID" \
      --parent-job-id "$PARENT" ${EVAL_DS:+--eval-dataset "$EVAL_DS"} \
      --specs @specs/evaluate.yaml \
      --backend-type slurm --partition "$SLURM_PARTITION" --cluster-name "$SLURM_CLUSTER_NAME" \
      --env HF_TOKEN="$HF_TOKEN" --output json | id_of)
    save eval_job_id "$JOB"; echo "evaluate job: $JOB"
    ;;

  status)
    J="${JOB_ID:-$(load train_job_id)}"; tao cosmos-rl get-job-status --job-id "$J" 2>/dev/null || tao cosmos-rl get-job-status "$J"
    ;;

  logs)
    J="${JOB_ID:-$(load train_job_id)}"; tao cosmos-rl get-job-logs --job-id "$J" 2>/dev/null || tao cosmos-rl get-job-logs "$J"
    ;;

  download)
    J="${JOB_ID:-$(load train_job_id)}"; mkdir -p results
    tao cosmos-rl download-entire-job --job-id "$J" --output results 2>/dev/null || tao cosmos-rl download-entire-job "$J"
    ;;

  *) echo "unknown action $ACTION (train|evaluate|status|logs|download)"; exit 1 ;;
esac
