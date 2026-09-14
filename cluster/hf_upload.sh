#!/usr/bin/env bash
# Upload the fine-tuned adapter (+ eval results, report, optional 17 GB base) to a Hugging Face Hub repo.
# Run on the cluster login node (has internet + the Lustre venv with the `hf` CLI).
#
#   export HF_TOKEN=hf_xxx                     # write-scoped token from https://huggingface.co/settings/tokens
#   export LUSTRE_DIR=/lustre/fsw/general_sa/$USER
#   bash cluster/hf_upload.sh Balab2021/csicosmos3nanoquality                       # adapter + eval + card (~50 MB)
#   WITH_BASE=1 bash cluster/hf_upload.sh Balab2021/csicosmos3nanoquality           # also base/ (17 GB, via srun)
#   PUBLIC=1 ... to create a public repo (default: private)
set -euo pipefail
REPO_ID="${1:?usage: hf_upload.sh <namespace/repo> }"
RUN="${RUN:-cosmos3_nano_tube_lora_3044273}"
STAMP="${STAMP:-20260913221703}"      # output/<timestamp> inside the run
EPOCH="${EPOCH:-epoch_5}"
LUSTRE_DIR="${LUSTRE_DIR:-/lustre/fsw/${ACCOUNT:-general_sa}/$USER}"
WORK="${WORK:-$LUSTRE_DIR/cosmos3tao}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
R="$WORK/results/$RUN"
ADAPTER="$R/output/$STAMP/safetensors/$EPOCH"
BASE="$WORK/models/Cosmos3-Nano-qwen3vl"
: "${HF_TOKEN:?export HF_TOKEN=<write token> first}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" HF_HOME="${HF_HOME:-$WORK/hf_cache}"
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"
pip install -q --upgrade huggingface_hub >/dev/null

[ -f "$ADAPTER/adapter_config.json" ] || { echo "FATAL: adapter not found at $ADAPTER"; exit 1; }
STAGE=$(mktemp -d "$WORK/.hf_stage.XXXX")
trap 'rm -rf "$STAGE"' EXIT

echo ">> staging adapter + eval + report"
mkdir -p "$STAGE/adapter" "$STAGE/eval/finetuned" "$STAGE/eval/baseline"
cp "$ADAPTER"/* "$STAGE/adapter/"
# make the adapter self-describing for Hub users: base model = the HF repo's base/ subfolder
python - "$STAGE/adapter/adapter_config.json" "$REPO_ID" <<'EOF'
import json, sys
p, repo = sys.argv[1], sys.argv[2]
c = json.load(open(p)); c["base_model_name_or_path"] = f"{repo}/base"; json.dump(c, open(p, "w"), indent=2)
EOF
cp "$R"/eval_finetuned/{metrics.json,predictions.jsonl} "$STAGE/eval/finetuned/" 2>/dev/null || true
cp "$R"/eval_baseline/{metrics.json,predictions.jsonl}  "$STAGE/eval/baseline/"  2>/dev/null || true
cp "$REPO_DIR/hf/README.md" "$STAGE/README.md"
cp "$REPO_DIR"/report/run_*.html "$STAGE/report.html" 2>/dev/null || true
cp "$R/train_spec.yaml" "$STAGE/train_spec.yaml" 2>/dev/null || true
du -sh "$STAGE"

VIS=(--private); [ "${PUBLIC:-0}" = "1" ] && VIS=()
echo ">> creating repo $REPO_ID (${VIS[*]:-public})"
hf repo create "$REPO_ID" --type model "${VIS[@]}" --exist-ok >/dev/null

echo ">> uploading adapter, eval, card"
hf upload "$REPO_ID" "$STAGE" . --commit-message "LoRA adapter, evaluation results and model card ($RUN, $EPOCH)"

if [ "${WITH_BASE:-0}" = "1" ]; then
  echo ">> uploading base/ (17 GB) as a SLURM step so the login node is not OOM-killed"
  srun -A "${ACCOUNT:-general_sa}" -p "${PARTITION:-gb200}" -N1 -n1 --cpus-per-task=8 --mem=32G --time=03:00:00 \
       --job-name="${ACCOUNT:-general_sa}-cosmos3.hf-upload" --export=ALL \
       "$WORK/venv/bin/hf" upload "$REPO_ID" "$BASE" base --commit-message "Cosmos3-Nano reasoner re-keyed as Qwen3-VL (bf16)"
fi
echo
echo "Done: https://huggingface.co/$REPO_ID"
