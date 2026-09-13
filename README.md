# cosmos3tao

NVIDIA Cosmos 3 + TAO (Train Adapt Optimize): LoRA fine-tune **nvidia/Cosmos3-Nano** for a
**vial-filling line color inspection** task. Training runs as a SLURM job inside the TAO cosmos-rl container
(Pyxis/Enroot) on an NVIDIA GPU cluster. Nothing trains on the laptop.

**Verified end to end on 2026-09-13** on the lyris cluster, partition `gb200` (1 node × 4 GB200 GPUs, arm64):
5 epochs, 625 optimizer steps, ~25 min wall time, best validation loss 0.0353 at epoch 5. Everything below is the
exact sequence that worked.

## The inspection task

Side-view camera on a MagneMotion-style linear track. Each wheeled carrier holds one labeled clear vial that
contains colored liquid (about one third full), nothing, or a stack of colored cubes. The **manufacturing plan**
(expected content per position / vial ID) is passed as text in the prompt, since it is not visible in the plant
image. The model reports actual content, fill level, per-vial status (`OK`, `wrong_color`, `underfill`,
`overfill`, `empty`, `wrong_content`), the deviating positions, and pass/fail. A real plant photo is in
`data/real/plant.jpg`.

## Layout

| Path | Purpose |
| --- | --- |
| `cluster/ptyche_setup.sh` | **Cluster, once:** venv on Lustre, synthetic dataset, model download, container → `.sqsh` |
| `cluster/ptyche_prepare_model.sh` + `cluster/convert_omni_to_qwen3vl.py` | **Cluster, once:** Cosmos3-Nano (omni) → Qwen3-VL HF checkpoint |
| `cluster/ptyche_train.sbatch` | **Cluster:** 1 node × 4 GPU batch job (partition/account overridden on the command line) |
| `cluster/train_in_container.sh` | Runs inside the container: preflight, render spec → TOML, launch `cosmos-rl` |
| `cluster/render_spec.py` | Fills model/data/results paths and GPU count into the spec; emits YAML + TOML |
| `cluster/ptyche_interactive.sh` | Same allocation as an interactive `--pty bash` inside the TAO container |
| `cluster/probe_container.sh` | Read-only diagnostics of the TAO image (launchers, hooks, converters) |
| `specs/cosmos3_nano_lora_sft.yaml` | TAO cosmos-rl LoRA SFT spec (LoRA r=16 on q/k/v/o, lr 2e-5, 5 epochs) |
| `specs/evaluate.yaml`, `specs/inference.yaml` | TAO evaluate / inference specs |
| `dataset/generate_tube_dataset.py` | Synthetic side-view images + LLaVA-format `annotations.json` |
| `cluster/ptyche_eval.sbatch` + `cluster/eval_in_container.py` | **Cluster:** task accuracy of fine-tuned adapter vs zero-shot base on the val split |
| `report/make_report.py` | Executive HTML training report from `train.log` (`report/run_lyris_gb200_3044273.html` is the real example) |
| `inspect/inspect_tubes.py` | Inspection client + scorer (hosted Cosmos 3 reasoner or your fine-tuned endpoint) |
| `setup.ps1`, `requirements.txt` | Windows: Python 3.14 venv for local tooling only |
| `cluster/submit_tao_job.sh` | Alternative path through a TAO FTMS API deployment, if you have one |

## Cluster requirements

- SLURM with Pyxis/Enroot (`srun --container-image`), a shared Lustre directory, internet from the login node.
- GPUs supported by the TAO 7.0.1 cosmos-rl image: **H100/A100 (x86_64)** and **GB200 (arm64, sm_100)**.
  **GB300 (sm_103) does not work**: the image's NVRTC rejects the architecture at the first CUDA kernel
  (`nvrtc: error: invalid value for --gpu-architecture`). Do not use theia `gb300` or hecate `batch-xdr`.
- ≥ 256 GB total GPU memory for Cosmos3-Nano SFT. 4 × GB200 (186 GB each) is plenty.
- Enroot credentials for nvcr.io in `~/.config/enroot/.credentials` (one line: `machine nvcr.io login $oauthtoken password <NGC key>`).

## End-to-end runbook (lyris, GB200)

Run everything on the cluster login node. Re-export the variables in every new shell.

### 0. Log in

```bash
ssh <user>-mfa@login-lyris.nvidia.com
```

### 1. Environment variables

```bash
export ACCOUNT=general_sa PARTITION=gb200 LUSTRE_DIR=/lustre/fsw/general_sa/$USER
```

### 2. Clone (or update) the repo on Lustre

```bash
mkdir -p $LUSTRE_DIR/cosmos3tao && cd $LUSTRE_DIR/cosmos3tao && { [ -d repo ] && git -C repo pull || git clone https://github.com/balakreshnan/cosmos3tao.git repo; } && cd repo
```

### 3. Enroot credentials

```bash
ls ~/.config/enroot/.credentials || { mkdir -p ~/.config/enroot; echo "machine nvcr.io login \$oauthtoken password <YOUR_NGC_KEY>" > ~/.config/enroot/.credentials; chmod 600 ~/.config/enroot/.credentials; }
```

### 4. One-time setup (~20 min)

Creates `$LUSTRE_DIR/cosmos3tao/{venv,data,models,sqsh,logs,results}`: Python venv, 800 train / 200 val
synthetic images, the 33 GB `nvidia/Cosmos3-Nano` download (as a SLURM step, Xet disabled), and the container
imported to a `.sqsh` named for the compute node's CPU architecture. Every step is skipped if already present.

```bash
bash cluster/ptyche_setup.sh
```

Expect `Setup complete.` and a sqsh named `..._7.0.1-cosmos-rl-aarch64.sqsh`.

### 5. Convert the checkpoint (~10 min, once)

The Hugging Face repo is a single Omni checkpoint (`model_type=cosmos3_omni`: 8B reasoner + video generator +
action/audio heads). The TAO image's Transformers only knows Qwen3-VL, which is exactly what the reasoner is. This
extracts the 750 reasoner + vision tensors into a Qwen3-VL checkpoint at `models/Cosmos3-Nano-qwen3vl` using
`Qwen/Qwen3-VL-8B-Instruct` as the architecture donor (same dims), running inside the container as a CPU step.

```bash
bash cluster/ptyche_prepare_model.sh
```

Expect the last lines `qwen3_vl ['Qwen3VLForConditionalGeneration']`.

### 6. Submit training

```bash
sbatch --account=$ACCOUNT --partition=$PARTITION --export=ALL,ACCOUNT=$ACCOUNT,LUSTRE_DIR=$LUSTRE_DIR --output=$LUSTRE_DIR/cosmos3tao/logs/%x-%j.out --error=$LUSTRE_DIR/cosmos3tao/logs/%x-%j.err cluster/ptyche_train.sbatch
```

Knobs via `--export`: `EPOCHS=3`, `GPUS=4` (default; capped at the node's GPU count). For a shorter queue wait
use `--partition=gb200-backfill --time=3:00:00`.

### 7. Monitor

```bash
J=$(squeue -u $USER -h -o %i -n general_sa-cosmos3.tao-finetune | head -1); LOG=$LUSTRE_DIR/cosmos3tao/logs/general_sa-cosmos3.tao-finetune-$J.out; echo "job $J"; until [ -f "$LOG" ]; do sleep 30; done; tail -f "$LOG"
```

Status and history:

```bash
squeue -u $USER; sacct -u $USER --starttime today --format=JobID,JobName%32,State,Elapsed,ExitCode
```

The log starts with a preflight (GPUs, `model_type: qwen3_vl`, sample counts, rendered spec, dataset hook), then
`== launching: cosmos-rl --config .../spec.toml /opt/cosmos_rl/tao_sft_example.py`, then per-step loss lines.
Success ends with `Cosmos-RL SFT training completed successfully` and `== training exit code: 0`.

### 8. Results

```
$LUSTRE_DIR/cosmos3tao/results/cosmos3_nano_tube_lora_<jobid>/
  train_spec.yaml, train_spec.toml, spec.toml   rendered config
  train.log                                     full log (all ranks)
  output/<timestamp>/checkpoints/epoch_N/policy cosmos-rl checkpoints (last 3 epochs kept)
  output/<timestamp>/safetensors/epoch_N/       HF safetensors export of the LoRA policy
  output/best/best_score.json                   best validation score + epoch
```

### 9. Training report on the laptop

From PowerShell in the repo, copy the small files (skip the multi-GB weights) and build the report:

```powershell
New-Item -ItemType Directory -Force results | Out-Null
```

```powershell
scp <user>-mfa@login-lyris.nvidia.com:/lustre/fsw/general_sa/<user>/cosmos3tao/results/cosmos3_nano_tube_lora_<jobid>/train.log results\
```

```powershell
scp "<user>-mfa@login-lyris.nvidia.com:/lustre/fsw/general_sa/<user>/cosmos3tao/results/cosmos3_nano_tube_lora_<jobid>/train_spec.*" results\
```

```powershell
.\.venv\Scripts\python.exe report\make_report.py --log results\train.log --spec results\train_spec.yaml --out results\report.html; Start-Process results\report.html
```

One self-contained HTML page: executive summary (auto-written from the log), KPI tiles (best validation loss,
training loss, generalization gap, steps, wall time, compute), loss-over-training chart with epoch bands and
validation points, per-epoch table, learning-rate / gradient-norm / step-time charts, insights, configuration and
reproduce commands. Crosshair tooltips, smoothing, log axis, data-table view, dark mode, print-friendly.
Add `--cluster "lyris · gb200 · 1 node × 4 GB200" --job <jobid>` for the header. Example from the verified run:
`report/run_lyris_gb200_3044273.html`.

## Running on another cluster

Same recipe. Only these change: the login host, `ACCOUNT`, `PARTITION`, and whether the Lustre path exists.
Setup probes the compute partition's CPU architecture and imports a matching `.sqsh`; x86 and arm64 images
coexist under `sqsh/`.

| Cluster | Partition to use | Notes |
| --- | --- | --- |
| lyris | `gb200` / `gb200-backfill` | **verified**; arm64 GB200, 4 GPUs/node |
| ptyche (Pre-Tyche) | `batch`, `36x2-a01r`, `backfill` | x86 nodes; no `gpu` GRES, nodes allocated whole; long queues |
| ptyche `a02grace`, hecate `batch-xdr`, theia `gb300` | avoid | GB300 sm_103 unsupported by TAO 7.0.1 image |

Job names follow the `account-<subproject>.<details>` convention some clusters enforce.

## Local (Windows) tooling

```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python dataset\generate_tube_dataset.py --out data\tube_inspection --train 800 --val 200
python inspect\inspect_tubes.py --split data\tube_inspection\val --limit 50   # zero-shot baseline via NVIDIA API
python inspect\inspect_tubes.py --image data\real\plant.jpg --plan "position 1 (VIAL 0174): colored cubes; position 2: colored cubes; position 3: empty; position 4 (VIAL 0019): orange liquid; position 5: blue liquid; position 6: yellow liquid; position 7 (VIAL 0042): blue liquid; position 8: orange liquid"
```

Copy `.env.example` to `.env` for the NVIDIA API key. `nvidia-tao-client` is installed with `--no-deps`
because its dependency tree includes uWSGI, which does not build on Windows.

## Troubleshooting (all hit and fixed during the first run)

| Symptom | Cause / fix |
| --- | --- |
| `hf download` killed on login node | login-node memory limit + Xet backend; setup now downloads inside a SLURM step with `HF_HUB_DISABLE_XET=1` |
| `Invalid generic resource (gres) specification` | cluster has no `gpu` GRES; sbatch requests none, container uses `CUDA_VISIBLE_DEVICES=0..GPUS-1` |
| `No partition specified` | `$PARTITION` unset in a new shell; re-run step 1 |
| `cosmos_rl: command not found` | launcher is `cosmos-rl --config spec.toml /opt/cosmos_rl/tao_sft_example.py` (what TAO FTMS runs) |
| `model type cosmos3_omni not recognized` | convert with `ptyche_prepare_model.sh` (step 5) |
| `train_batch_per_replica(1) must be divisible by mini_batch(4)` | spec uses batch 8 / mini-batch 2 |
| `nvrtc: error: invalid value for --gpu-architecture` | GB300 node; resubmit on `gb200` |
| sqsh named `-x86_64` but nodes are arm64 | old setup used the login node's arch; now probes the partition. Rename the file to `-aarch64.sqsh` |
| YAML `2e-5` parsed as a string | write floats as `2.0e-5` in the spec |

## References

- [TAO Cosmos-Reason fine-tuning docs](https://docs.nvidia.com/tao/tao-toolkit/latest/text/vlm_finetuning/cosmos_rl.html)
- [Running TAO via FTMS](https://docs.nvidia.com/tao/tao-toolkit/latest/text/tao_toolkit_api/index.html)
- [TAO skill bank](https://github.com/NVIDIA-TAO/tao-skill-bank)
- [nvidia/Cosmos3-Nano on Hugging Face](https://huggingface.co/nvidia/Cosmos3-Nano)
- [Qwen/Qwen3-VL-8B-Instruct (architecture donor)](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
