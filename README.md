# cosmos3tao

NVIDIA Cosmos 3 + TAO (Train Adapt Optimize) example: LoRA fine-tune **nvidia/Cosmos3-Nano** for a
**tube-filling line color inspection** task. Training runs on the Pre-Tyche SLURM cluster inside the TAO
cosmos-rl container (Pyxis/Enroot). Nothing trains on the laptop.

## The inspection task

Side-view camera on a MagneMotion-style linear track. Each wheeled carrier holds one labeled clear vial
that contains colored liquid (about one third full), nothing, or a stack of colored cubes. The
**manufacturing plan** (expected content per position / vial ID) is passed as text in the prompt, since it is
not visible in the plant image. The model reports actual content, fill level, per-vial status (`OK`,
`wrong_color`, `underfill`, `overfill`, `empty`, `wrong_content`), the deviating positions, and pass/fail.

Test on a real plant photo:

```powershell
python inspect\inspect_tubes.py --image plant.jpg --plan "position 1 (VIAL 0019): orange liquid; position 2 (VIAL 0020): blue liquid; position 3 (VIAL 0021): yellow liquid"
```

## Layout

| Path | Purpose |
| --- | --- |
| `setup.ps1` | Windows: Python 3.14 venv for local tooling (dataset gen, TAO client, inference client) |
| `dataset/generate_tube_dataset.py` | Synthetic images + LLaVA-format `annotations.json` (TAO `vlm/llava`) |
| `specs/cosmos3_nano_lora_sft.yaml` | TAO cosmos-rl LoRA SFT spec (4 GPUs) |
| `specs/evaluate.yaml`, `specs/inference.yaml` | TAO evaluate / inference specs |
| `cluster/ptyche_setup.sh` | **Cluster, once:** venv on Lustre, dataset, model download, container → sqsh |
| `cluster/ptyche_prepare_model.sh`, `cluster/convert_omni_to_qwen3vl.py` | **Cluster, once:** Cosmos3-Nano (omni) → Qwen3-VL HF checkpoint |
| `cluster/ptyche_train.sbatch` | **Cluster:** 1 node × 4 GPU batch job |
| `cluster/ptyche_interactive.sh` | **Cluster:** same allocation as an interactive `--pty bash` in the TAO container |
| `cluster/train_in_container.sh` | Runs inside the container: preflight, render spec, launch training |
| `cluster/render_spec.py` | Fills model/data/results paths and GPU count into the spec |
| `cluster/submit_tao_job.sh` | Alternative: submit through a TAO FTMS API deployment (if you have one) |
| `inspect/inspect_tubes.py` | Inspection client + scorer (hosted Cosmos 3 reasoner or your fine-tuned endpoint) |

## Run on Pre-Tyche (4 GPUs)

On the login node, after `ssh <you>@login-ptyche`:

```bash
export ACCOUNT=general_sa
export LUSTRE_DIR=/lustre/fsw/$ACCOUNT/$USER
mkdir -p $LUSTRE_DIR/cosmos3tao && cd $LUSTRE_DIR/cosmos3tao
git clone https://github.com/balakreshnan/cosmos3tao.git repo && cd repo
```

One-time setup (venv + dataset + ~33 GB model download + container import):

```bash
bash cluster/ptyche_setup.sh
```

Convert the checkpoint once. The Hugging Face download is `model_type=cosmos3_omni` (reasoner + video generator in
one Omni checkpoint), which the TAO 7.0.1 cosmos-rl image cannot load. This extracts the reasoner and vision tower
into a standard Qwen3-VL checkpoint (`models/Cosmos3-Nano-qwen3vl`), running inside the container as a short job:

```bash
bash cluster/ptyche_prepare_model.sh
```

Submit the batch fine-tune:

```bash
sbatch cluster/ptyche_train.sbatch
```

```bash
squeue -u $USER
```

```bash
tail -f $LUSTRE_DIR/cosmos3tao/logs/general_sa-cosmos3.tao-finetune-*.out
```

Or do the same interactively (drops you into the TAO container with 4 GPUs, then run the trainer):

```bash
bash cluster/ptyche_interactive.sh
```

```bash
bash /tao-repo/cluster/train_in_container.sh
```

Supported GPUs for the TAO 7.0.1 cosmos-rl image: H100/A100 (x86) and GB200 (arm64, sm_100). GB300 (sm_103)
fails at the first CUDA kernel with `nvrtc: error: invalid value for --gpu-architecture`; use partition `gb200`.

Knobs: `GPUS=4 EPOCHS=5 sbatch cluster/ptyche_train.sbatch`. If the image exposes a different launcher
than `cosmos_rl train -e <spec> -r <results>`, the preflight prints what is on PATH; set
`TAO_TRAIN_CMD` accordingly. Results, spec, `train.log` and `status.json` land in
`$LUSTRE_DIR/cosmos3tao/results/<run>/`.

## Training report (interactive HTML)

Copy the run folder (without the multi-GB checkpoints) from the cluster and build a self-contained report:

```bash
rsync -av --exclude 'checkpoints' --exclude 'safetensors' <user>@login-lyris:/lustre/fsw/general_sa/<user>/cosmos3tao/results/<run>/ results/<run>/
```

```powershell
python report\make_report.py --results results\<run> --out results\<run>\report.html
```

Loss / lr / grad-norm curves with crosshair tooltips, smoothing, log axis, table view, dark mode, run config and
checkpoint events. See `report/sample_report.html` for an example built from a synthetic log.

## Local (Windows) tooling

```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python dataset\generate_tube_dataset.py --out data\tube_inspection --train 800 --val 200
python inspect\inspect_tubes.py --split data\tube_inspection\val --limit 50   # zero-shot baseline via NVIDIA API
```

Copy `.env.example` to `.env` for the NVIDIA API key. `nvidia-tao-client` is installed with `--no-deps`
because its dependency tree includes uWSGI, which does not build on Windows.

## References

- [TAO Cosmos-Reason fine-tuning docs](https://docs.nvidia.com/tao/tao-toolkit/latest/text/vlm_finetuning/cosmos_rl.html)
- [Running TAO via FTMS](https://docs.nvidia.com/tao/tao-toolkit/latest/text/tao_toolkit_api/index.html)
- [TAO skill bank](https://github.com/NVIDIA-TAO/tao-skill-bank)
- [nvidia/Cosmos3-Nano on Hugging Face](https://huggingface.co/nvidia/Cosmos3-Nano)
