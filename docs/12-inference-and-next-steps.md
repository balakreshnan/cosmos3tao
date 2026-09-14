# 12 · Inference and next steps

## What you have after training

- **Base model** `models/Cosmos3-Nano-qwen3vl` (17 GB, Qwen3-VL layout)
- **Adapter** `results/<run>/output/best/safetensors` (~100 MB: `adapter_config.json` + LoRA A/B for 144 modules)
- A merged model is never written to disk; the evaluator merges in memory. To produce one, run the merge code
  from `eval_in_container.py` and `model.save_pretrained(...)`.

## Looking at the weights on the cluster

```bash
R=$LUSTRE_DIR/cosmos3tao/results/cosmos3_nano_tube_lora_3044273
ls -la $R/output/best/                                  # symlinks 'safetensors' and 'checkpoint' -> best epoch
readlink -f $R/output/best/safetensors                  # …/output/20260913221703/safetensors/epoch_5
ls -lh $(readlink -f $R/output/best/safetensors)        # adapter_config.json + adapter *.safetensors
du -sh $(readlink -f $R/output/best/safetensors) $LUSTRE_DIR/cosmos3tao/models/Cosmos3-Nano-qwen3vl
cat $R/output/best/best_score.json
```

Inspect the tensors without loading a model (inside the container or the login venv after `pip install safetensors`):

```bash
python - <<'EOF'
from safetensors import safe_open; import glob, os
d = os.path.realpath(os.path.expandvars("$LUSTRE_DIR/cosmos3tao/results/cosmos3_nano_tube_lora_3044273/output/best/safetensors"))
n = 0
for f in sorted(glob.glob(d + "/*.safetensors")):
    with safe_open(f, "pt") as sf:
        keys = list(sf.keys()); n += len(keys)
        for k in keys[:4]: print(k, tuple(sf.get_slice(k).get_shape()))
print(n, "tensors")   # 288 = 144 LoRA modules x (A, B)
EOF
```

## Downloading to the laptop

Two things are needed: the converted base (~17 GB, 5 shards) and the adapter (~100 MB). Copy the adapter's real
directory, not the symlink. PowerShell, from the repo:

```powershell
New-Item -ItemType Directory -Force models\adapter_epoch5 | Out-Null
scp -r "<user>-mfa@login-lyris.nvidia.com:/lustre/fsw/general_sa/<user>/cosmos3tao/results/cosmos3_nano_tube_lora_3044273/output/20260913221703/safetensors/epoch_5/*" models\adapter_epoch5\
scp -r "<user>-mfa@login-lyris.nvidia.com:/lustre/fsw/general_sa/<user>/cosmos3tao/models/Cosmos3-Nano-qwen3vl" models\
```

The 17 GB copy takes a while over VPN; `rsync -avP` from WSL resumes if interrupted. `models/` is git-ignored.

## Local inference on a laptop GPU

`inspect/local_infer.py` loads the base in bfloat16 (~16.5 GB on the GPU), merges the adapter in memory, and
answers one question about one image. A 24 GB GPU is enough for a single image at the training resolution.

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu130 torch
.\.venv\Scripts\python.exe -m pip install "transformers>=4.57" accelerate safetensors peft pillow
```

```powershell
# synthetic validation image; plan looked up from ground_truth.json, answer compared to it
.\.venv\Scripts\python.exe inspect\local_infer.py --image data\tube_inspection\val\images\vials_val_00004.png --ground-truth data\tube_inspection\val\ground_truth.json

# the real plant photo
.\.venv\Scripts\python.exe inspect\local_infer.py --image data\real\plant.jpg --plan "position 1 (VIAL 0174): colored cubes; position 2: colored cubes; position 3: empty; position 4 (VIAL 0019): orange liquid; position 5: blue liquid; position 6: yellow liquid; position 7 (VIAL 0042): blue liquid; position 8: orange liquid"

# base model for comparison
.\.venv\Scripts\python.exe inspect\local_infer.py --image data\real\plant.jpg --plan "..." --no-adapter

# write a merged standalone checkpoint (17 GB) so future loads need no adapter
.\.venv\Scripts\python.exe inspect\local_infer.py --merge-out models\Cosmos3-Nano-tube-merged
```

First load takes 1–2 minutes (reading 17 GB from disk); generation of a JSON report takes ~10–20 s on a laptop
GPU. If you see CUDA out-of-memory, add `--device-map auto` (offloads layers to CPU RAM, slower) or lower
`--max-pixels`.

## Using the model

### From Python inside the container (batch or ad-hoc)

`cluster/eval_in_container.py` is the reference: load base, merge adapter, `apply_chat_template` with the image
and prompt, greedy `generate`. For a single image use `--annotations` with a one-element JSON, or import the
functions from an interactive shell (`ptyche_interactive.sh`).

### Serving

Any OpenAI-compatible server that supports Qwen3-VL and LoRA works (vLLM with `--enable-lora`, or a merged
checkpoint without LoRA flags). TAO also offers `tao cosmos-rl start-inference-microservice` through its FTMS
API. Once served, the laptop client works unchanged:

```powershell
python inspect\inspect_tubes.py --image data\real\plant.jpg --plan "position 1 (VIAL 0174): colored cubes; …" --base-url http://HOST:8000/v1 --model cosmos3-nano-tube
```

`inspect/inspect_tubes.py` builds the same role text + plan + JSON instruction the model was trained on, calls
the endpoint, extracts the JSON, and, when scoring a split, compares against `ground_truth.json`. The same
script targets the hosted NVIDIA Cosmos 3 reasoner from `.env` for a zero-shot comparison.

### The prompt contract

The model was trained on one fixed prompt structure. Keep it:

```
You are a quality inspector watching a vial filling line from the side. Each wheeled carrier on the track holds
one clear vial with a printed label. Vials are numbered by position from left to right. A correctly filled
liquid vial is filled to roughly one third of its height.
Manufacturing plan: position 1 (VIAL 0007): colored cubes; position 2 (VIAL 0008): …
Produce the inspection report as compact JSON with keys: vials (list of {position, vial_id, planned, actual,
fill_pct, status}), deviating_positions, pass. status is one of OK, wrong_color, underfill, overfill, empty,
wrong_content. Output JSON only.
```

Vocabulary the model knows: contents `orange, blue, yellow, red, green, purple, clear, cubes, empty`; statuses
`OK, wrong_color, underfill, overfill, empty, wrong_content`.

## Where the model is weak and how to fix it

| Finding (chapter 09) | Lever | Where |
| --- | --- | --- |
| Overfill under-detected (9 of 11 missed defects) | raise `overfill` share in defect sampling; widen the gap between normal (28–42 %) and over (58–80 %) or add intermediate cases | `generate_tube_dataset.py` `sample_run`, `NORMAL_FILL` |
| Clear liquid hard to see | darker/bluer tint for `clear`, or drop `clear` if the plant never uses it | `LIQUIDS["clear"]` |
| Fill level judged from an unmarked glass | draw tick marks on the vial | `render()` |
| Resolution | `custom.vision.total_pixels: 313600 → 640000` | spec (costs tokens/time) |
| Capacity | add `gate_proj,up_proj,down_proj` to LoRA targets, or `r: 32` | spec |
| Deviation-list question weaker (88 %) than the JSON report (96.7 %) | more `deviation_list` samples per image, or ask the model for the JSON and derive the list | generator / client |

## Toward real plant images

Synthetic accuracy is 96.7 %; the real plant photo is out of distribution (perspective, lighting, label fonts,
reflections). Plan:

1. Collect 50–200 real frames with a plan and the actual state per vial (a CSV is enough).
2. Convert them to the same `annotations.json` format (5 questions per image) and mix into training at
   10–30 % of the data, keeping the synthetic set for volume.
3. Hold out real frames for evaluation; report synthetic and real accuracy separately.
4. Re-run steps 6–10 of the README. Total turnaround is under an hour on a GB200 node.

## Second training round checklist

- change the generator or spec as above; regenerate data (`rm -rf $WORK/data/tube_inspection && bash cluster/ptyche_setup.sh`)
- `sbatch … cluster/ptyche_train.sbatch` (new `results/cosmos3_nano_tube_lora_<newid>`)
- `sbatch … RUN=<new run> cluster/ptyche_eval.sbatch`
- regenerate the report with both evals and compare against `run_lyris_gb200_3044273.html`
