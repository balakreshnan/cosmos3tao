# 12 · Inference and next steps

## What you have after training

- **Base model** `models/Cosmos3-Nano-qwen3vl` (17 GB, Qwen3-VL layout)
- **Adapter** `results/<run>/output/best/safetensors` (~100 MB: `adapter_config.json` + LoRA A/B for 144 modules)
- A merged model is never written to disk; the evaluator merges in memory. To produce one, run the merge code
  from `eval_in_container.py` and `model.save_pretrained(...)`.

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
