# 09 · Evaluation

README step 10. Files: [`cluster/ptyche_eval.sbatch`](../cluster/ptyche_eval.sbatch) and
[`cluster/eval_in_container.py`](../cluster/eval_in_container.py).

## Why a separate evaluation

Training reports **loss** (how well next-token probabilities match the reference text). The business question is
**accuracy** (does the model give the right answer). We therefore generate answers for held-out validation
questions and grade them, for two models:

- the **fine-tuned** model: converted base + LoRA adapter from `output/best/safetensors`
- the **base model** zero-shot: converted base only, same prompts

Both run at once on two GPUs of one node, ~10 minutes.

## `ptyche_eval.sbatch`

Same skeleton as the training sbatch. `RUN` selects the results folder; if unset it picks the newest
`results/cosmos3_nano_tube_lora_*` that has an `output/` dir. Inside the container it:

1. resolves the adapter: `output/best/*safetensors*` if present, else the highest `output/*/safetensors/epoch_N`
2. prints `adapter_config.json` and library versions (transformers 4.57.6, torch 2.12, peft 0.17.1 in this image)
3. launches two evaluations in the background with `CUDA_VISIBLE_DEVICES=0` and `=1`, waits for both, prints the
   tails of `eval_finetuned.log` and `eval_baseline.log`, exits non-zero if either failed

## `eval_in_container.py`

### Sampling

Reads `annotations.json`, groups by `category`, takes the first `LIMIT/5` of each → 300 samples = 60 per
question type, always the same ones (deterministic, comparable across models).

### Model loading

`Qwen3VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=bfloat16, device_map="cuda",
attn_implementation="sdpa")`, with cuDNN's fused attention disabled
(`torch.backends.cuda.enable_cudnn_sdp(False)`): on GB200 cuDNN had no valid execution plan for these shapes
and crashed the first baseline run.

### Merging the LoRA adapter

PEFT's `PeftModel.from_pretrained` rejected cosmos-rl's `adapter_config.json` (it contains non-standard fields
such as `r_pattern: null`). The script therefore merges the adapter itself:

1. Read all `*.safetensors` under the adapter dir; collect keys containing `lora_A` / `lora_B` (144 pairs).
2. For each pair, find the base weight by **path suffix**: strip the `lora_A.default.weight` tail and prefixes
   like `base_model.`/`policy.`, then look for the unique base state-dict key ending in the longest matching
   suffix (e.g. `layers.5.self_attn.q_proj`). This works regardless of how the exporter prefixed keys.
3. `W += (alpha / r) · B @ A` in float32, cast back to bf16. `alpha/r = 32/16 = 2` from the adapter config.
4. Log `merged 144 LoRA deltas into base weights; unmatched: 0`. Zero merges is a hard error.

This is mathematically identical to PEFT's `merge_and_unload()`.

### Inference

For each sample: open the image, take the human turn minus the `<image>` token, build the Qwen3-VL chat
template with `apply_chat_template(add_generation_prompt=True)`, run `generate(max_new_tokens=768,
do_sample=False)` (greedy), decode only the new tokens.

### Scoring

| category | rule |
| --- | --- |
| content, count, match, deviation_list | normalized exact match: lowercase, strip, drop trailing period, ignore spaces; `<think>…</think>` blocks removed first |
| report | parse the first `{…}` as JSON; the sample counts as correct only if JSON is valid **and** every vial's `actual` and `status` match **and** the deviating set matches **and** `pass` matches |

For reports we additionally accumulate per-field rates: `json_valid_rate`, `per_vial_content_accuracy`,
`per_vial_status_accuracy`, `per_vial_fill_within_10pct`, `deviating_set_exact`, `pass_fail_accuracy`.

### Outputs

`eval_<name>/predictions.jsonl` (id, image, category, gold, pred, correct) and `eval_<name>/metrics.json`.
Progress lines every 20 samples show running correct counts per category.

## Results of the verified run (300 questions)

| Question type | Base model | Fine-tuned |
| --- | --- | --- |
| Content of one vial | 68.3 % | **100 %** |
| Vial count | 100 % | **100 %** |
| Does vial N match the plan | 76.7 % | **100 %** |
| List all deviating positions | 10.0 % | **88.3 %** |
| Full JSON inspection report | 0.0 % | **95.0 %** |
| **Overall** | **51.0 %** | **96.7 %** |

Report fields (60 reports, 398 vials), fine-tuned: valid JSON 100 %, per-vial content 99.7 %, per-vial status
99.2 %, fill within 10 pts 99.0 %, deviating set exact 96.7 %, pass/fail 100 %. Base model: valid JSON 100 %
but per-vial content 15.8 % and pass/fail 51.7 %.

Interpretation: the untuned model already sees colors and counts, but does not perform the *comparison* against
the plan; it tends to echo the plan ("colored cubes", "33.3") instead of reporting what is visible. Fine-tuning
taught the comparison. All 10 fine-tuned misses are under-detections of **overfill** (9 of 11 missed defects),
mostly on **clear** liquid, and none are false alarms; the pass/fail verdict was never wrong.

## Caveats

- Same generator for train and val → this is in-distribution accuracy. Real plant images will be lower.
- 60 samples per category gives roughly ±4 points of sampling uncertainty at these accuracies.
- Greedy decoding; a served model with temperature > 0 would score slightly differently.

## Re-running with more samples or a different checkpoint

```bash
sbatch … --export=ALL,…,RUN=<run>,LIMIT=1000 cluster/ptyche_eval.sbatch     # all 1000 val samples, ~35 min
```

To evaluate a specific epoch rather than `best`, edit `ADAPTER=` in the sbatch or run `eval_in_container.py`
by hand from an interactive shell (`ptyche_interactive.sh`).
