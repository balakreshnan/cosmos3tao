# 06 · The training spec

File: [`specs/cosmos3_nano_lora_sft.yaml`](../specs/cosmos3_nano_lora_sft.yaml). Based on the TAO Cosmos-Reason
documentation's LoRA SFT example, adapted to our data and 4 GPUs. At run time
[`cluster/render_spec.py`](../cluster/render_spec.py) fills in paths and GPU count and writes both
`train_spec.yaml` and `train_spec.toml`; cosmos-rl reads the TOML.

The YAML structure maps 1:1 onto cosmos-rl's `Config` dataclass; TAO's own service literally does
`toml.dumps(spec)`.

## `train` · optimizer and schedule

| field | value | meaning |
| --- | --- | --- |
| `epoch` | 5 | passes over the 4000 training samples |
| `train_batch_per_replica` | 8 | samples per optimizer step per replica (one replica = the 4-GPU FSDP group) |
| `train_policy.mini_batch` | 2 | samples per forward/backward; 8/2 = 4 gradient-accumulation micro-steps. **Must divide `train_batch_per_replica`** (the doc example 1/4 is rejected) |
| `optm_name`, `optm_lr` | AdamW, `2.0e-5` | LoRA tolerates a higher LR than full SFT (doc default 1e-6). Write floats as `2.0e-5`: YAML parses `2e-5` as a **string** |
| `optm_weight_decay` | 0.01 | |
| `optm_warmup_epochs` | 0.2 | ≈ first 25 steps ramp the LR up (visible in the report's learning-rate chart) |
| `optm_min_lr_factor` | 0.1 | decay floor at 10 % of peak |
| `optm_grad_norm_clip` | 1.0 | gradient clipping; the report shows early spikes above this being clipped |
| `param_dtype` / `master_dtype` | bfloat16 / float32 | compute in bf16, keep fp32 master weights for the optimizer |
| `fsdp_offload` | false | 4 × 186 GB is plenty; no CPU offload |
| `ckpt.enable_checkpoint`, `save_freq_in_epoch` | true, 1 | one checkpoint per epoch |
| `ckpt.max_keep` | 3 | older epochs are deleted (log: `Removed old checkpoint: …/epoch_2`) |
| `ckpt.export_safetensors` | true | also write a Hugging Face / PEFT-style adapter per epoch |
| `train_policy.type` | sft | supervised fine-tuning (cosmos-rl also does GRPO RL) |
| `train_policy.dataset.name` / `test_size` | tube_inspection / 0.05 | name is cosmetic; test_size unused because we pass an explicit val set |
| `enable_dataset_cache`, `dataloader_num_workers` | true, 8 | preprocessed samples cached; 8 loader workers |

## `validation`

`enable: true`, `freq_in_epoch: 1` → validation loss is computed at every epoch end on the val split
(`[SFT] Validation loss: 0.0353 for train step 625/625, epoch 4` in the log, zero-indexed epoch).

## `policy` · the model and how it is sharded

| field | value | meaning |
| --- | --- | --- |
| `model_name_or_path` | filled by render_spec: `/tao-workspace/models/Cosmos3-Nano-qwen3vl` | the converted checkpoint (chapter 05) |
| `model_max_length` | 8192 | max tokens per sample; our prompts with one 313 600-pixel image are ~1–2k tokens |
| `model_gradient_checkpointing` | true | recompute activations in backward to save memory |
| `parallelism.dp_shard_size` | 4 (= `GPUS`) | FSDP: model and optimizer state sharded across 4 GPUs, each GPU processes different samples |
| `parallelism.dp_replicate_size`, `tp_size`, `cp_size`, `pp_size` | 1 | no data replication, tensor, context or pipeline parallelism |
| `lora.r` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.05 | rank-16 adapters, scale α/r = 2 |
| `lora.target_modules` | q_proj, k_proj, v_proj, o_proj | attention projections in all 36 language layers → 144 LoRA modules. Vision tower and MLPs stay frozen |
| `lora.modules_to_save` | [] | nothing trained in full |

Trainable parameters: 4 projections × 36 layers × (A: 16×4096 + B: out×16). Roughly 14 M parameters, ~0.2 % of
the model.

## `custom` · what the TAO data hook reads

| field | value |
| --- | --- |
| `dataset.annotation_path` | `/tao-workspace/data/tube_inspection/train/annotations.json` |
| `dataset.media_path` | `/tao-workspace/data/tube_inspection/train` (image paths in annotations are relative to it) |
| `dataset.system_prompt` | empty; our role text is inside each prompt |
| `train_dataset` / `val_dataset` | same information in the newer key names; `val_dataset` drives the per-epoch validation |
| `vision.fps` | 1 (videos only) |
| `vision.total_pixels` | 313 600 ≈ 560 × 560 worth of visual tokens; the 1400 × 700 image is resized to fit. Raising this sharpens the meniscus at the cost of tokens |

## `logging`

`logger: [console, tao]` → console lines every step, plus TAO's structured status logger. `experiment_name` is
overwritten with the run name (`cosmos3_nano_tube_lora_<jobid>`).

## Fields you will most likely change

- `train.epoch`, or `EPOCHS=` at submit time (render_spec overrides).
- `policy.parallelism.dp_shard_size`, or `GPUS=` (must equal the visible GPU count).
- `custom.vision.total_pixels` if fill-level judgement needs more resolution.
- `lora.target_modules` to add `gate_proj,up_proj,down_proj` for more capacity.
- `train.optm_lr` if loss plateaus early (raise) or spikes (lower).

## How rendering works

```bash
python cluster/render_spec.py --template specs/cosmos3_nano_lora_sft.yaml --out $RESULTS/train_spec.yaml \
   --model $MODEL --train-root $TRAIN --val-root $VAL --results $RESULTS --gpus 4 --epochs 5 --experiment-name $RUN
```

sets `policy.model_name_or_path`, `parallelism.dp_shard_size`, `train.epoch`, `train.output_dir`,
`results_dir`, `logging.experiment_name`, the `custom.*` paths, then writes YAML and a TOML produced by a small
dependency-free emitter (`to_toml`) that we verified round-trips to the identical dict as the YAML.
