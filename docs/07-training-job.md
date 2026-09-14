# 07 · The training job

README step 6. Three files cooperate:

| file | runs on | role |
| --- | --- | --- |
| [`cluster/ptyche_train.sbatch`](../cluster/ptyche_train.sbatch) | compute node, host side | SLURM header, picks the `.sqsh`, mounts, starts one container step |
| [`cluster/train_in_container.sh`](../cluster/train_in_container.sh) | inside the container | preflight, render spec, launch `cosmos-rl` |
| [`cluster/render_spec.py`](../cluster/render_spec.py) | inside the container | YAML template → run-specific YAML + TOML |

## 1. `ptyche_train.sbatch`

```
#SBATCH --account=general_sa --partition=36x2-a01r --nodes=1 --ntasks-per-node=1 --time=5:00:00
#SBATCH --job-name=general_sa-cosmos3.tao-finetune --no-requeue
```

The header holds ptyche defaults; the README's `sbatch --account … --partition …` flags override them, which is
why the same file works on every cluster. There is deliberately **no `--gres`**: these clusters have no GPU
GRES and allocate whole nodes.

Body:

1. Resolve `WORK`, `REPO_DIR` (= `$SLURM_SUBMIT_DIR`, the repo you ran `sbatch` from), `IMAGE`, and the sqsh
   for this node's `uname -m`; fall back to the registry image if the file is missing.
2. `RUN=cosmos3_nano_tube_lora_$SLURM_JOB_ID` → `results/$RUN/`.
3. Mounts: `$WORK:/tao-workspace,$REPO_DIR:/tao-repo:ro`.
4. One `srun` with Pyxis flags (chapter 02) running `bash /tao-repo/cluster/train_in_container.sh`, forwarding
   `GPUS,EPOCHS,RUN,TAO_TRAIN_CMD,HF_TOKEN,WANDB_MODE`.

The first line of the SLURM log confirms all of this: `== job 3044273 node=lyris0288 image=…sqsh gpus=4 run=…`.

## 2. `train_in_container.sh` — preflight

Inside the image, as your user:

1. `GPUS` is capped at the visible GPU count and `CUDA_VISIBLE_DEVICES=0,1,2,3` is set.
2. Prints `nvidia-smi` (4 × "NVIDIA Graphics Device, 286524 MiB" on GB200), Python version, and which
   launchers exist on PATH (`cosmos-rl -> /opt/venv/cosmos_rl/bin/cosmos-rl`).
3. Requires the **converted** model at `/tao-workspace/models/Cosmos3-Nano-qwen3vl` and prints its
   `model_type` (`qwen3_vl`). If only the Omni download exists it exits with instructions to run
   `ptyche_prepare_model.sh`.
4. Counts train/val samples (4000 / 1000).
5. Renders the spec (chapter 06) into `results/$RUN/train_spec.yaml` and `.toml`, printing the `policy` and
   `custom` blocks so the log documents exactly what ran.
6. Sets offline Hugging Face mode (`HF_HUB_OFFLINE=1`, cache on Lustre) so compute nodes never need internet,
   `WANDB_MODE=disabled`, and the single-node rendezvous variables.

## 3. Launch

```bash
source /opt/venv/cosmos_rl/bin/activate
cp $RESULTS/train_spec.toml $RESULTS/spec.toml
cosmos-rl --config $RESULTS/spec.toml /opt/cosmos_rl/tao_sft_example.py
```

This is exactly what TAO's fine-tuning microservice runs; we found it in the TAO client package
(`vlm_entrypoint.py`: `f"{network}{suffix} --config {config_path} {script_path}"` with
`script_path=/opt/cosmos_rl/tao_sft_example.py`). Two parts:

- **`cosmos-rl`** is the launcher: it starts a **controller** (a small web service plus Redis on port 12800
  that coordinates replicas) and then `torchrun`s one **policy worker** per visible GPU.
- **`tao_sft_example.py`** is TAO's dataset hook, the positional "custom script" cosmos-rl expects. It reads
  `custom.dataset.*` from the TOML, loads `annotations.json`, opens each image from `media_path`, builds the
  Qwen3-VL chat prompt with the image and the human turn, and returns the `gpt` turn as the supervised target.
  It also installs TAO's status logger.

`TAO_TRAIN_CMD` and `TAO_SFT_HOOK` environment variables let you override either half without editing code.

## 4. What happens inside cosmos-rl, in log order

| log line | meaning |
| --- | --- |
| `Building 1-D device mesh with ['dp_shard'], [4]` | FSDP group of 4 GPUs formed |
| `Using custom validation dataset from …/val/annotations.json` | the hook found our val set |
| `No default validation data packer found for qwen3_vl, using HFVLMDataPacker` | harmless warning |
| `Training epoch 1/5` | epoch boundary (also used by the report) |
| `Step: 1/625, Loss: 0.28958, Grad norm: 0.60693, Iteration time: 4.93s, optimizer/lr_model.language_model: 1.6e-07` | one optimizer step: 8 samples, 4 micro-batches. 625 = 5 epochs × 4000 samples ÷ 8 ÷ 4 GPUs… ⇒ 125 steps per epoch |
| `[SFT] Triggering epoch-based validation at step 125 (end of epoch 0)` → `Validation loss: 0.0971 …` | per-epoch validation on all 1000 val samples (248 per rank) |
| `checkpoint saved successfully at …/checkpoints/epoch_1/policy` | cosmos-rl native checkpoint (sharded) |
| `Best checkpoint updated to epoch_1 with score: 0.0971` | best-so-far by validation loss; `output/best/` symlinks follow it |
| `Removed old checkpoint … epoch_2` | `max_keep: 3` at work |
| `Cosmos-RL SFT training completed successfully` / `Job SUCCESS` | done; `== training exit code: 0` follows from our wrapper |

Step time was steady at ~2.0 s, so one epoch takes ~4.3 minutes and the whole run 24 minutes plus 3 minutes of
model loading.

## 5. Parallelism, concretely

With `dp_shard_size: 4` every GPU holds a quarter of every weight and optimizer tensor (FSDP). For a forward
pass, each layer's shards are all-gathered just in time, used, and freed. Each GPU sees different samples (2 per
micro-batch); gradients are reduced across the 4 GPUs. `train_batch_per_replica: 8` means 8 samples per
optimizer step **per GPU**, so 32 samples per step globally — the 125 steps/epoch figure = 4000 ÷ 32.

## 6. Interactive variant

`cluster/ptyche_interactive.sh` opens the same allocation as `srun --pty bash` inside the container. From there
`bash /tao-repo/cluster/train_in_container.sh` runs the identical procedure with live output; useful for
debugging spec changes without waiting on the batch queue.
