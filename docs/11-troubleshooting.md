# 11 · Troubleshooting

Every problem hit while bringing the pipeline up, in the order they appeared, with the root cause and the fix
that is now in the code. If you see one of these, the fix is already applied; pull the repo.

| # | Symptom | Root cause | Fix (now in code) |
| --- | --- | --- | --- |
| 1 | `hf download … Killed` at ~12 GB on the login node | login-node memory cgroup + Xet download backend buffering | `ptyche_setup.sh` downloads inside `srun --mem=64G` with `HF_HUB_DISABLE_XET=1`, resumable |
| 2 | `sbatch: error: Invalid generic resource (gres) specification` | cluster defines no `gpu` GRES; nodes are whole | removed `--gres`; container limits itself via `CUDA_VISIBLE_DEVICES=0..GPUS-1` |
| 3 | `No partition specified or system default partition` | `$PARTITION` empty in a new shell | always re-export variables (README step 1) |
| 4 | Job pending for hours with `Reason=Priority` | busy partition | `--partition=gb200-backfill --time=3:00:00`, or another cluster |
| 5 | `cosmos_rl: command not found` | guessed launcher name | launcher is `cosmos-rl --config spec.toml /opt/cosmos_rl/tao_sft_example.py`, found by reading TAO's client source |
| 6 | `model type cosmos3_omni … Transformers does not recognize this architecture` | HF checkpoint is the Omni bundle | conversion to Qwen3-VL (chapter 05), `ptyche_prepare_model.sh` |
| 7 | `FileNotFoundError: …donor/config.json` during conversion | `hf download --include a b c` treated `b c` as positional filenames | filenames passed positionally; existence check added |
| 8 | Converter ran with `/usr/lib/python3.12` (no torch) | `source activate` inside `bash -lc` did not take effect | explicit `/opt/venv/cosmos_rl/bin/python` |
| 9 | `train_batch_per_replica(1) of sft must be divisible by mini_batch(4)` | doc example values | spec uses batch 8 / mini-batch 2 |
| 10 | Learning rate arrived as a string | YAML parses `2e-5` as text | write `2.0e-5`, `1.0e-8` |
| 11 | `nvrtc: error: invalid value for --gpu-architecture (-arch)` at the first CUDA kernel | GB300 (`sm_103`) not supported by the image's CUDA/NVRTC | run on GB200 (`gb200`) or x86 H100; documented in README |
| 12 | sqsh named `-x86_64.sqsh` but compute nodes are arm64 | login node arch ≠ compute arch | setup probes the partition with `srun uname -m`; rename existing file if needed |
| 13 | Eval job: `RUN: set RUN=<…>` | `RUN=x sbatch … RUN=$RUN` expands `$RUN` before the prefix assignment | `export RUN=…` first; sbatch defaults to newest run |
| 14 | Eval sbatch exited without printing Python logs | `set -e` aborted on `wait` returning non-zero | `set +e` around the waits; both logs always printed |
| 15 | Eval fine-tuned: `'NoneType' object has no attribute 'keys'` from PEFT | cosmos-rl's `adapter_config.json` has non-standard null fields | manual LoRA merge with suffix key matching (chapter 09) |
| 16 | Eval baseline: `cuDNN Frontend error: No valid execution plans built` | cuDNN fused SDPA on GB200 for these shapes | `torch.backends.cuda.enable_cudnn_sdp(False)` |
| 17 | `git add -A` swept the 100 MB dataset into a commit | dataset regenerated locally inside the repo | `data/tube_inspection/` and `results/` are git-ignored |
| 18 | `scp … Permission denied (publickey)` | cluster requires the MFA login name | `scp <user>-mfa@login-lyris.nvidia.com:…` |
| 19 | PowerShell one-liner created folders named `scp`, `-r` | command pasted into `cmd.exe`, where `;` does not separate commands | run in PowerShell; one command per line |
| 20 | `No space left on device` writing to `~` | 50 GB home quota filled by `~/.cache` (Enroot layers, Hugging Face, pip, cosmos-rl dataset cache) | `rm -rf ~/.cache/{huggingface,pip,enroot,cosmos}`; set `ENROOT_CACHE_PATH`, `ENROOT_DATA_PATH`, `HF_HOME`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME` to Lustre in `~/.bashrc`; the training container now sets `XDG_CACHE_HOME` to Lustre |
| 21 | `scp … <user>/…: No such file or directory` | `<user>` placeholder left in the Lustre path | Lustre path uses the plain username; only the SSH login has `-mfa` |
| 22 | `readlink -f output/best/safetensors` empty on the login node | symlinks were created inside the container and point to `/tao-workspace/...` | use `output/<timestamp>/safetensors/epoch_N` directly |

## General debugging recipe

1. `sacct -j <id> --format=JobID,State,Elapsed,ExitCode` — did it run, how long, exit code.
2. `.err` file: SLURM/srun errors (`Exited with exit code 1`, unbound variable messages).
3. `.out` file: our preflight prints, then cosmos-rl output including Python tracebacks.
   `grep -n -E "Error|error:|Traceback" file | head`.
4. For evaluation, the per-model logs `results/<run>/eval_finetuned.log` / `eval_baseline.log`.
5. Reproduce interactively: `bash cluster/ptyche_interactive.sh`, then run the failing command by hand inside the
   container.

## Reading a cosmos-rl crash

cosmos-rl prints every rank's traceback and, for NVRTC failures, the entire generated CUDA source (thousands of
lines). The meaningful line is the last `Error`/`RuntimeError` per rank, and the `Root Cause (first observed
failure)` block from torchrun at the end. Everything else is noise.
