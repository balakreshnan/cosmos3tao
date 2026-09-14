# Glossary

| Term | Meaning here |
| --- | --- |
| **TAO** | NVIDIA Train Adapt Optimize toolkit. Provides containers and recipes for fine-tuning vision models; for Cosmos Reason it wraps cosmos-rl. We use its container and its dataset hook directly. |
| **FTMS** | TAO Fine-Tuning Micro-Services: TAO's REST API deployment. Not used here (no deployment available); we run the same commands FTMS would run. |
| **cosmos-rl** | NVIDIA's training framework for Cosmos Reason models (SFT and RL). CLI `cosmos-rl --config x.toml <dataset script>`. Runs a controller + policy workers via torchrun. |
| **Cosmos3-Nano** | 16B-parameter open "omni" model: 8B reasoner (Qwen3-VL-based VLM) + 8B video generator. We fine-tune only the reasoner. |
| **Omni checkpoint / `cosmos3_omni`** | The Hugging Face format bundling reasoner, generator, action and audio heads under one custom architecture. Not loadable by stock Transformers. |
| **Qwen3-VL** | Alibaba's vision-language architecture. The Cosmos3 reasoner is a Qwen3-VL-8B; we re-key the weights into that layout. |
| **Architecture donor** | `Qwen/Qwen3-VL-8B-Instruct`, whose `config.json` supplies the Qwen3-VL config (including mrope settings) for the converted checkpoint. Its weights are not used. |
| **LoRA** | Low-Rank Adaptation: train small matrices A (r×in) and B (out×r) added to frozen weights as W + (α/r)·BA. Here r=16, α=32 on q/k/v/o projections. |
| **Adapter** | The saved LoRA matrices plus `adapter_config.json` (PEFT format). ~100 MB. Needs the base model to run. |
| **Merge** | Adding (α/r)·BA into W so the model runs without LoRA machinery. Done in memory by the evaluator. |
| **SFT** | Supervised fine-tuning: minimize next-token loss on reference answers. |
| **FSDP** | Fully Sharded Data Parallel: each GPU holds a shard of every parameter/optimizer state, gathers on demand. `dp_shard_size: 4`. |
| **Epoch / step / micro-batch** | Epoch = one pass over 4000 samples. Step = one optimizer update (8 samples per GPU × 4 GPUs = 32). Micro-batch = 2 samples per forward pass, 4 accumulated per step. 125 steps per epoch. |
| **Validation loss** | Same loss on the 1000 held-out val samples, computed at each epoch end. Used to pick the best checkpoint. |
| **Accuracy** | Fraction of evaluation questions answered correctly (chapter 09); distinct from loss. |
| **LLaVA format** | JSON list of `{images, conversations:[{from:human,value:"<image>\n…"},{from:gpt,value:…}]}`; TAO's `vlm/llava` dataset type. |
| **SLURM** | Cluster scheduler: `sbatch`, `srun`, `squeue`, `sacct`, `scancel`, `sinfo`. |
| **Partition** | A named pool of nodes with its own limits (`gb200`, `gb200-backfill`, …). |
| **Account** | SLURM charge account (`general_sa`); required on every job. |
| **Pyxis** | SLURM plugin adding `--container-image` and friends to `srun`. |
| **Enroot** | Container runtime Pyxis uses; `enroot import` converts a Docker image to a `.sqsh`. |
| **`.sqsh`** | Squashfs archive of a container image, architecture-specific, fast to start. |
| **Lustre** | The cluster's shared parallel filesystem, `/lustre/fsw/<account>/<user>`. |
| **arm64 / aarch64 vs x86_64** | CPU architectures. Grace-based GB200/GB300 nodes are arm64; container images and `.sqsh` files must match. |
| **SM (sm_100, sm_103)** | GPU compute-capability generation. GB200 = sm_100 (supported by the TAO 7.0.1 image), GB300 = sm_103 (not). |
| **NVRTC** | CUDA runtime compiler PyTorch uses to JIT some kernels; it must know the GPU's SM. |
| **SDPA / cuDNN attention** | PyTorch scaled-dot-product attention backends; cuDNN's failed on GB200 in eval, so it is disabled there. |
| **Hugging Face offline mode** | `HF_HUB_OFFLINE=1`: never contact the Hub; everything must already be on Lustre. Compute nodes may lack internet. |
| **Xet** | New Hugging Face download backend; memory-hungry on login nodes, disabled with `HF_HUB_DISABLE_XET=1`. |
| **MFA login** | `<user>-mfa@login-<cluster>.nvidia.com`: the SSH identity with device-code authentication. Lustre paths still use `<user>`. |
