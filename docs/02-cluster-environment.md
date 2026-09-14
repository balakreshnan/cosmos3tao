# 02 · Cluster environment

Covers README steps 0–3 and the concepts every later step relies on.

## SLURM in two minutes

SLURM is the batch scheduler. You never SSH to a GPU node; you ask SLURM for one.

| Command | What it does | Used in this repo |
| --- | --- | --- |
| `sbatch script` | Queue a batch script; returns a job id; runs when a node frees up | training, evaluation |
| `srun cmd` | Run one command in an allocation, blocking; inside `sbatch` it launches the container step | setup (download, import), prepare, inside sbatch |
| `squeue -u $USER` | Your pending/running jobs (`PD` pending, `R` running) | monitoring |
| `sacct -j ID` | History and exit code of a finished job | debugging |
| `scancel ID` | Cancel | |
| `sinfo --summarize` | Partitions and idle node counts (`A/I/O/T` = allocated/idle/other/total) | choosing a partition |

Every job must name an **account** (`--account=general_sa`) and a **partition** (`--partition=gb200`). Nodes on
these clusters are allocated whole: there is no `--gres=gpu:N`, you get all 4 GPUs and restrict yourself with
`CUDA_VISIBLE_DEVICES`. A job name convention `account-<subproject>.<details>` is enforced on some clusters;
ours is `general_sa-cosmos3.tao-finetune`.

Wall-time limits: 5 h on `gb200`, 8 h on `gb200-backfill`. Backfill partitions let short jobs slip into gaps, so
`--partition=gb200-backfill --time=3:00:00` often starts sooner.

## Pyxis and Enroot: containers under SLURM

**Enroot** unpacks a Docker image into a squashfs file (`.sqsh`). **Pyxis** is a SLURM plugin that adds
`--container-image`, `--container-mounts`, `--container-env` to `srun`. So

```bash
srun --container-image=/path/img.sqsh --container-mounts=/lustre/...:/tao-workspace bash -c '...'
```

starts the command inside the image with the Lustre directory visible at `/tao-workspace`. Flags we use:

| Flag | Why |
| --- | --- |
| `--container-image` | the `.sqsh` file (fast) or `nvcr.io#nvidia/tao/tao-toolkit:7.0.1-cosmos-rl` (pulls every time) |
| `--container-mounts=$WORK:/tao-workspace,$REPO:/tao-repo:ro` | data/models/results read-write, code read-only. **Never mount over `/workspace`**: the image keeps its cosmos-rl package there |
| `--container-mount-home` | your `$HOME` inside, for caches and credentials |
| `--no-container-remap-root` | run as yourself, so output files are owned by you |
| `--container-env=A,B` | forward named environment variables |
| `--mpi=pmix` | cluster convention for multi-process steps |

Pulling from `nvcr.io` needs credentials in `~/.config/enroot/.credentials`:

```
machine nvcr.io login $oauthtoken password <NGC API key>
```

The literal string `$oauthtoken` is the username. Keep the file mode 600.

## CPU architecture matters

lyris/theia/hecate GB200/GB300 nodes are **arm64** (Grace CPUs); ptyche nodes are x86_64. A `.sqsh` is
architecture-specific because Enroot pulls the image variant matching the machine doing the import. The TAO
7.0.1 cosmos-rl image is multi-arch (amd64 and arm64), so both work, but you need the right file. Our scripts
name it `…-cosmos-rl-<arch>.sqsh` and probe the *compute* partition's arch with a 3-minute `srun uname -m`,
because the login node may be x86 while the GPU nodes are arm64 (that exact mismatch bit us on hecate).

## GPU architecture matters more

| GPU | SM | Works with TAO 7.0.1 image? |
| --- | --- | --- |
| A100 / H100 (x86) | 80 / 90 | yes |
| GB200 (arm64) | 100 | **yes, verified** |
| GB300 (arm64) | 103 | **no**: the image's NVRTC (runtime CUDA compiler) rejects `sm_103`; the first JIT-compiled kernel dies with `nvrtc: error: invalid value for --gpu-architecture` |

Use partition `gb200`. Avoid theia `gb300`, hecate `batch-xdr`, ptyche `a02grace`.

## Lustre: the shared filesystem

`/lustre/fsw/<account>/<user>` is visible from login and compute nodes. Everything big lives there:

```
$LUSTRE_DIR/cosmos3tao/
  repo/      git checkout (this repository)
  venv/      Python venv for login-node helpers (dataset generation, hf download)
  data/      tube_inspection/{train,val}
  models/    Cosmos3-Nano (Omni), Cosmos3-Nano-qwen3vl (converted), Qwen3-VL-8B-Instruct-donor (configs only)
  sqsh/      container images
  hf_cache/  Hugging Face cache used inside the container (offline mode)
  logs/      SLURM stdout/stderr per job
  results/   one folder per training run
```

Login nodes have tight memory limits: the 33 GB model download was OOM-killed there, which is why setup runs it
as a SLURM step.

## Environment variables the scripts read

| Variable | Default | Used by |
| --- | --- | --- |
| `ACCOUNT` | `general_sa` | all `srun`/`sbatch` |
| `PARTITION` | `batch` (setup/prepare) — set it to `gb200` | setup, prepare, eval |
| `LUSTRE_DIR` | `/lustre/fsw/$ACCOUNT/$USER` | all |
| `WORK` | `$LUSTRE_DIR/cosmos3tao` | all |
| `TAO_IMAGE` | `nvcr.io/nvidia/tao/tao-toolkit:7.0.1-cosmos-rl` | all |
| `GPUS` | 4 | train (capped at visible GPUs) |
| `EPOCHS` | 5 | train |
| `RUN` | newest `results/cosmos3_nano_tube_lora_*` | eval |
| `LIMIT` | 300 | eval sample count |
| `HF_TOKEN` | unset | only if the HF repo were gated (it is not) |

`sbatch` does **not** inherit your shell by default in a reliable way, so the README passes
`--export=ALL,ACCOUNT=…,LUSTRE_DIR=…` explicitly. A prefix assignment like `RUN=x sbatch … RUN=$RUN` does not
work: `$RUN` is expanded by your shell before the prefix applies. Export first.

## Where the SLURM log goes

`--output=$LUSTRE_DIR/cosmos3tao/logs/%x-%j.out` expands `%x` to the job name and `%j` to the job id, giving
`logs/general_sa-cosmos3.tao-finetune-3044273.out`. The file only appears once the job starts running, which is
why the monitor one-liner in the README waits for it in a loop.
