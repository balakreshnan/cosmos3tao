# 04 · Setup script: `cluster/ptyche_setup.sh`

README step 4. Run once per cluster; every stage is skipped if its output already exists, so re-running is safe
and is the way to resume after an interruption.

## Inputs

Environment: `ACCOUNT`, `PARTITION`, `LUSTRE_DIR` (or `WORK`), optional `TAO_IMAGE`, `MODEL_ID`, `N_TRAIN`,
`N_VAL`, `HF_TOKEN`. It creates `$WORK/{data,models,results,sqsh,logs}`.

## Stage 1 · Python venv on Lustre

```bash
python3 -m venv "$WORK/venv"; pip install pillow numpy pyyaml huggingface_hub tqdm
```

A small login-node environment for the generator and the `hf` download CLI. It is **not** used for training;
the container has its own Python (`/opt/venv/cosmos_rl`).

## Stage 2 · Dataset

```bash
python repo/dataset/generate_tube_dataset.py --out $WORK/data/tube_inspection --train 800 --val 200
```

See chapter 03. Skipped when `train/annotations.json` exists.

## Stage 3 · Model download, as a SLURM step

```bash
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HOME="$WORK/hf_cache"
srun -A $ACCOUNT -p $PARTITION -N1 -n1 --cpus-per-task=16 --mem=64G --time=02:00:00 \
     --job-name=$ACCOUNT-cosmos3.hf-download \
     $WORK/venv/bin/hf download nvidia/Cosmos3-Nano --local-dir $WORK/models/Cosmos3-Nano \
        --exclude 'assets/*' --exclude 'images/*' --max-workers 8
```

Why a SLURM step: the first attempt on the login node was killed at 12 GB by the node's memory limit. The new
`huggingface_hub` Xet backend buffers aggressively; disabling it and running on a compute node with 64 GB fixed
it. `--exclude` skips demo videos. The download is resumable; re-running setup continues where it stopped.

What arrives (33 GB): `config.json` with `model_type: cosmos3_omni`, `model.safetensors.index.json` pointing
at `transformer/diffusion_pytorch_model-0000N-of-00007.safetensors` and `vision_encoder/model.safetensors`,
tokenizer files, `preprocessor_config.json`, plus generator-only parts (`vae/`, `scheduler/`,
`sound_tokenizer/`). Chapter 05 explains which of these we keep.

## Stage 4 · Container to `.sqsh`

```bash
ARCH=$(srun -A $ACCOUNT -p $PARTITION -N1 -n1 --time=00:03:00 uname -m | tail -1)   # e.g. aarch64
SQSH=$WORK/sqsh/nvcr.io_nvidia_tao_tao-toolkit_7.0.1-cosmos-rl-$ARCH.sqsh
srun -A $ACCOUNT -p $PARTITION -N1 -n1 --cpus-per-task=4 --time=01:00:00 \
     enroot import -o $SQSH docker://nvcr.io#nvidia/tao/tao-toolkit:7.0.1-cosmos-rl
```

`enroot import` authenticates with `~/.config/enroot/.credentials`, fetches the ~56 image layers for the
compute node's CPU architecture, and packs them into one ~10 GB squashfs file. Starting a job from a `.sqsh`
takes seconds; starting from the registry re-downloads the image each time. The arch probe exists because the
login node's `uname -m` can differ from the compute nodes'.

The `docker://nvcr.io#nvidia/...` form is Enroot's syntax: `#` separates registry from repository.

## Output

```
Setup complete.
  dataset : $WORK/data/tube_inspection/{train,val}
  model   : $WORK/models/Cosmos3-Nano
  sqsh    : $WORK/sqsh/nvcr.io_nvidia_tao_tao-toolkit_7.0.1-cosmos-rl-aarch64.sqsh
```

plus a suggested `sbatch` command with the account/partition/paths filled in.

## Timing on lyris

venv 1 min · dataset 1 min · download 1–3 min from a compute node · import 2 min.

## If something goes wrong

| Symptom | Fix |
| --- | --- |
| `hf download` killed | you are on an older script; pull and re-run, download now runs under `srun` |
| `srun: job queued and waiting for resources` for a long time | pick a partition with idle nodes (`sinfo --summarize`) via `PARTITION=` |
| `enroot import` authentication failure | check `~/.config/enroot/.credentials` line for `nvcr.io` |
| sqsh named `-x86_64` on an arm64 cluster | older script used the login node's arch; rename to `-aarch64.sqsh` |
