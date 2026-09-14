# 08 · Monitoring and results

README steps 7–8.

## While the job is queued

```bash
squeue -u $USER                                   # ST = PD pending, R running
squeue -j <id> -o "%.10i %.8T %.20S %R"           # estimated START_TIME and REASON
squeue -j <id> --start
sinfo --summarize                                 # idle nodes per partition (A/I/O/T)
sprio -u $USER                                    # your priority
```

`Reason = Priority` means others are ahead; `Resources` means you are next. Estimates are conservative and
usually improve. If it is hours away, cancel and resubmit to `gb200-backfill` with `--time=3:00:00`.

The SLURM log file does not exist until the job starts. The README one-liner waits for it:

```bash
J=<id>; LOG=$LUSTRE_DIR/cosmos3tao/logs/general_sa-cosmos3.tao-finetune-$J.out; until [ -f "$LOG" ]; do sleep 30; done; tail -f "$LOG"
```

The job keeps running if you log out. On return, re-export the variables and `tail -f` the same file.

## While it runs

The log has three phases: our preflight (chapter 07 §2), cosmos-rl startup (model discovery, dataset
creation, ~3 minutes), then one `Step:` line per optimizer step from every rank. Lines are prefixed
`[rankN]:`; rank 0 is authoritative. Useful filters:

```bash
grep -E "^\[rank0\].*Step: [0-9]+/" $LOG | tail -5          # latest steps
grep -E "Validation loss|Best checkpoint" $LOG               # per-epoch results
grep -c "Step:" $LOG
grep -n -E "Error|error:|Traceback" $LOG | head              # first sign of trouble
```

Live per-step output also goes to `results/<run>/train.log` (our `tee`).

## After it ends

```bash
sacct -j <id> --format=JobID,JobName%32,State,Elapsed,ExitCode
tail -n 30 $LOG
```

`State=COMPLETED` with `== training exit code: 0` is success. `FAILED` with exit 1 means the Python step died;
the traceback is in the `.out` (cosmos-rl prints tracebacks to stdout) and `srun: error: … Exited with exit
code 1` in the `.err`. Chapter 11 lists the ones we met.

## The results folder

```
results/cosmos3_nano_tube_lora_3044273/
  train_spec.yaml      rendered spec (human-readable)
  train_spec.toml      same, as TOML
  spec.toml            the file cosmos-rl actually read
  train.log            complete console log, all ranks (~270 KB for this run)
  output/
    20260913221703/                         cosmos-rl timestamped output dir
      checkpoints/epoch_3/policy/            native sharded checkpoint (resume from here)
      checkpoints/epoch_4/policy/
      checkpoints/epoch_5/policy/
      safetensors/epoch_3/                   PEFT-style LoRA export
      safetensors/epoch_4/
      safetensors/epoch_5/ (adapter_config.json + adapter safetensors)
    best/
      best_score.json                        {"score": 0.0353, ...}
      safetensors -> ../20260913221703/safetensors/epoch_5      symlink to the best epoch
      checkpoint  -> …/checkpoints/epoch_5/policy
```

Only the last `max_keep = 3` epochs are retained. `output/best/safetensors` is what the evaluator (chapter 09)
and any inference server should load. The adapter export contains `adapter_config.json` (`peft_type: LORA`,
`r: 16`, `lora_alpha: 32`, targets q/k/v/o, `base_model_name_or_path` pointing at the converted checkpoint) and
the LoRA A/B matrices for 144 modules. It does **not** contain the base weights; those stay in
`models/Cosmos3-Nano-qwen3vl`.

## Getting small files to the laptop

Windows PowerShell; the MFA username differs from your Lustre username:

```powershell
scp <user>-mfa@login-lyris.nvidia.com:/lustre/fsw/general_sa/<user>/cosmos3tao/results/<run>/train.log results\
scp "<user>-mfa@login-lyris.nvidia.com:/lustre/fsw/general_sa/<user>/cosmos3tao/results/<run>/train_spec.*" results\
```

Do not `scp` the checkpoint folders casually (several GB each). For `rsync` use
`--exclude checkpoints --exclude safetensors`.

## Resuming

Set `train.resume: true` in the spec and point `train.output_dir`… — cosmos-rl resumes from the newest native
checkpoint in the output dir. Not needed for a 24-minute run, but relevant if you raise epochs beyond the 5-hour
wall limit.
