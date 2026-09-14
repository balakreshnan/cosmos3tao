# 10 · The HTML report

README step 9 (+ step 10 for the accuracy section). File: [`report/make_report.py`](../report/make_report.py).
Example output: [`report/run_lyris_gb200_3044273.html`](../report/run_lyris_gb200_3044273.html).

## What it consumes

| input | flag | required |
| --- | --- | --- |
| `train.log` | `--log` (or `--results <dir>`) | yes |
| `train_spec.yaml` / `spec.toml` | `--spec` | recommended (config table, KPIs) |
| evaluation dirs with `metrics.json` + `predictions.jsonl` | `--eval "label=dir"` repeatable; first = fine-tuned, second = baseline | optional |
| validation images | `--media data/tube_inspection/val` | optional, enables the failure gallery |
| `ground_truth.json` | `--ground-truth` | optional, enables failure diagnoses |
| header text | `--cluster "…" --job 3044273 --title "…"` | optional |

## Parsing the log

`parse_log` walks every line, keeps only `[rank0]` lines (all ranks print the same metrics), and recognizes:

- `Step: N/M, Loss: x, Grad norm: x, Iteration time: 2.01s, optimizer/lr_…: x` → per-step metrics. Fragments
  are split on commas and matched as `name: number`, so multi-word names like `Grad norm` survive.
- `[SFT] Validation loss: x for train step N/M, epoch E` → `val/loss` at step N
- `Training epoch k/K` → epoch markers
- `Step: N, checkpoint saved successfully at …/epoch_k/policy` → checkpoint events
- `Best checkpoint updated to epoch_k with score: x` → best marker
- timestamps → wall time and per-epoch minutes

## Analysis (`analyze`)

- **Per-epoch table**: epoch boundaries are taken from the validation steps (each epoch ends with a validation),
  so train-mean, end-of-epoch loss, validation loss and minutes line up exactly.
- **KPIs**: best validation loss (with % vs first epoch), steady-state train loss (mean of final 10 % of steps,
  vs first 5 %), generalization gap (final val − train; "healthy" if < 25 % + 0.01), steps, wall time and
  steps/min, GPU count/dtype.
- **Narrative**: a few sentences assembled from the numbers (model, LoRA config, loss trajectory, whether
  validation improved monotonically, convergence judgement: last-epoch change under 5 % ⇒ converged).
- **Insights**: LR warm-up step and peak, gradient-norm median and spikes, step-time stability, checkpoints,
  overfitting check, next step.

## Accuracy section (`build_accuracy`)

Only when `--eval` is given. Horizontal bars per question type (fine-tuned solid, baseline thin), a
report-field table with both models, and a **failure gallery**: every incorrect fine-tuned prediction with a
680-px JPEG thumbnail embedded as base64, expected vs predicted, and a diagnosis derived from ground truth
(`missed pos 5: overfill (clear, fill 61%)`). It also derives a pattern insight (which status/content dominates
the misses) and prepends an accuracy KPI and sentence to the summary.

## The page

Single HTML file, no external assets, ~330 KB with the gallery. Sections: hero banner with status pill and run
metadata · executive summary · KPI tiles · task accuracy · loss-over-training (smoothed line with faint raw
trace, validation points, shaded epoch bands) · per-epoch table · learning rate / gradient norm / step time ·
insights · metrics-by-step table (toggle) · configuration · reproduce commands. Controls: smoothing window, log
y-axis, data-table toggle, light/dark theme; crosshair tooltips on every chart; print-friendly.

Charts are hand-rolled SVG following the dataviz conventions in this workspace (one y-axis per chart, fixed
categorical palette, thin marks, legend for ≥ 2 series, text in ink colors, no dual axes).

## Commands (laptop, PowerShell)

```powershell
.\.venv\Scripts\python.exe report\make_report.py --log results\train.log --spec results\train_spec.yaml `
  --out report\run_lyris_gb200_3044273.html --cluster "lyris · gb200 · 1 node × 4 GB200 (arm64)" --job 3044273 `
  --eval "fine-tuned (LoRA epoch 5)=results\eval_finetuned" --eval "base model (zero-shot)=results\eval_baseline" `
  --media data\tube_inspection\val --ground-truth data\tube_inspection\val\ground_truth.json
```

The console prints which metrics were found; if it says none, the log format changed — send the first
`Step:` lines.

## Adapting to another run

Nothing is hard-coded to this task except the friendly names for question categories (`CAT_LABEL`) and report
fields (`FIELD_LABEL`); unknown categories fall back to their raw names. The narrative reads model name, LoRA
settings, epochs and GPU count from the spec.
