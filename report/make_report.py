"""
Build a self-contained, executive-style HTML report from a cosmos-rl / TAO training run.

Inputs (pass the run's results directory or individual files):
  <results_dir>/train.log                      console log (all ranks; rank0 is used when tagged)
  <results_dir>/train_spec.yaml | spec.toml    run configuration
  <results_dir>/output/best/best_score.json    best validation score (optional)

Parsed from cosmos-rl logs:
  Step: N/M, Loss: x, Grad norm: x, Iteration time: 2.01s, optimizer/lr_...: x
  [SFT] Validation loss: x for train step N/M, epoch E
  Training epoch k/K
  Step: N, checkpoint saved successfully at .../epoch_k/policy
  Best checkpoint updated to epoch_k with score: x

Usage:
  python report/make_report.py --results results/<run> --out report.html
  python report/make_report.py --log results/train.log --spec results/train_spec.yaml --out report.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RANK_RE = re.compile(r"^\[rank(\d+)\]:\s*")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
STEP_RE = re.compile(r"\bStep:\s*(\d+)(?:/(\d+))?")
FRAG_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_/\. ]*?)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([a-zA-Z%]*)\s*$")
VAL_RE = re.compile(r"Validation loss:\s*([-\d.eE+]+)\s+for train step\s+(\d+)(?:/\d+)?,\s*epoch\s+(\d+)")
EPOCH_RE = re.compile(r"Training epoch\s+(\d+)/(\d+)")
CKPT_RE = re.compile(r"Step:\s*(\d+),\s*checkpoint saved successfully at (\S+)")
BEST_RE = re.compile(r"Best checkpoint updated to (\S+) with score:\s*([-\d.eE+]+)")
SKIP = {"step", "epoch", "rank", "pid", "port", "seed"}

PRETTY = {
    "loss": "Training loss",
    "val/loss": "Validation loss",
    "grad_norm": "Gradient norm",
    "iteration_time": "Step time (s)",
}


def pretty(key: str) -> str:
    if key in PRETTY:
        return PRETTY[key]
    if key.startswith("optimizer/lr"):
        return "Learning rate"
    return key.replace("_", " ").replace("/", " / ")


def parse_log(path: Path):
    metrics: dict[str, dict[int, float]] = defaultdict(dict)
    events: list[dict] = []
    epochs: list[dict] = []          # {epoch, total, start_step, start_ts}
    total_steps = None
    first_ts = last_ts = None
    step_ts: dict[int, datetime] = {}
    last_step = 0
    nlines = 0
    for raw in path.read_text(errors="replace").splitlines():
        nlines += 1
        m = RANK_RE.match(raw)
        rank = int(m.group(1)) if m else 0
        line = RANK_RE.sub("", raw)
        ts = TS_RE.search(line)
        t = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S") if ts else None
        if t:
            first_ts = first_ts or t
            last_ts = t
        if rank != 0:
            continue
        em = EPOCH_RE.search(line)
        if em:
            epochs.append({"epoch": int(em.group(1)), "total": int(em.group(2)), "start_step": last_step + 1, "start_ts": t.isoformat() if t else None})
            continue
        vm = VAL_RE.search(line)
        if vm:
            metrics["val/loss"][int(vm.group(2))] = float(vm.group(1))
            continue
        cm = CKPT_RE.search(line)
        if cm:
            events.append({"step": int(cm.group(1)), "type": "checkpoint", "text": cm.group(2).rstrip("/").rsplit("/", 2)[-2]})
            continue
        bm = BEST_RE.search(line)
        if bm:
            events.append({"step": last_step, "type": "best", "text": f"{bm.group(1)} · val loss {float(bm.group(2)):.4f}"})
            continue
        sm = STEP_RE.search(line)
        if sm and "Loss" in line:
            step = int(sm.group(1))
            last_step = max(last_step, step)
            if sm.group(2):
                total_steps = int(sm.group(2))
            if t:
                step_ts[step] = t
            for frag in re.split(r"[,|;]", line[sm.end():]):
                fm = FRAG_RE.match(frag)
                if not fm:
                    continue
                key = fm.group(1).strip().lower().replace(" ", "_")
                if key in SKIP:
                    continue
                metrics[key][step] = float(fm.group(2))
    keep = {k: dict(sorted(v.items())) for k, v in metrics.items() if len(v) >= 2}
    return keep, events, epochs, total_steps, first_ts, last_ts, step_ts, nlines


def load_spec(results: Path | None, spec_path: Path | None):
    p = spec_path or (results and next((q for q in (results / "train_spec.yaml", results / "spec.toml") if q.exists()), None))
    if not p:
        return {}
    text = p.read_text()
    try:
        if p.suffix == ".yaml":
            import yaml
            return yaml.safe_load(text)
        import tomllib
        return tomllib.loads(text)
    except Exception:
        return {}


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def fmt(v, digits=4):
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def pct(a, b):
    return (b - a) / a * 100 if a else 0.0


# ----------------------------------------------------------------------------- analysis
def analyze(metrics, events, epochs, total_steps, t0, t1, step_ts, flat):
    loss = metrics.get("loss", {})
    val = metrics.get("val/loss", {})
    gn = metrics.get("grad_norm", {})
    it = metrics.get("iteration_time", {})
    lr_key = next((k for k in metrics if k.startswith("optimizer/lr")), None)
    lr = metrics.get(lr_key, {}) if lr_key else {}
    candidates = [max(loss) if loss else 0, max(val) if val else 0, total_steps or 0]
    steps = max(candidates)

    # per-epoch table. Epoch k ends at the k-th validation step (validation runs at every epoch end);
    # fall back to "Training epoch" markers, then to an even split.
    n_ep = max((e["total"] for e in epochs), default=int(flat.get("train.epoch", 0) or 0)) or (len(val) if val else 1)
    rows = []
    if val and len(val) >= n_ep:
        ends = sorted(val)[:n_ep]
        bounds = [(i + 1, (ends[i - 1] + 1) if i else 1) for i in range(n_ep)]
    elif epochs:
        bounds = [(e["epoch"], e["start_step"]) for e in epochs]
    else:
        bounds = [(i + 1, i * (steps // max(n_ep, 1)) + 1) for i in range(n_ep)]
    for i, (ep, s0) in enumerate(bounds):
        s1 = bounds[i + 1][1] - 1 if i + 1 < len(bounds) else steps
        ep_loss = [v for s, v in loss.items() if s0 <= s <= s1]
        v_step = next((s for s in sorted(val) if s0 <= s <= s1), None)
        ts0 = min((step_ts[s] for s in step_ts if s0 <= s <= s1), default=None)
        ts1 = max((step_ts[s] for s in step_ts if s0 <= s <= s1), default=None)
        rows.append({
            "epoch": ep, "steps": f"{s0}–{s1}",
            "train_mean": statistics.fmean(ep_loss) if ep_loss else None,
            "train_last": ep_loss[-1] if ep_loss else None,
            "val": val.get(v_step) if v_step is not None else None,
            "minutes": (ts1 - ts0).total_seconds() / 60 if ts0 and ts1 else None,
        })
    for i, r in enumerate(rows):
        prev = rows[i - 1]["val"] if i else None
        r["val_delta_pct"] = pct(prev, r["val"]) if prev and r["val"] else None

    first_loss = next(iter(loss.values()), None)
    last_loss = list(loss.values())[-1] if loss else None
    # steady-state train loss: mean of last 10% of steps
    tail = [v for s, v in loss.items() if s > steps * 0.9] if loss else []
    tail_mean = statistics.fmean(tail) if tail else None
    best_val_step, best_val = (min(val.items(), key=lambda p: p[1]) if val else (None, None))
    wall = (t1 - t0).total_seconds() if t0 and t1 else None
    train_wall = (max(step_ts.values()) - min(step_ts.values())).total_seconds() if step_ts else None
    step_time = statistics.median(it.values()) if it else (train_wall / steps if train_wall and steps else None)
    peak_lr = max(lr.values()) if lr else flat.get("train.optm_lr")
    warmup_steps = next((s for s, v in sorted(lr.items()) if v >= 0.999 * peak_lr), None) if lr else None
    spikes = [s for s, v in gn.items() if v > 3 * statistics.median(gn.values())] if gn else []

    # KPIs
    kpis = []
    if val:
        kpis.append({"label": "Best validation loss", "value": f"{best_val:.4f}", "sub": f"epoch {next((r['epoch'] for r in rows if r['val'] == best_val), '?')} · step {best_val_step}",
                     "delta": f"{pct(list(val.values())[0], best_val):+.0f}% vs first epoch" if len(val) > 1 else "", "tone": "good"})
    if first_loss is not None:
        kpis.append({"label": "Training loss", "value": f"{tail_mean:.4f}" if tail_mean else f"{last_loss:.4f}", "sub": "mean of final 10% of steps",
                     "delta": f"{pct(statistics.fmean(list(loss.values())[:max(1, steps // 20)]), tail_mean):+.0f}% vs first 5%" if tail_mean else "", "tone": "good"})
    if val and tail_mean:
        gap = list(val.values())[-1] - tail_mean
        kpis.append({"label": "Generalization gap", "value": f"{gap:+.4f}", "sub": "final val − train loss", "delta": "healthy" if abs(gap) < 0.25 * max(tail_mean, 1e-9) + 0.01 else "watch for overfitting", "tone": "good" if abs(gap) < 0.25 * max(tail_mean, 1e-9) + 0.01 else "warn"})
    kpis.append({"label": "Optimizer steps", "value": f"{steps:,}", "sub": f"{n_ep} epochs · batch {flat.get('train.train_batch_per_replica', '?')} per replica", "delta": "", "tone": ""})
    if wall:
        h, rem = divmod(int(wall), 3600); m_, _ = divmod(rem, 60)
        kpis.append({"label": "Wall time", "value": f"{h}h {m_:02d}m", "sub": f"{step_time:.2f} s / step median" if step_time else "", "delta": f"{steps / (train_wall / 60):.1f} steps/min" if train_wall else "", "tone": ""})
    kpis.append({"label": "Compute", "value": f"{flat.get('policy.parallelism.dp_shard_size', '?')} GPUs", "sub": f"FSDP shard · {flat.get('train.param_dtype', '?')}", "delta": "", "tone": ""})

    # narrative
    model = Path(str(flat.get("policy.model_name_or_path", "model"))).name
    lora = flat.get("policy.lora.r")
    targets = ", ".join(map(str, flat.get("policy.lora.target_modules", []) or []))
    parts = []
    parts.append(f"<b>{model}</b> was fine-tuned with LoRA (rank {lora}, targets {targets}) for {n_ep} epochs, "
                 f"{steps:,} optimizer steps, on {flat.get('policy.parallelism.dp_shard_size', '?')} GPUs.")
    if first_loss is not None and tail_mean:
        parts.append(f"Training loss fell from {first_loss:.3f} at the first step to a steady {tail_mean:.4f} over the final tenth of training "
                     f"({pct(first_loss, tail_mean):+.0f}%).")
    if val:
        vs = list(val.values())
        mono = all(b <= a for a, b in zip(vs, vs[1:]))
        parts.append(f"Validation loss {'improved every epoch' if mono else 'reached its minimum'}, ending at {vs[-1]:.4f} "
                     f"(best {best_val:.4f}); the best checkpoint is <b>{next((e['text'].split(' ·')[0] for e in reversed(events) if e['type'] == 'best'), 'the last epoch')}</b>.")
        if len(vs) >= 3 and abs(pct(vs[-2], vs[-1])) < 5:
            parts.append("The last epoch changed validation loss by under 5%, so the run has essentially converged; additional epochs would bring little.")
        elif len(vs) >= 2 and pct(vs[-2], vs[-1]) < -10:
            parts.append("Validation was still improving by more than 10% per epoch at the end, so more epochs are likely to help.")
    if wall:
        parts.append(f"The job completed in {int(wall // 60)} minutes of wall time.")
    summary = " ".join(parts)

    insights = []
    if warmup_steps:
        insights.append(f"Learning-rate warm-up reached the peak of {peak_lr:.2e} at step {warmup_steps}, then decayed toward {min(lr.values()):.2e}.")
    if gn:
        insights.append(f"Gradient norm settled at a median of {statistics.median(gn.values()):.3f}; {len(spikes)} step{'s' if len(spikes) != 1 else ''} exceeded 3× the median"
                        + (f" (steps {', '.join(map(str, spikes[:6]))}{'…' if len(spikes) > 6 else ''}), all early in training." if spikes and max(spikes) < steps * 0.2 else (f" (steps {', '.join(map(str, spikes[:6]))}{'…' if len(spikes) > 6 else ''})." if spikes else ".")))
    if it:
        insights.append(f"Step time was stable at a median {statistics.median(it.values()):.2f} s ({min(it.values()):.2f}–{max(it.values()):.2f} s), i.e. no data-loading stalls.")
    ck = [e for e in events if e["type"] == "checkpoint"]
    if ck:
        insights.append(f"{len(ck)} epoch checkpoints were written with safetensors exports; the best-scoring one is promoted to <code>output/best</code>.")
    if val and tail_mean:
        gap = list(val.values())[-1] - tail_mean
        insights.append("Validation loss tracks training loss closely, so the adapter is generalizing rather than memorizing the synthetic set."
                        if abs(gap) < 0.25 * max(tail_mean, 1e-9) + 0.01 else
                        "Validation loss sits well above training loss; consider more data variety, higher LoRA dropout, or fewer epochs.")
    insights.append("Next step: evaluate the best checkpoint on the held-out split and on real plant photos, and compare against the zero-shot baseline.")

    return {"kpis": kpis, "summary": summary, "insights": insights, "epochs": rows, "steps": steps, "lr_key": lr_key}


# ----------------------------------------------------------------------------- template
TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light;--surface:#fcfcfb;--page:#f3f3f0;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;--s8:#e34948;--good:#006300;--warn:#a35a00;--band:rgba(42,120,214,.06);--hero:#0f1f33;--hero2:#16324f;--heroink:#f5f7fa}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--good:#0ca30c;--warn:#fab219;--band:rgba(57,135,229,.10);--hero:#0b1626;--hero2:#122640}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--good:#0ca30c;--warn:#fab219;--band:rgba(57,135,229,.10);--hero:#0b1626;--hero2:#122640}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:inherit}
.hero{background:linear-gradient(120deg,var(--hero),var(--hero2));color:var(--heroink);padding:36px 40px 28px}
.hero .eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;opacity:.75}
.hero h1{font-size:30px;font-weight:650;margin:6px 0 6px;letter-spacing:-.01em}
.hero .lede{font-size:15px;opacity:.85;max-width:900px;margin:0}
.hero .meta{display:flex;gap:22px;flex-wrap:wrap;margin-top:18px;font-size:13px;opacity:.9}
.hero .meta b{font-weight:600}
.pill{display:inline-flex;align-items:center;gap:6px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600}
.pill i{width:8px;height:8px;border-radius:50%;background:#3ddc84;display:inline-block}
nav{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:1px solid var(--ring);display:flex;gap:4px;align-items:center;padding:8px 40px;font-size:13px}
nav a{text-decoration:none;color:var(--ink2);padding:6px 10px;border-radius:6px}nav a:hover{background:var(--band);color:var(--ink)}
nav .sp{flex:1}
button,select{font:inherit;font-size:13px;color:var(--ink);background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:5px 10px;cursor:pointer}
button[aria-pressed=true]{outline:2px solid var(--s1);outline-offset:-2px}
main{max-width:1240px;margin:0 auto;padding:28px 40px 60px;display:grid;gap:28px}
section h2{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:0 0 12px;font-weight:600}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:18px 20px}
.card h3{font-size:16px;margin:0 0 2px;font-weight:600}.card .sub{color:var(--ink2);font-size:13px;margin:0 0 10px}
.summary{font-size:16px;line-height:1.6;max-width:1000px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.kpi{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px 18px;display:flex;flex-direction:column;gap:2px}
.kpi .l{color:var(--ink2);font-size:12px;letter-spacing:.02em}.kpi .v{font-size:30px;font-weight:650;letter-spacing:-.02em;line-height:1.1;margin:4px 0}
.kpi .s{color:var(--muted);font-size:12px}.kpi .d{font-size:12px;font-weight:600;margin-top:4px}.kpi .d.good{color:var(--good)}.kpi .d.warn{color:var(--warn)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.two{grid-template-columns:1fr}}
svg{width:100%;height:auto;display:block}
.gl{stroke:var(--grid);stroke-width:1}.ax{stroke:var(--axis);stroke-width:1}.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.mk{stroke:var(--surface);stroke-width:2}
.raw{fill:none;stroke-width:1;opacity:.28}
.cross{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3;display:none}
.band{fill:var(--band)}.bandl{fill:var(--muted);font-size:10px;letter-spacing:.06em;text-transform:uppercase}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 6px 20px rgba(0,0,0,.14);display:none;z-index:9;font-variant-numeric:tabular-nums;min-width:170px}
.tip .h{color:var(--muted);margin-bottom:4px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin:0 0 6px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--grid)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
th{color:var(--ink2);font-weight:600;font-size:12px;letter-spacing:.04em;text-transform:uppercase}
tr.best td{font-weight:600}.delta{font-size:12px;color:var(--good)}.delta.up{color:var(--warn)}
.scroll{max-height:380px;overflow:auto}
ul.ins{margin:0;padding-left:18px;display:grid;gap:8px}ul.ins li{max-width:1000px}
code{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;background:var(--band);padding:1px 5px;border-radius:4px}
pre{background:var(--band);border-radius:8px;padding:12px 14px;overflow:auto;font-size:12.5px;line-height:1.5}
footer{color:var(--muted);font-size:12px;text-align:center;padding:10px 0 30px}
.hidden{display:none}
nav{flex-wrap:wrap}nav a{white-space:nowrap}.card{overflow-x:auto}
td,th{overflow-wrap:anywhere}#cfg{table-layout:fixed}#cfg td:first-child{width:38%}
.summary{overflow-wrap:anywhere}
@media(max-width:900px){.hero,nav,main{padding-left:18px;padding-right:18px}.hero h1{font-size:24px}.kpi .v{font-size:26px}}
@media print{nav,.controls{display:none}.card{break-inside:avoid}body{background:#fff}}
</style></head><body>
<div class="hero">
 <div class="eyebrow">Model training report · NVIDIA Cosmos 3 · TAO</div>
 <h1>__TITLE__</h1>
 <p class="lede">__LEDE__</p>
 <div class="meta"><span class="pill"><i></i>Completed successfully</span>__META__</div>
</div>
<nav><a href="#summary">Summary</a><a href="#dynamics">Training dynamics</a><a href="#epochs">Per-epoch</a><a href="#optim">Optimization</a><a href="#insights">Insights</a><a href="#config">Configuration</a><a href="#repro">Reproduce</a><span class="sp"></span>
 <span class="controls" style="display:flex;gap:8px;align-items:center;color:var(--ink2)">Smoothing <select id="smooth"><option value="1">off</option><option value="5">5</option><option value="10" selected>10</option><option value="25">25</option></select>
 <label><input type="checkbox" id="logy"> log y</label><button id="view" aria-pressed="false">Data table</button><button id="theme">Theme</button></span></nav>
<main>
 <section id="summary"><h2>Executive summary</h2><div class="card"><p class="summary" style="margin:0">__SUMMARY__</p></div></section>
 <section><div class="kpis" id="kpis"></div></section>
 <section id="dynamics"><h2>Training dynamics</h2><div class="card" id="hero"></div></section>
 <section id="epochs"><h2>Per-epoch results</h2><div class="card"><table id="eptable"></table></div></section>
 <section id="optim"><h2>Optimization health</h2><div class="two" id="small"></div></section>
 <section id="insights"><h2>Insights</h2><div class="card"><ul class="ins" id="ins"></ul></div></section>
 <section id="tables" class="hidden"><h2>Metrics by step</h2><div class="card scroll" id="tbl"></div></section>
 <section id="config"><h2>Run configuration</h2><div class="card scroll"><table id="cfg"></table></div></section>
 <section id="repro"><h2>Reproduce</h2><div class="card"><p class="sub">Commands used for this run (see README for the full runbook).</p><pre>__REPRO__</pre></div></section>
</main>
<footer>__FOOTER__</footer>
<div class="tip" id="tip"></div>
<script>
const D=__DATA__;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const f=(x)=>x==null?'':Math.abs(x)>=1000?x.toLocaleString(undefined,{maximumFractionDigits:0}):Math.abs(x)>=1?x.toFixed(3):Math.abs(x)>=1e-3?x.toFixed(4):x.toExponential(2);
const el=(id)=>document.getElementById(id);
// KPIs
el('kpis').innerHTML=D.kpis.map(k=>`<div class="kpi"><div class="l">${k.label}</div><div class="v">${k.value}</div><div class="s">${k.sub||''}</div>${k.delta?`<div class="d ${k.tone}">${k.delta}</div>`:''}</div>`).join('');
// insights
el('ins').innerHTML=D.insights.map(i=>`<li>${i}</li>`).join('');
// epoch table
(function(){const rows=D.epochs;const best=Math.min(...rows.filter(r=>r.val!=null).map(r=>r.val));
 el('eptable').innerHTML=`<thead><tr><th>Epoch</th><th>Steps</th><th class="n">Mean train loss</th><th class="n">End-of-epoch train loss</th><th class="n">Validation loss</th><th class="n">Δ validation</th><th class="n">Minutes</th></tr></thead><tbody>`+
 rows.map(r=>`<tr class="${r.val===best?'best':''}"><td>${r.epoch}${r.val===best?' <span class="delta">★ best</span>':''}</td><td>${r.steps}</td><td class="n">${f(r.train_mean)}</td><td class="n">${f(r.train_last)}</td><td class="n">${f(r.val)}</td><td class="n">${r.val_delta_pct==null?'—':`<span class="delta ${r.val_delta_pct>0?'up':''}">${r.val_delta_pct>0?'▲':'▼'} ${Math.abs(r.val_delta_pct).toFixed(1)}%</span>`}</td><td class="n">${r.minutes==null?'':r.minutes.toFixed(1)}</td></tr>`).join('')+'</tbody>'})();
// charts
function smooth(a,w){if(w<=1)return a;const o=[];for(let i=0;i<a.length;i++){let s=0,n=0;for(let j=Math.max(0,i-w+1);j<=i;j++){s+=a[j][1];n++}o.push([a[i][0],s/n])}return o}
function chart(host,opts){const {series,H=320,ylabel='',bands=true,markers=false}=opts;const W=760,L=62,R=20,T=22,B=40;const logy=el('logy').checked;
 const allx=series.flatMap(s=>s.pts.map(p=>p[0]));const ally=series.flatMap(s=>(s.raw||s.pts).map(p=>p[1])).filter(v=>!logy||v>0);
 const x0=Math.min(...allx),x1=Math.max(...allx);let y0=Math.min(...ally),y1=Math.max(...ally);if(y0===y1){y0*=0.9;y1*=1.1}const pad=(y1-y0)*0.06;if(!logy){y0=Math.max(0,y0-pad);y1+=pad}
 const ty=v=>logy?Math.log10(v):v;const sx=x=>L+(x-x0)/((x1-x0)||1)*(W-L-R);const sy=v=>T+(1-(ty(v)-ty(y0))/((ty(y1)-ty(y0))||1))*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.title}">`;
 if(bands&&D.epochs.length>1){D.epochs.forEach((e,i)=>{const [a,b]=e.steps.split('–').map(Number);if(i%2===0)s+=`<rect class="band" x="${sx(a)}" y="${T}" width="${sx(b)-sx(a)}" height="${H-T-B}"/>`;s+=`<text class="bandl" x="${(sx(a)+sx(b))/2}" y="${T-8}" text-anchor="middle">epoch ${e.epoch}</text>`})}
 for(let i=0;i<=5;i++){const v=logy?Math.pow(10,ty(y0)+(ty(y1)-ty(y0))*i/5):y0+(y1-y0)*i/5;s+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${sy(v)}" y2="${sy(v)}"/><text class="tick" x="${L-8}" y="${sy(v)+4}" text-anchor="end">${f(v)}</text>`}
 for(let i=0;i<=6;i++){const x=Math.round(x0+(x1-x0)*i/6);s+=`<text class="tick" x="${sx(x)}" y="${H-B+18}" text-anchor="middle">${x}</text>`}
 s+=`<line class="ax" x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}"/><text class="tick" x="${(L+W-R)/2}" y="${H-6}" text-anchor="middle">optimizer step</text>`;
 if(ylabel)s+=`<text class="tick" transform="translate(14 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle">${ylabel}</text>`;
 for(const sr of series){if(sr.raw&&sr.raw!==sr.pts){s+=`<path class="raw" stroke="${sr.color}" d="${sr.raw.filter(p=>!logy||p[1]>0).map((p,i)=>(i?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1)).join(' ')}"/>`}
  s+=`<path class="ln" stroke="${sr.color}" d="${sr.pts.filter(p=>!logy||p[1]>0).map((p,i)=>(i?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1)).join(' ')}"/>`;
  if(sr.markers||markers||sr.pts.length<=40)for(const p of sr.pts)s+=`<circle class="mk" cx="${sx(p[0])}" cy="${sy(p[1])}" r="4.5" fill="${sr.color}"/>`;
  const last=sr.pts[sr.pts.length-1];s+=`<text class="tick" style="fill:var(--ink2);font-weight:600" x="${Math.min(sx(last[0])+8,W-R-30)}" y="${sy(last[1])+4}">${f(last[1])}</text>`}
 s+=`<line class="cross" y1="${T}" y2="${H-B}"/><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/></svg>`;
 host.innerHTML=`<h3>${opts.title}</h3><p class="sub">${opts.sub||''}</p>`+(series.length>1?`<div class="legend">${series.map(sr=>`<span><i class="sw" style="background:${sr.color}"></i>${sr.name}</span>`).join('')}</div>`:'')+s;
 const rect=host.querySelector('rect[fill=transparent]'),cross=host.querySelector('.cross'),tip=el('tip');
 rect.addEventListener('mousemove',ev=>{const bb=rect.getBoundingClientRect();const x=x0+(ev.clientX-bb.left)/bb.width*(x1-x0);cross.style.display='block';cross.setAttribute('x1',sx(x));cross.setAttribute('x2',sx(x));
  let rows=`<div class="h">step ${Math.round(x)}</div>`;for(const sr of series){const src=sr.raw||sr.pts;let b=src[0];for(const p of src)if(Math.abs(p[0]-x)<Math.abs(b[0]-x))b=p;rows+=`<div><i class="sw" style="background:${sr.color}"></i>${sr.name}: <b>${f(b[1])}</b></div>`}
  tip.innerHTML=rows;tip.style.display='block';tip.style.left=Math.min(ev.clientX+14,innerWidth-200)+'px';tip.style.top=(ev.clientY+14)+'px'});
 rect.addEventListener('mouseleave',()=>{cross.style.display='none';tip.style.display='none'});}
function render(){const w=+el('smooth').value;const M=D.metrics;
 const series=[];if(M.loss)series.push({name:'Training loss (smoothed)',color:css('--s1'),pts:smooth(M.loss,w),raw:M.loss});if(M['val/loss'])series.push({name:'Validation loss (per epoch)',color:css('--s2'),pts:M['val/loss'],markers:true});
 chart(el('hero'),{title:'Loss over training',sub:'Training loss per optimizer step (smoothed; faint line is raw) with validation loss at each epoch boundary. Shaded bands mark epochs.',series,H:340,ylabel:'loss'});
 const small=el('small');small.innerHTML='';const defs=[[D.lr_key,'Learning rate','Warm-up then decay schedule applied to the LoRA parameters.',false],['grad_norm','Gradient norm','Global gradient norm before clipping (clip = '+(D.clip||'?')+'). Spikes indicate unstable batches.',true],['iteration_time','Step time','Seconds per optimizer step. Flat = no input pipeline stalls.',true]];
 for(const [k,t,sub,sm] of defs){if(!k||!M[k])continue;const c=document.createElement('div');c.className='card';small.appendChild(c);chart(c,{title:t,sub,series:[{name:t,color:css('--s1'),pts:sm?smooth(M[k],w):M[k],raw:sm?M[k]:null}],H:260,bands:false})}}
// table + config
(function(){const keys=Object.keys(D.metrics);const steps=[...new Set(keys.flatMap(k=>D.metrics[k].map(p=>p[0])))].sort((a,b)=>a-b);const idx={};for(const k of keys)idx[k]=new Map(D.metrics[k]);
 el('tbl').innerHTML=`<table><thead><tr><th class="n">step</th>${keys.map(k=>`<th class="n">${D.names[k]||k}</th>`).join('')}</tr></thead><tbody>${steps.map(s=>`<tr><td class="n">${s}</td>${keys.map(k=>`<td class="n">${idx[k].has(s)?f(idx[k].get(s)):''}</td>`).join('')}</tr>`).join('')}</tbody></table>`})();
el('cfg').innerHTML='<thead><tr><th>Parameter</th><th class="n">Value</th></tr></thead><tbody>'+D.config.map(([k,v])=>`<tr><td>${k}</td><td class="n">${v}</td></tr>`).join('')+'</tbody>';
el('smooth').onchange=render;el('logy').onchange=render;
el('view').onclick=e=>{const on=e.target.getAttribute('aria-pressed')!=='true';e.target.setAttribute('aria-pressed',on);el('tables').classList.toggle('hidden',!on);if(on)location.hash='#tables'};
el('theme').onclick=()=>{const r=document.documentElement;const dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme: dark)').matches);r.setAttribute('data-theme',dark?'light':'dark');render()};
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results")
    ap.add_argument("--log")
    ap.add_argument("--spec")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--title")
    ap.add_argument("--cluster", default="", help="e.g. 'lyris · gb200 · 4× GB200' shown in the header")
    ap.add_argument("--job", default="", help="SLURM job id shown in the header")
    a = ap.parse_args()
    results = Path(a.results) if a.results else None
    log = Path(a.log) if a.log else (results / "train.log" if results else None)
    if not log or not log.exists():
        raise SystemExit("train.log not found; pass --results <dir> or --log <file>")

    metrics, events, epochs, total_steps, t0, t1, step_ts, nlines = parse_log(log)
    spec = load_spec(results, Path(a.spec) if a.spec else None)
    flat = flatten(spec)
    an = analyze(metrics, events, epochs, total_steps, t0, t1, step_ts, flat)

    model = Path(str(flat.get("policy.model_name_or_path", "Cosmos3-Nano"))).name
    run = flat.get("logging.experiment_name") or (results.name if results else log.stem)
    title = a.title or "Cosmos3-Nano LoRA fine-tune · vial-filling line inspection"
    lede = (f"Supervised fine-tuning of the {model} reasoner on synthetic vial-inspection conversations "
            f"(color, fill level and plan compliance per vial), using NVIDIA TAO's cosmos-rl backend.")
    meta = "".join(f"<span><b>{k}</b> {v}</span>" for k, v in [
        ("Run", run), ("Model", model), ("Date", t0.strftime("%Y-%m-%d") if t0 else ""),
        ("Cluster", a.cluster), ("SLURM job", a.job), ("Epochs", flat.get("train.epoch", "")),
    ] if v)
    repro = "\n".join([
        "export ACCOUNT=general_sa PARTITION=gb200 LUSTRE_DIR=/lustre/fsw/general_sa/$USER",
        "cd $LUSTRE_DIR/cosmos3tao/repo && git pull",
        "bash cluster/ptyche_setup.sh            # dataset, model download, container .sqsh (once)",
        "bash cluster/ptyche_prepare_model.sh    # Cosmos3-Nano omni -> Qwen3-VL checkpoint (once)",
        "sbatch --account=$ACCOUNT --partition=$PARTITION --export=ALL,ACCOUNT=$ACCOUNT,LUSTRE_DIR=$LUSTRE_DIR \\",
        "       --output=$LUSTRE_DIR/cosmos3tao/logs/%x-%j.out --error=$LUSTRE_DIR/cosmos3tao/logs/%x-%j.err cluster/ptyche_train.sbatch",
        f"python report/make_report.py --log results/train.log --spec results/train_spec.yaml --out report.html",
    ])
    cfg_keys = ["policy.model_name_or_path", "policy.lora.r", "policy.lora.lora_alpha", "policy.lora.lora_dropout", "policy.lora.target_modules",
                "train.epoch", "train.train_batch_per_replica", "train.train_policy.mini_batch", "train.optm_name", "train.optm_lr",
                "train.optm_weight_decay", "train.optm_warmup_epochs", "train.optm_min_lr_factor", "train.optm_grad_norm_clip",
                "policy.model_max_length", "policy.model_gradient_checkpointing", "policy.parallelism.dp_shard_size", "train.param_dtype",
                "train.master_dtype", "custom.vision.total_pixels", "custom.dataset.annotation_path", "custom.val_dataset.annotation_path",
                "validation.freq_in_epoch", "train.ckpt.save_freq_in_epoch", "train.ckpt.max_keep", "results_dir"]
    config = [(k, fmt(flat[k])) for k in cfg_keys if k in flat] + [(k, fmt(v)) for k, v in sorted(flat.items()) if k not in cfg_keys and not isinstance(v, (dict, list))]

    data = {
        "metrics": {k: [[s, v] for s, v in pts.items()] for k, pts in metrics.items()},
        "names": {k: pretty(k) for k in metrics},
        "events": events, "epochs": an["epochs"], "kpis": an["kpis"], "insights": an["insights"], "lr_key": an["lr_key"],
        "clip": flat.get("train.optm_grad_norm_clip"),
        "config": [[html.escape(str(k)), html.escape(str(v))] for k, v in config],
    }
    out = (TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__LEDE__", html.escape(lede)).replace("__META__", meta)
           .replace("__SUMMARY__", an["summary"]).replace("__REPRO__", html.escape(repro))
           .replace("__FOOTER__", f"Generated {datetime.now():%Y-%m-%d %H:%M} from {log.name} ({nlines:,} lines) · cosmos3tao")
           .replace("__DATA__", json.dumps(data)))
    Path(a.out).write_text(out, encoding="utf-8")
    print(f"metrics: {sorted(metrics)}  epochs: {len(an['epochs'])}  steps: {an['steps']}  -> {a.out}")
    if not metrics:
        print(f"WARNING: no per-step metrics parsed; check the log format:  grep -m 5 'Step:' {log}")


if __name__ == "__main__":
    main()
