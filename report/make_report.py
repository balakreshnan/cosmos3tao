"""
Build a self-contained interactive HTML report from a cosmos-rl / TAO training run.

Inputs (any subset; pass the run's results directory or individual files):
  <results_dir>/train.log                      console log (all ranks; rank0 is used when tagged)
  <results_dir>/train_spec.yaml | spec.toml    run configuration (shown as a table)
  <results_dir>/output/**/status.json          TAO status logger lines (JSON per line), if present
  <results_dir>/output/best/best_score.json    best validation score

The parser is format-tolerant: it picks up every `Step: N` line and any `name: value` / `name=value`
numeric pairs on it (loss, lr, grad_norm, tokens/s, ...), plus validation lines, epoch boundaries and
checkpoint events. Metrics are plotted as one SVG line chart per metric with crosshair + tooltip, a
legend, a table view, and dark mode. No external assets.

Usage:
  python report/make_report.py --results results/cosmos3_nano_tube_lora_3044273 --out report.html
  python report/make_report.py --log train.log --out report.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RANK_RE = re.compile(r"^\[rank(\d+)\]:\s*")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
STEP_RE = re.compile(r"\b[Ss]tep[:=\s]+(\d+)")
EPOCH_RE = re.compile(r"\b[Ee]poch[:=\s/]+(\d+)")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_/\.]*)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\b")
SKIP_KEYS = {"step", "epoch", "rank", "pid", "port", "seed", "line", "iter", "iteration", "total_steps", "num_steps",
             "world_size", "local_rank", "global_step"}
CKPT_RE = re.compile(r"checkpoint saved successfully at (\S+)")
BEST_RE = re.compile(r"Best checkpoint updated to (\S+) with score: ([-\d.eE+]+)")


def parse_log(path: Path):
    metrics: dict[str, dict[int, float]] = defaultdict(dict)   # metric -> {step: value}
    events: list[dict] = []
    step_epoch: dict[int, int] = {}
    first_ts = last_ts = None
    lines_total = 0
    for raw in path.read_text(errors="replace").splitlines():
        lines_total += 1
        m = RANK_RE.match(raw)
        rank = int(m.group(1)) if m else 0
        line = RANK_RE.sub("", raw)
        ts = TS_RE.search(line)
        if ts:
            t = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S")
            first_ts = first_ts or t
            last_ts = t
        if rank != 0:
            continue  # metrics are identical across ranks; keep rank0 only
        sm = STEP_RE.search(line)
        if sm:
            step = int(sm.group(1))
            em = EPOCH_RE.search(line)
            if em:
                step_epoch[step] = int(em.group(1))
            body = line[sm.end():]
            is_val = "val" in line.lower() and "valid" in line.lower()
            # split on commas / pipes so multi-word keys ("Grad Norm: 1.2") survive
            pairs = []
            for frag in re.split(r"[,|;]", body):
                fm = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_/\. ]*?)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$", frag)
                if fm:
                    pairs.append((fm.group(1).strip().replace(" ", "_"), fm.group(2)))
            if not pairs:
                pairs = KV_RE.findall(body)
            for k, v in pairs:
                key = k.strip().lower()
                if key in SKIP_KEYS or key.startswith("rank"):
                    continue
                if is_val and not key.startswith("val"):
                    key = "val/" + key
                try:
                    metrics[key][step] = float(v)
                except ValueError:
                    pass
            cm = CKPT_RE.search(line)
            if cm:
                events.append({"step": step, "type": "checkpoint", "text": cm.group(1).rsplit("/", 2)[-2]})
        bm = BEST_RE.search(line)
        if bm:
            events.append({"step": max(step_epoch, default=0), "type": "best", "text": f"{bm.group(1)} score {float(bm.group(2)):.4g}"})
    # drop constants / junk: keep metrics with >= 2 points and some variation or a known name
    known = ("loss", "lr", "learning", "grad", "norm", "token", "acc", "reward", "kl", "entropy", "throughput", "mfu", "time")
    keep = {}
    for k, pts in metrics.items():
        vals = list(pts.values())
        if len(pts) >= 2 and (any(w in k for w in known) or len(set(vals)) > 2):
            keep[k] = dict(sorted(pts.items()))
    return keep, events, step_epoch, first_ts, last_ts, lines_total


def parse_status_json(results: Path):
    """TAO status logger writes JSON lines with kpi/graphical values; merge anything numeric keyed by step."""
    metrics: dict[str, dict[int, float]] = defaultdict(dict)
    for p in results.rglob("status.json"):
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = None
            for key in ("step", "cur_iter", "global_step", "iteration"):
                v = d.get(key) or (d.get("graphical") or {}).get(key) or (d.get("kpi") or {}).get(key)
                if isinstance(v, (int, float)):
                    step = int(v); break
            if step is None:
                continue
            for section in ("graphical", "kpi"):
                for k, v in (d.get(section) or {}).items():
                    if isinstance(v, (int, float)) and k not in SKIP_KEYS:
                        metrics[k.lower()][step] = float(v)
    return {k: dict(sorted(v.items())) for k, v in metrics.items() if len(v) >= 2}


def load_spec(results: Path | None, spec_path: Path | None):
    p = spec_path or (results and next((q for q in (results / "train_spec.yaml", results / "spec.toml") if q.exists()), None))
    if not p:
        return {}
    text = p.read_text()
    if p.suffix == ".yaml":
        try:
            import yaml
            return yaml.safe_load(text)
        except Exception:
            return {"raw": text}
    try:
        import tomllib
        return tomllib.loads(text)
    except Exception:
        return {"raw": text}


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7;--s8:#e34948;--good:#006300}
:root[data-theme=dark]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--good:#0ca30c}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--good:#0ca30c}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;align-items:baseline;gap:16px;padding:20px 28px 8px}h1{font-size:20px;margin:0;font-weight:600}header .sub{color:var(--ink2)}
.bar{display:flex;gap:8px;align-items:center;padding:0 28px 12px;flex-wrap:wrap}
button,select{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:6px 10px;cursor:pointer}
button[aria-pressed=true]{outline:2px solid var(--s1);outline-offset:-2px}
main{padding:0 28px 40px;display:grid;gap:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile .l{color:var(--ink2);font-size:12px}.tile .v{font-size:26px;font-weight:600;margin-top:2px}.tile .d{color:var(--muted);font-size:12px}
.tile .v.good{color:var(--good)}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.card h2{font-size:14px;font-weight:600;margin:0 0 8px;color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}
svg{width:100%;height:auto;display:block}
.gl{stroke:var(--grid);stroke-width:1}.ax{stroke:var(--axis);stroke-width:1}.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.mk{stroke:var(--surface);stroke-width:2}
.cross{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3;display:none}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;padding:8px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none;z-index:9;font-variant-numeric:tabular-nums}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin:6px 0 2px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.ev{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 4}.evl{fill:var(--muted);font-size:10px}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
th{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--surface)}
.scroll{max-height:420px;overflow:auto}
.hidden{display:none}
details summary{cursor:pointer;color:var(--ink2)}
</style></head><body>
<header><h1>__TITLE__</h1><span class="sub">__SUBTITLE__</span></header>
<div class="bar">
 <span style="color:var(--ink2)">Smoothing</span>
 <select id="smooth"><option value="1">none</option><option value="5" selected>5 steps</option><option value="10">10 steps</option><option value="25">25 steps</option></select>
 <label><input type="checkbox" id="logy"> log y-axis</label>
 <button id="view" aria-pressed="false">Table view</button>
 <button id="theme">Dark / light</button>
 <span style="margin-left:auto;color:var(--muted)">__GENERATED__</span>
</div>
<main>
 <section class="tiles" id="tiles"></section>
 <section id="charts" class="grid"></section>
 <section id="tables" class="hidden"></section>
 <section class="card"><details><summary>Run configuration</summary><div class="scroll"><table id="cfg"></table></div></details></section>
 <section class="card"><details><summary>Events</summary><div class="scroll"><table id="evt"></table></div></details></section>
</main>
<div class="tip" id="tip"></div>
<script>
const DATA=__DATA__;
const SERIES=['--s1','--s2','--s3','--s4','--s5','--s6','--s7','--s8'];
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const f=(x)=>Math.abs(x)>=1000?x.toLocaleString(undefined,{maximumFractionDigits:0}):Math.abs(x)>=1?x.toFixed(3):x.toPrecision(3);
// ---- tiles
const T=document.getElementById('tiles');
for(const t of DATA.tiles){T.insertAdjacentHTML('beforeend',`<div class="tile"><div class="l">${t.label}</div><div class="v ${t.good?'good':''}">${t.value}</div><div class="d">${t.detail||''}</div></div>`)}
// ---- charts: group train/val pairs of same metric
const groups=[];const seen=new Set();
for(const k of Object.keys(DATA.metrics)){ if(seen.has(k))continue; const base=k.replace(/^(val|train|eval)[\/_]/,''); const g=Object.keys(DATA.metrics).filter(x=>x.replace(/^(val|train|eval)[\/_]/,'')===base); g.forEach(x=>seen.add(x)); groups.push({title:base,keys:g}); }
function smooth(arr,w){ if(w<=1)return arr; const out=[];for(let i=0;i<arr.length;i++){let s=0,n=0;for(let j=Math.max(0,i-w+1);j<=i;j++){s+=arr[j][1];n++}out.push([arr[i][0],s/n])}return out}
const C=document.getElementById('charts');
function render(){ C.innerHTML=''; const w=+document.getElementById('smooth').value, logy=document.getElementById('logy').checked;
 for(const [gi,g] of groups.entries()){
  const W=640,H=300,L=56,R=16,Tp=18,B=34; const card=document.createElement('div');card.className='card';
  const series=g.keys.map((k,i)=>({k,color:css(SERIES[i%8]),pts:smooth(DATA.metrics[k],k.startsWith('val')?1:w),raw:DATA.metrics[k]}));
  const allx=series.flatMap(s=>s.pts.map(p=>p[0])), ally=series.flatMap(s=>s.pts.map(p=>p[1])).filter(v=>!logy||v>0);
  const x0=Math.min(...allx),x1=Math.max(...allx); let y0=Math.min(...ally),y1=Math.max(...ally); if(y0===y1){y0-=1;y1+=1}
  const sx=x=>L+(x-x0)/(x1-x0||1)*(W-L-R); const ty=v=>logy?Math.log10(v):v; const sy=v=>Tp+(1-(ty(v)-ty(y0))/((ty(y1)-ty(y0))||1))*(H-Tp-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${g.title} over training steps">`;
  const yt=5;for(let i=0;i<=yt;i++){const v=logy?Math.pow(10,ty(y0)+(ty(y1)-ty(y0))*i/yt):y0+(y1-y0)*i/yt;const y=sy(v);svg+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/><text class="tick" x="${L-6}" y="${y+4}" text-anchor="end">${f(v)}</text>`}
  const xt=6;for(let i=0;i<=xt;i++){const x=x0+(x1-x0)*i/xt;svg+=`<text class="tick" x="${sx(x)}" y="${H-B+16}" text-anchor="middle">${Math.round(x)}</text>`}
  svg+=`<line class="ax" x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}"/><text class="tick" x="${(L+W-R)/2}" y="${H-4}" text-anchor="middle">step</text>`;
  for(const e of DATA.events.filter(e=>e.type==='checkpoint')){const x=sx(e.step);if(x>=L&&x<=W-R)svg+=`<line class="ev" x1="${x}" x2="${x}" y1="${Tp}" y2="${H-B}"/><text class="evl" x="${x+3}" y="${Tp+10}">${e.text}</text>`}
  for(const s of series){const d=s.pts.filter(p=>!logy||p[1]>0).map((p,i)=>(i?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1)).join(' ');svg+=`<path class="ln" stroke="${s.color}" d="${d}"/>`;
   if(s.pts.length<=60)for(const p of s.pts)svg+=`<circle class="mk" cx="${sx(p[0])}" cy="${sy(p[1])}" r="4" fill="${s.color}"/>`;
   const last=s.pts[s.pts.length-1];svg+=`<text class="tick" x="${Math.min(sx(last[0])+6,W-R-40)}" y="${sy(last[1])+4}" fill="${css('--ink2')}">${f(last[1])}</text>`}
  svg+=`<line class="cross" id="cx${gi}" y1="${Tp}" y2="${H-B}"/><rect x="${L}" y="${Tp}" width="${W-L-R}" height="${H-Tp-B}" fill="transparent" data-g="${gi}"/></svg>`;
  const legend=series.length>1?`<div class="legend">${series.map(s=>`<span><i class="sw" style="background:${s.color}"></i>${s.k}</span>`).join('')}</div>`:'';
  card.innerHTML=`<h2>${g.title}</h2>${legend}${svg}`; C.appendChild(card);
  const rect=card.querySelector('rect'), cross=card.querySelector('.cross'), tip=document.getElementById('tip');
  rect.addEventListener('mousemove',ev=>{const bb=rect.getBoundingClientRect();const x=x0+(ev.clientX-bb.left)/bb.width*(x1-x0);cross.style.display='block';cross.setAttribute('x1',sx(x));cross.setAttribute('x2',sx(x));
    let rows='';for(const s of series){let best=s.raw[0];for(const p of s.raw)if(Math.abs(p[0]-x)<Math.abs(best[0]-x))best=p;rows+=`<div><i class="sw" style="background:${s.color}"></i>${s.k}: <b>${f(best[1])}</b> <span style="color:var(--muted)">@ step ${best[0]}</span></div>`}
    tip.innerHTML=rows;tip.style.display='block';tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+14)+'px'});
  rect.addEventListener('mouseleave',()=>{cross.style.display='none';tip.style.display='none'});
 }}
// ---- tables
const TB=document.getElementById('tables');
(function(){const keys=Object.keys(DATA.metrics);const steps=[...new Set(keys.flatMap(k=>DATA.metrics[k].map(p=>p[0])))].sort((a,b)=>a-b);
 let h=`<div class="card"><h2>Metrics by step</h2><div class="scroll"><table><thead><tr><th class="n">step</th>${keys.map(k=>`<th class="n">${k}</th>`).join('')}</tr></thead><tbody>`;
 const idx={};for(const k of keys){idx[k]=new Map(DATA.metrics[k])}
 for(const s of steps)h+=`<tr><td class="n">${s}</td>${keys.map(k=>`<td class="n">${idx[k].has(s)?f(idx[k].get(s)):''}</td>`).join('')}</tr>`;
 TB.innerHTML=h+'</tbody></table></div></div>'})();
document.getElementById('cfg').innerHTML=DATA.config.map(([k,v])=>`<tr><td>${k}</td><td class="n">${v}</td></tr>`).join('');
document.getElementById('evt').innerHTML='<tr><th class="n">step</th><th>type</th><th>detail</th></tr>'+DATA.events.map(e=>`<tr><td class="n">${e.step}</td><td>${e.type}</td><td>${e.text}</td></tr>`).join('');
// ---- controls
document.getElementById('smooth').onchange=render;document.getElementById('logy').onchange=render;
document.getElementById('view').onclick=e=>{const on=e.target.getAttribute('aria-pressed')!=='true';e.target.setAttribute('aria-pressed',on);C.classList.toggle('hidden',on);TB.classList.toggle('hidden',!on)};
document.getElementById('theme').onclick=()=>{const r=document.documentElement;const dark=r.getAttribute('data-theme')==='dark'||(!r.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme: dark)').matches);r.setAttribute('data-theme',dark?'light':'dark');render()};
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", help="run results directory (train.log, spec, output/)")
    ap.add_argument("--log", help="explicit train.log path")
    ap.add_argument("--spec", help="explicit spec yaml/toml path")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    results = Path(a.results) if a.results else None
    log = Path(a.log) if a.log else (results / "train.log" if results else None)
    if not log or not log.exists():
        raise SystemExit("train.log not found; pass --results <dir> or --log <file>")

    metrics, events, step_epoch, t0, t1, nlines = parse_log(log)
    if results:
        for k, v in parse_status_json(results).items():
            metrics.setdefault(k, v)
    spec = load_spec(results, Path(a.spec) if a.spec else None)
    flat = flatten(spec)

    best = None
    if results and (results / "output" / "best" / "best_score.json").exists():
        try:
            best = json.loads((results / "output" / "best" / "best_score.json").read_text())
        except Exception:
            best = None

    # ---- tiles
    tiles = []
    train_key = next((k for k in metrics if "loss" in k and not k.startswith("val")), None)
    val_key = next((k for k in metrics if k.startswith("val") and "loss" in k), None)
    steps = max((max(v) for v in metrics.values()), default=max(step_epoch, default=0))
    if train_key:
        pts = list(metrics[train_key].items())
        first, last = pts[0][1], pts[-1][1]
        tiles.append({"label": "Final train loss", "value": f"{last:.4g}", "detail": f"from {first:.4g} at step {pts[0][0]}  ({(1 - last / first) * 100:+.0f}%)" if first else ""})
    if val_key:
        pts = list(metrics[val_key].items())
        mn = min(pts, key=lambda p: p[1])
        tiles.append({"label": "Best validation loss", "value": f"{mn[1]:.4g}", "detail": f"step {mn[0]}", "good": True})
    elif best is not None:
        score = best.get("score", best) if isinstance(best, dict) else best
        try:
            tiles.append({"label": "Best validation score", "value": f"{float(score):.4g}", "detail": str(best.get("checkpoint", "") if isinstance(best, dict) else ""), "good": True})
        except Exception:
            pass
    tiles.append({"label": "Optimizer steps", "value": f"{steps:,}", "detail": f"{flat.get('train.epoch', '?')} epochs · batch {flat.get('train.train_batch_per_replica', '?')} × {flat.get('policy.parallelism.dp_shard_size', '?')} GPUs"})
    if t0 and t1:
        dur = t1 - t0
        h, rem = divmod(int(dur.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        tiles.append({"label": "Wall time", "value": f"{h}h {m:02d}m", "detail": f"{steps / max(dur.total_seconds(), 1) * 60:.1f} steps/min" if steps else ""})
    tiles.append({"label": "Trainable setup", "value": f"LoRA r={flat.get('policy.lora.r', '?')}", "detail": f"alpha {flat.get('policy.lora.lora_alpha', '?')} · lr {flat.get('train.optm_lr', '?')} · {', '.join(map(str, flat.get('policy.lora.target_modules', []) or []))}"})

    cfg_rows = [(k, fmt(v)) for k, v in sorted(flat.items()) if not isinstance(v, (dict, list)) or True][:400]
    run_name = flat.get("logging.experiment_name") or (results.name if results else log.stem)
    model = flat.get("policy.model_name_or_path", "")
    title = a.title or f"Training report · {run_name}"
    subtitle = f"{Path(str(model)).name}  ·  {len(metrics)} metrics from {nlines:,} log lines" if model else f"{len(metrics)} metrics from {nlines:,} log lines"

    data = {
        "metrics": {k: [[s, v] for s, v in pts.items()] for k, pts in metrics.items()},
        "events": events,
        "tiles": tiles,
        "config": [[html.escape(str(k)), html.escape(str(v))] for k, v in cfg_rows],
    }
    out = HTML.replace("__TITLE__", html.escape(title)).replace("__SUBTITLE__", html.escape(subtitle)) \
              .replace("__GENERATED__", f"generated {datetime.now():%Y-%m-%d %H:%M}") \
              .replace("__DATA__", json.dumps(data))
    Path(a.out).write_text(out, encoding="utf-8")
    print(f"metrics found: {sorted(metrics)}")
    print(f"events: {len(events)}  steps: {steps}  -> {a.out}")
    if not metrics:
        print("WARNING: no per-step metrics parsed. Send me 20 lines from the log containing 'Step' so I can adapt the parser:")
        print(f"  grep -m 20 -i 'step' {log}")


if __name__ == "__main__":
    main()
