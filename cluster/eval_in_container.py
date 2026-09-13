"""
Task-accuracy evaluation of the (LoRA fine-tuned) Cosmos3-Nano / Qwen3-VL model on the vial-inspection
validation split. Runs INSIDE the TAO cosmos-rl container on one GPU (transformers + safetensors).

  python eval_in_container.py --model /tao-workspace/models/Cosmos3-Nano-qwen3vl \
      --adapter /tao-workspace/results/<run>/output/<ts>/safetensors/epoch_5 \
      --annotations /tao-workspace/data/tube_inspection/val/annotations.json \
      --media /tao-workspace/data/tube_inspection/val --limit 300 --out /tao-workspace/results/<run>/eval_finetuned

  # zero-shot baseline of the converted base model (no adapter):
  python eval_in_container.py --model ... --annotations ... --media ... --limit 300 --out .../eval_baseline

Adapter loading handles three export layouts: PEFT adapter dir (adapter_config.json), raw LoRA A/B safetensors
(merged into the base weights here), or a full merged safetensors checkpoint.

Outputs: <out>/predictions.jsonl (one row per sample) and <out>/metrics.json (accuracy per question category
plus per-field accuracy for the JSON inspection report).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

# ----------------------------------------------------------------------------- model
def load_model(model_dir: str, adapter: str | None, device: str):
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    except ImportError:
        from transformers import AutoModelForImageTextToText as Cls
    # cuDNN fused SDPA has no valid execution plan for some shapes on GB200 -> use flash/math SDPA kernels
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa")
    model.eval()
    if adapter:
        apply_adapter(model, Path(adapter))
    return processor, model


def apply_adapter(model, adir: Path):
    files = sorted(adir.rglob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors under {adir}")
    keys = []
    for f in files:
        with safe_open(str(f), "pt") as sf:
            keys += list(sf.keys())
    lora_keys = [k for k in keys if "lora_A" in k or "lora_B" in k]
    sd = model.state_dict()
    if lora_keys:
        cfg = {}
        for c in (adir / "adapter_config.json", adir.parent / "adapter_config.json"):
            if c.exists():
                cfg = json.loads(c.read_text()); break
        r = cfg.get("r") or cfg.get("lora_rank") or 16
        alpha = cfg.get("lora_alpha") or 32
        scale = alpha / r
        print(f"-- manual LoRA merge: {len(lora_keys)//2} modules, scale alpha/r = {alpha}/{r}")
        print("   sample adapter keys:", lora_keys[:3])
        print("   sample base keys:   ", [k for k in sd if k.endswith("q_proj.weight")][:2])
        tensors = {}
        for f in files:
            with safe_open(str(f), "pt") as sf:
                for k in sf.keys():
                    if "lora_" in k:
                        tensors[k] = sf.get_tensor(k)
        # index base weights by their module path so we can match adapter keys by suffix
        base_by_suffix: dict[str, list[str]] = defaultdict(list)
        for k in sd:
            if k.endswith(".weight"):
                parts = k[:-len(".weight")].split(".")
                for n in range(1, min(6, len(parts)) + 1):
                    base_by_suffix[".".join(parts[-n:])].append(k)

        def find_target(adapter_key: str):
            mod = re.sub(r"\.lora_[AB](\.[^.]+)?\.weight$", "", adapter_key)
            parts = [p for p in mod.split(".") if p not in ("base_model", "policy")]
            for n in range(min(6, len(parts)), 0, -1):
                hits = base_by_suffix.get(".".join(parts[-n:]), [])
                if len(hits) == 1:
                    return hits[0]
            return None

        merged, unmatched = 0, []
        for ka, A in tensors.items():
            if "lora_A" not in ka:
                continue
            kb = ka.replace("lora_A", "lora_B")
            if kb not in tensors:
                continue
            tgt = find_target(ka)
            if tgt is None:
                unmatched.append(ka); continue
            W = sd[tgt]
            delta = (tensors[kb].to(torch.float32) @ A.to(torch.float32)) * scale
            if delta.shape != W.shape:
                unmatched.append(f"{ka} shape {tuple(delta.shape)} vs {tuple(W.shape)}"); continue
            W.add_(delta.to(W.dtype).to(W.device))
            merged += 1
        print(f"-- merged {merged} LoRA deltas into base weights; unmatched: {len(unmatched)}")
        if unmatched[:5]:
            print("   unmatched examples:", unmatched[:5])
        if merged == 0:
            raise SystemExit("LoRA keys found but none matched the base model; see sample keys above")
    else:
        print(f"-- loading full weights from {adir} ({len(keys)} tensors)")
        state = {}
        for f in files:
            with safe_open(str(f), "pt") as sf:
                for k in sf.keys():
                    state[k] = sf.get_tensor(k)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"-- loaded; missing={len(missing)} unexpected={len(unexpected)}")
        if len(unexpected) > len(state) // 2:
            print("   sample unexpected keys:", list(unexpected)[:5])


# ----------------------------------------------------------------------------- inference
@torch.inference_mode()
def answer(processor, model, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return processor.batch_decode([gen], skip_special_tokens=True)[0].strip()


# ----------------------------------------------------------------------------- scoring
def norm(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    return s.strip().strip(".").strip().lower()


def extract_json(s: str):
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def score_report(pred_text: str, gold_text: str) -> dict:
    gold = json.loads(gold_text)
    pred = extract_json(pred_text)
    out = {"json_valid": pred is not None, "vials": len(gold["vials"]), "actual_ok": 0, "status_ok": 0, "fill_ok": 0, "dev_exact": 0, "pass_ok": 0}
    if not pred:
        return out
    pv = {v.get("position"): v for v in pred.get("vials", []) if isinstance(v, dict)}
    for g in gold["vials"]:
        p = pv.get(g["position"], {})
        out["actual_ok"] += int(str(p.get("actual", "")).lower() == g["actual"])
        out["status_ok"] += int(str(p.get("status", "")) == g["status"])
        try:
            out["fill_ok"] += int(abs(float(p.get("fill_pct", -999)) - g["fill_pct"]) <= 10)
        except (TypeError, ValueError):
            pass
    try:
        pdv = sorted(int(x) for x in pred.get("deviating_positions", []))
    except (TypeError, ValueError):
        pdv = None
    out["dev_exact"] = int(pdv == sorted(gold["deviating_positions"]))
    out["pass_ok"] = int(bool(pred.get("pass")) == gold["pass"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--media", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=300, help="number of samples (spread evenly across categories)")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ann = json.loads(Path(a.annotations).read_text())
    by_cat = defaultdict(list)
    for s in ann:
        by_cat[s.get("category", "other")].append(s)
    per = max(1, a.limit // max(1, len(by_cat)))
    subset = [s for c in sorted(by_cat) for s in by_cat[c][:per]]
    print(f"-- {len(subset)} samples across {len(by_cat)} categories ({per} each)")

    processor, model = load_model(a.model, a.adapter, a.device)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    agg = defaultdict(lambda: {"n": 0, "correct": 0})
    rep = defaultdict(int)
    t0 = time.time()
    with open(out_dir / "predictions.jsonl", "w") as fh:
        for i, s in enumerate(subset):
            img = Image.open(Path(a.media) / s["images"][0]).convert("RGB")
            prompt = s["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "")
            gold = s["conversations"][1]["value"]
            pred = answer(processor, model, img, prompt, a.max_new_tokens)
            cat = s.get("category", "other")
            if cat == "report":
                sc = score_report(pred, gold)
                for k, v in sc.items():
                    rep[k] += int(v)
                rep["n"] += 1
                correct = sc["json_valid"] and sc["actual_ok"] == sc["vials"] and sc["status_ok"] == sc["vials"] and sc["dev_exact"] and sc["pass_ok"]
            else:
                correct = norm(pred) == norm(gold) or norm(pred).replace(" ", "") == norm(gold).replace(" ", "")
            agg[cat]["n"] += 1
            agg[cat]["correct"] += int(correct)
            fh.write(json.dumps({"id": s["id"], "image": s["images"][0], "category": cat, "gold": gold, "pred": pred, "correct": bool(correct)}) + "\n")
            if (i + 1) % 20 == 0:
                print(f"   {i+1}/{len(subset)}  {time.time()-t0:.0f}s  " + "  ".join(f"{c}:{v['correct']}/{v['n']}" for c, v in sorted(agg.items())), flush=True)

    metrics = {
        "model": a.model, "adapter": a.adapter, "samples": len(subset), "seconds": round(time.time() - t0, 1),
        "accuracy_by_category": {c: {"n": v["n"], "correct": v["correct"], "accuracy": v["correct"] / v["n"]} for c, v in sorted(agg.items())},
        "overall_accuracy": sum(v["correct"] for v in agg.values()) / max(1, sum(v["n"] for v in agg.values())),
    }
    if rep["n"]:
        nv = max(1, rep["vials"])
        metrics["report_fields"] = {
            "json_valid_rate": rep["json_valid"] / rep["n"],
            "per_vial_content_accuracy": rep["actual_ok"] / nv,
            "per_vial_status_accuracy": rep["status_ok"] / nv,
            "per_vial_fill_within_10pct": rep["fill_ok"] / nv,
            "deviating_set_exact": rep["dev_exact"] / rep["n"],
            "pass_fail_accuracy": rep["pass_ok"] / rep["n"],
            "n_reports": rep["n"], "n_vials": rep["vials"],
        }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
