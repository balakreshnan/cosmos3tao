"""
Run the fine-tuned Cosmos3-Nano (Qwen3-VL layout + LoRA adapter) locally on one GPU and inspect an image.

Setup (Windows, once):  pip install --index-url https://download.pytorch.org/whl/cu130 torch
                        pip install "transformers>=4.57" accelerate safetensors peft pillow
Weights (downloaded from the cluster, see docs/12-inference-and-next-steps.md):
    models/Cosmos3-Nano-qwen3vl/        converted base checkpoint (~17 GB)
    models/adapter_epoch5/              LoRA adapter (adapter_config.json + *.safetensors, ~100 MB)

Examples
--------
  # one synthetic validation image with its plan taken from ground_truth.json
  python inspect/local_infer.py --image data/tube_inspection/val/images/vials_val_00004.png --ground-truth data/tube_inspection/val/ground_truth.json

  # the real plant photo with a hand-written plan
  python inspect/local_infer.py --image data/real/plant.jpg --plan "position 1 (VIAL 0174): colored cubes; position 2: colored cubes; position 3: empty; position 4 (VIAL 0019): orange liquid; position 5: blue liquid; position 6: yellow liquid; position 7 (VIAL 0042): blue liquid; position 8: orange liquid"

  # any question instead of the JSON report
  python inspect/local_infer.py --image ... --plan "..." --question "List the positions that do NOT match the manufacturing plan, as comma-separated integers in ascending order. Answer 'none' if all match."

  # base model only (no adapter), for comparison
  python inspect/local_infer.py --image ... --plan "..." --no-adapter

  # write a merged standalone checkpoint once, then load it later with --model models/Cosmos3-Nano-tube-merged --no-adapter
  python inspect/local_infer.py --merge-out models/Cosmos3-Nano-tube-merged

Memory: bf16 weights are ~16.5 GB; a 24 GB GPU handles one image + a ~1.5k-token prompt. If you hit CUDA OOM,
add --device-map auto (CPU offload, slower) or lower --max-pixels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

CONTEXT = (
    "You are a quality inspector watching a vial filling line from the side. Each wheeled carrier on the track "
    "holds one clear vial with a printed label. Vials are numbered by position from left to right. "
    "A correctly filled liquid vial is filled to roughly one third of its height."
)
REPORT_Q = (
    "Produce the inspection report as compact JSON with keys: vials (list of {position, vial_id, planned, actual, "
    "fill_pct, status}), deviating_positions, pass. status is one of OK, wrong_color, underfill, overfill, empty, "
    "wrong_content. Output JSON only."
)


def load(model_dir: str, adapter: str | None, device_map: str):
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    except ImportError:
        from transformers import AutoModelForImageTextToText as Cls
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_dir)
    kw = {"device_map": device_map, "attn_implementation": "sdpa"}
    try:
        model = Cls.from_pretrained(model_dir, dtype=torch.bfloat16, **kw)          # transformers >= 5
    except TypeError:
        model = Cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16, **kw)    # transformers 4.x
    model.eval()
    print(f"-- base model loaded in {time.time() - t0:.0f}s on {model.device}", file=sys.stderr)
    if adapter:
        merge_adapter(model, Path(adapter))
    return processor, model


def merge_adapter(model, adir: Path):
    """Merge LoRA A/B into the base weights (W += alpha/r * B@A). Works for cosmos-rl's PEFT-style export."""
    files = sorted(adir.rglob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors under {adir}")
    cfg = json.loads((adir / "adapter_config.json").read_text()) if (adir / "adapter_config.json").exists() else {}
    r, alpha = cfg.get("r", 16), cfg.get("lora_alpha", 32)
    scale = alpha / r
    sd = model.state_dict()
    by_suffix: dict[str, list[str]] = defaultdict(list)
    for k in sd:
        if k.endswith(".weight"):
            parts = k[:-7].split(".")
            for n in range(1, min(6, len(parts)) + 1):
                by_suffix[".".join(parts[-n:])].append(k)

    def target(adapter_key):
        mod = re.sub(r"\.lora_[AB](\.[^.]+)?\.weight$", "", adapter_key)
        parts = [p for p in mod.split(".") if p not in ("base_model", "policy")]
        for n in range(min(6, len(parts)), 0, -1):
            hits = by_suffix.get(".".join(parts[-n:]), [])
            if len(hits) == 1:
                return hits[0]
        return None

    tensors = {}
    for f in files:
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                if "lora_" in k:
                    tensors[k] = sf.get_tensor(k)
    merged = 0
    for ka, A in tensors.items():
        if "lora_A" not in ka:
            continue
        kb = ka.replace("lora_A", "lora_B")
        tgt = target(ka)
        if kb not in tensors or tgt is None:
            continue
        W = sd[tgt]
        delta = (tensors[kb].to(torch.float32) @ A.to(torch.float32)) * scale
        W.add_(delta.to(W.dtype).to(W.device))
        merged += 1
    print(f"-- merged {merged} LoRA modules from {adir} (alpha/r = {alpha}/{r})", file=sys.stderr)
    if merged == 0:
        raise SystemExit("adapter keys did not match the base model")


@torch.inference_mode()
def ask(processor, model, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0, inputs["input_ids"].shape[1]:]
    txt = processor.batch_decode([gen], skip_special_tokens=True)[0].strip()
    print(f"-- {len(gen)} tokens in {time.time() - t0:.1f}s", file=sys.stderr)
    return txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Cosmos3-Nano-qwen3vl")
    ap.add_argument("--adapter", default="models/adapter_epoch5")
    ap.add_argument("--no-adapter", action="store_true", help="base model only")
    ap.add_argument("--image")
    ap.add_argument("--plan", help="manufacturing plan text; omit with --ground-truth to look it up")
    ap.add_argument("--ground-truth", help="ground_truth.json to fetch the plan and compare the answer")
    ap.add_argument("--question", default=REPORT_Q)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--max-pixels", type=int, default=313600, help="image token budget (matches training)")
    ap.add_argument("--device-map", default="cuda", help="'cuda' or 'auto' (CPU offload if the GPU is too small)")
    ap.add_argument("--merge-out", help="save a merged standalone checkpoint to this dir and exit")
    a = ap.parse_args()

    processor, model = load(a.model, None if a.no_adapter else a.adapter, a.device_map)
    try:
        processor.image_processor.max_pixels = a.max_pixels
    except Exception:
        pass

    if a.merge_out:
        Path(a.merge_out).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(a.merge_out, safe_serialization=True)
        processor.save_pretrained(a.merge_out)
        print(f"merged model written to {a.merge_out}")
        return
    if not a.image:
        ap.error("--image is required (or --merge-out)")

    plan = a.plan
    gt = None
    if a.ground_truth:
        gts = json.loads(Path(a.ground_truth).read_text())
        key = next((k for k in gts if Path(k).name == Path(a.image).name), None)
        if key:
            gt = gts[key]
            plan = plan or gt["plan_text"]
    prompt = CONTEXT + ("\nManufacturing plan: " + plan if plan else "") + "\n" + a.question
    image = Image.open(a.image).convert("RGB")
    answer = ask(processor, model, image, prompt, a.max_new_tokens)

    m = re.search(r"\{.*\}", answer, re.S)
    if m:
        try:
            pred = json.loads(m.group(0))
            print(json.dumps(pred, indent=2))
            if gt:
                wrong = [(g["position"], g["actual"], g["status"]) for g in gt["vials"]
                         if not any(v.get("position") == g["position"] and v.get("actual") == g["actual"] and v.get("status") == g["status"] for v in pred.get("vials", []))]
                print(f"\nground truth: deviating={gt['deviating_positions']} pass={gt['pass']}  |  "
                      f"{'ALL VIALS CORRECT' if not wrong else 'mismatched vials: ' + str(wrong)}", file=sys.stderr)
            return
        except json.JSONDecodeError:
            pass
    print(answer)


if __name__ == "__main__":
    main()
