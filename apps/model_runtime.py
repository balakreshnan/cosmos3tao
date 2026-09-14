"""
Model runtime shared by the web app: loads the Qwen3-VL-layout Cosmos3-Nano base once, keeps the LoRA deltas
so the adapter can be switched on (inspection questions) and off (vial grounding with the base model's
native Qwen3-VL localisation ability) without a second copy of the weights.

    rt = Runtime("models/Cosmos3-Nano-qwen3vl", "models/adapter_epoch5")
    report = rt.inspect(image, plan_text)          # fine-tuned adapter ON  -> dict (JSON report)
    boxes  = rt.locate_vials(image, n_expected)    # adapter OFF            -> [(x1,y1,x2,y2), ...] left->right
"""

from __future__ import annotations

import json
import re
import threading
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
QUESTIONS = {
    "Full JSON inspection report": REPORT_Q,
    "List deviating positions": "List the positions whose vial does NOT match the manufacturing plan, as comma-separated integers in ascending order. Answer 'none' if all match.",
    "Count vials": "How many vials are visible on the track? Answer with an integer.",
    "Free-form question": "",
}
# Qwen3-VL grounding; the base model answers with coordinates normalised to 0-1000.
# Two phrasings: the first gives full-height boxes on real photos but sometimes malformed JSON on renders,
# the second is always well-formed but tends to stop at the clamp ring. We try both and keep the better one.
GROUND_PROMPTS = [
    "Locate every vial (glass tube) on the track in this image. Output only a JSON list, one entry per vial "
    "ordered from left to right, each as {\"position\": <int>, \"bbox_2d\": [x1, y1, x2, y2]} with the bounding box "
    "of the whole vial from its top opening to its bottom.",
    "Detect all glass vials (test tubes) on the track in the image and output their bounding boxes as a JSON "
    "list of {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"vial\"}.",
]
GROUND_Q = GROUND_PROMPTS[0]
# four numbers followed by ']' — tolerates the malformed `"position": 130, 300, 150, 641]` variant
BOX_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


class Runtime:
    def __init__(self, model_dir: str, adapter_dir: str | None, device_map: str = "cuda"):
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
        self.processor = AutoProcessor.from_pretrained(model_dir)
        kw = {"device_map": device_map, "attn_implementation": "sdpa"}
        try:
            self.model = Cls.from_pretrained(model_dir, dtype=torch.bfloat16, **kw)
        except TypeError:
            self.model = Cls.from_pretrained(model_dir, torch_dtype=torch.bfloat16, **kw)
        self.model.eval()
        self.lock = threading.Lock()
        self.deltas: dict[str, torch.Tensor] = {}
        self.adapter_on = False
        if adapter_dir and Path(adapter_dir).exists():
            self._load_deltas(Path(adapter_dir))
            self.set_adapter(True)
        self.load_seconds = time.time() - t0

    # ---- LoRA handling ---------------------------------------------------------------------------------
    def _load_deltas(self, adir: Path):
        cfg = json.loads((adir / "adapter_config.json").read_text()) if (adir / "adapter_config.json").exists() else {}
        scale = cfg.get("lora_alpha", 32) / cfg.get("r", 16)
        sd = self.model.state_dict()
        by_suffix: dict[str, list[str]] = defaultdict(list)
        for k in sd:
            if k.endswith(".weight"):
                parts = k[:-7].split(".")
                for n in range(1, min(6, len(parts)) + 1):
                    by_suffix[".".join(parts[-n:])].append(k)
        tensors = {}
        for f in sorted(adir.rglob("*.safetensors")):
            with safe_open(str(f), "pt") as sf:
                for k in sf.keys():
                    if "lora_" in k:
                        tensors[k] = sf.get_tensor(k)
        for ka, A in tensors.items():
            if "lora_A" not in ka:
                continue
            kb = ka.replace("lora_A", "lora_B")
            mod = re.sub(r"\.lora_[AB](\.[^.]+)?\.weight$", "", ka)
            parts = [p for p in mod.split(".") if p not in ("base_model", "policy")]
            tgt = None
            for n in range(min(6, len(parts)), 0, -1):
                hits = by_suffix.get(".".join(parts[-n:]), [])
                if len(hits) == 1:
                    tgt = hits[0]; break
            if kb in tensors and tgt is not None:
                W = sd[tgt]
                self.deltas[tgt] = ((tensors[kb].to(torch.float32) @ A.to(torch.float32)) * scale).to(W.dtype).to(W.device)
        self.sd = sd
        print(f"-- LoRA deltas prepared for {len(self.deltas)} modules")

    @torch.inference_mode()
    def set_adapter(self, on: bool):
        if on == self.adapter_on or not self.deltas:
            return
        for k, d in self.deltas.items():
            self.sd[k].add_(d if on else -d)
        self.adapter_on = on

    # ---- generation ------------------------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 768, adapter: bool = True) -> tuple[str, tuple[int, int]]:
        """Returns (text, (resized_w, resized_h)) — the processor's working resolution, needed for boxes."""
        with self.lock:
            self.set_adapter(adapter)
            msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
            grid = inputs.get("image_grid_thw")
            patch = getattr(self.processor.image_processor, "patch_size", 16)
            resized = (int(grid[0, 2]) * patch, int(grid[0, 1]) * patch) if grid is not None else image.size
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen = out[0, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode([gen], skip_special_tokens=True)[0].strip(), resized

    def inspect(self, image: Image.Image, plan: str, question: str = REPORT_Q) -> tuple[str, dict | None]:
        prompt = CONTEXT + (f"\nManufacturing plan: {plan}" if plan.strip() else "") + "\n" + question
        text, _ = self.generate(image, prompt, adapter=True)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return text, json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            return text, None

    def locate_vials(self, image: Image.Image, n_expected: int | None = None) -> list[tuple[int, int, int, int]]:
        """Ask the BASE model (adapter off) for vial boxes; returns pixel boxes in the original image, left→right."""
        best, best_score = None, None
        for prompt in GROUND_PROMPTS:
            text, (rw, rh) = self.generate(image, prompt, max_new_tokens=512, adapter=False)
            cand = [[float(v) for v in m.groups()] for m in BOX_RE.finditer(text)]
            cand = [b for b in cand if b[2] > b[0] and b[3] > b[1]]
            if not cand:
                continue
            # prefer the answer whose count matches the report; tie-break on taller boxes (whole vial, not just glass)
            score = (-(abs(len(cand) - n_expected) if n_expected else 0), sum(b[3] - b[1] for b in cand) / len(cand))
            if best is None or score > best_score:
                best, best_score = (cand, rw, rh), score
            if n_expected and len(cand) in (n_expected, n_expected + 1) and prompt is GROUND_PROMPTS[0]:
                break  # first prompt already good; skip the second call
        if best is None:
            return []
        boxes, rw, rh = best
        W, H = image.size
        mx = max(max(b[2] for b in boxes), max(b[3] for b in boxes))
        if mx <= 1.0:                                                    # normalised 0-1
            sx, sy = W, H
        elif mx <= 1000:                                                 # Qwen3-VL convention: normalised 0-1000
            sx, sy = W / 1000.0, H / 1000.0
        elif abs(mx - max(rw, rh)) < abs(mx - max(W, H)):                # pixels of the resized working image
            sx, sy = W / rw, H / rh
        else:                                                            # original pixels
            sx, sy = 1.0, 1.0
        px = [(int(max(0, b[0] * sx)), int(max(0, b[1] * sy)), int(min(W, b[2] * sx)), int(min(H, b[3] * sy))) for b in boxes]
        px.sort(key=lambda b: b[0])
        # drop near-duplicates (same vial reported twice)
        dedup = []
        for b in px:
            if not dedup or b[0] - dedup[-1][0] > 0.4 * (dedup[-1][2] - dedup[-1][0]):
                dedup.append(b)
        # more boxes than vials in the report: remove fragments clipped by the image border first
        # (a half-visible vial at the frame edge is not part of the plan), then the narrowest leftovers
        if n_expected and len(dedup) > n_expected:
            widths = sorted(b[2] - b[0] for b in dedup)
            med = widths[len(widths) // 2]
            edge = [b for b in dedup if (b[0] <= 2 or b[2] >= W - 2) and (b[2] - b[0]) < 0.8 * med]
            for b in edge:
                if len(dedup) > n_expected:
                    dedup.remove(b)
            while len(dedup) > n_expected:
                dedup.remove(min(dedup, key=lambda b: b[2] - b[0]))
        return dedup


def fallback_boxes(image: Image.Image, n: int) -> list[tuple[int, int, int, int]]:
    """Evenly spaced vertical strips over the middle band of the image, used when grounding fails."""
    W, H = image.size
    if n <= 0:
        return []
    pitch = W / n
    return [(int(i * pitch + pitch * 0.15), int(H * 0.2), int((i + 1) * pitch - pitch * 0.15), int(H * 0.8)) for i in range(n)]
