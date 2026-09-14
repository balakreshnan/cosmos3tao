"""
Vial-filling line inspection — local web app.

Upload a line photo, paste the manufacturing plan, get the fine-tuned Cosmos3-Nano's JSON report, a per-vial
table, and the image annotated with one box per vial coloured by status (green OK, red deviation).

Run:
    python apps/inspection_app.py                      # http://127.0.0.1:7860
    python apps/inspection_app.py --share              # temporary public gradio.live link
    python apps/inspection_app.py --model models/Cosmos3-Nano-qwen3vl --adapter models/adapter_epoch5 --port 7860

Boxes come from the BASE model's Qwen3-VL grounding ability (the LoRA adapter is switched off for that call and
back on for inspection). If grounding returns nothing, evenly spaced placeholder boxes are drawn and labelled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from model_runtime import QUESTIONS, REPORT_Q, Runtime, fallback_boxes  # noqa: E402

OK_COLOR, BAD_COLOR, NEUTRAL = (34, 170, 90), (220, 50, 50), (60, 120, 220)
DEFAULT_PLAN = ("position 1 (VIAL 0174): colored cubes; position 2: colored cubes; position 3: empty; "
                "position 4 (VIAL 0019): orange liquid; position 5: blue liquid; position 6: yellow liquid; "
                "position 7 (VIAL 0042): blue liquid; position 8: orange liquid")


def font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate(image: Image.Image, boxes, report: dict | None, approx: bool) -> Image.Image:
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    lw = max(3, W // 400)
    f = font(max(14, W // 60))
    vials = {v.get("position"): v for v in (report or {}).get("vials", []) if isinstance(v, dict)}
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        v = vials.get(i, {})
        status = v.get("status")
        color = NEUTRAL if status is None else (OK_COLOR if status == "OK" else BAD_COLOR)
        d.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        label = f"#{i}" + (f" {v.get('actual', '?')}" if v else "") + (f" · {status}" if status and status != "OK" else "")
        if v and v.get("fill_pct") not in (None, 0):
            label += f" · {v['fill_pct']}%"
        tw, th = d.textbbox((0, 0), label, font=f)[2:]
        ty = max(0, y1 - th - 8)
        d.rectangle([x1, ty, x1 + tw + 10, ty + th + 6], fill=color)
        d.text((x1 + 5, ty + 3), label, fill="white", font=f)
    if approx:
        d.text((10, H - 30), "boxes approximate (grounding unavailable)", fill=BAD_COLOR, font=f)
    if report is not None:
        verdict = "PASS" if report.get("pass") else f"FAIL · deviating: {report.get('deviating_positions')}"
        tw, th = d.textbbox((0, 0), verdict, font=font(max(18, W // 45)))[2:]
        d.rectangle([W - tw - 30, 10, W - 10, th + 24], fill=OK_COLOR if report.get("pass") else BAD_COLOR)
        d.text((W - tw - 20, 16), verdict, fill="white", font=font(max(18, W // 45)))
    return img


def build_app(rt: Runtime):
    def run(image, plan, qtype, custom_q, draw_boxes):
        if image is None:
            return None, "Upload an image first.", None, ""
        t0 = time.time()
        question = QUESTIONS[qtype] or custom_q or REPORT_Q
        raw, report = rt.inspect(image, plan or "", question)
        t_inspect = time.time() - t0
        n = len(report["vials"]) if report and "vials" in report else None
        boxes, approx = [], False
        if draw_boxes:
            try:
                boxes = rt.locate_vials(image, n)
            except Exception as e:  # grounding is best-effort
                print("grounding failed:", e)
            if n and len(boxes) != n:
                # keep count consistent with the report: trust the report's vial count
                boxes, approx = (boxes[:n] if len(boxes) > n else fallback_boxes(image, n)), len(boxes) < n
            elif not boxes and n:
                boxes, approx = fallback_boxes(image, n), True
        annotated = annotate(image, boxes, report, approx) if boxes else image
        rows = []
        if report:
            for v in report.get("vials", []):
                rows.append([v.get("position"), v.get("vial_id", ""), v.get("planned", ""), v.get("actual", ""),
                             v.get("fill_pct", ""), "✅ OK" if v.get("status") == "OK" else f"❌ {v.get('status')}"])
        summary = (f"**{'PASS ✅' if report.get('pass') else 'FAIL ❌'}** · deviating positions: "
                   f"{report.get('deviating_positions') or 'none'} · {n} vials · inspection {t_inspect:.1f}s"
                   + (f" · boxes {time.time() - t0 - t_inspect:.1f}s" if draw_boxes else "")) if report else f"Answer: **{raw}**  ({t_inspect:.1f}s)"
        return annotated, json.dumps(report, indent=2) if report else raw, rows or None, summary

    with gr.Blocks(title="Vial line inspection · Cosmos3-Nano LoRA", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Vial-filling line inspection\nFine-tuned **Cosmos3-Nano** (NVIDIA TAO · LoRA). Upload a side-view photo, "
                    "give the manufacturing plan, get the inspection report and annotated image.")
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="pil", label="Line photo", height=360)
                plan = gr.Textbox(label="Manufacturing plan (expected content per position)", value=DEFAULT_PLAN, lines=4)
                qtype = gr.Dropdown(list(QUESTIONS), value="Full JSON inspection report", label="Question")
                custom_q = gr.Textbox(label="Custom question (when 'Free-form question' is selected)", lines=2)
                boxes_on = gr.Checkbox(value=True, label="Draw vial bounding boxes (base-model grounding, adds ~10 s)")
                btn = gr.Button("Inspect", variant="primary")
                gr.Examples([[str(p)] for p in [Path("data/real/plant.jpg")] if p.exists()], inputs=[img_in], label="Example")
            with gr.Column(scale=1):
                summary = gr.Markdown()
                img_out = gr.Image(label="Annotated", height=360)
                table = gr.Dataframe(headers=["position", "vial_id", "planned", "actual", "fill %", "status"], label="Per-vial result", interactive=False)
                json_out = gr.Code(label="Model output (JSON)", language="json")
        btn.click(run, [img_in, plan, qtype, custom_q, boxes_on], [img_out, json_out, table, summary])
        gr.Markdown(f"<small>model loaded in {rt.load_seconds:.0f}s · adapter modules: {len(rt.deltas)} · "
                    f"greedy decoding · boxes come from the base model's grounding and may be approximate</small>")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Cosmos3-Nano-qwen3vl")
    ap.add_argument("--adapter", default="models/adapter_epoch5")
    ap.add_argument("--device-map", default="cuda")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    rt = Runtime(a.model, a.adapter, a.device_map)
    build_app(rt).launch(server_name="127.0.0.1", server_port=a.port, share=a.share, inbrowser=not a.share)


if __name__ == "__main__":
    main()
