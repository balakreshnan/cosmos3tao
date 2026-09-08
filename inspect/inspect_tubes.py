"""
Tube-filling inspection client.

Sends an inspection image to a Cosmos 3 reasoner (hosted NVIDIA endpoint by default, or your fine-tuned
Cosmos3-Nano served as a TAO inference microservice / any OpenAI-compatible endpoint) and asks for the
structured inspection report. Optionally scores the answer against ground_truth.json from the generator,
so you can measure the zero-shot baseline before fine-tuning and the fine-tuned model afterwards.

Examples
--------
  # single image, hosted Cosmos 3 reasoner from .env
  python inspect/inspect_tubes.py --image data/tube_inspection/val/images/tubes_val_00000.png

  # score the whole val split (first 50 images) zero-shot
  python inspect/inspect_tubes.py --split data/tube_inspection/val --limit 50

  # same, against your fine-tuned model served by TAO's inference microservice
  python inspect/inspect_tubes.py --split data/tube_inspection/val --base-url http://<host>:8000/v1 --model cosmos3-nano-tube
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROMPT = (
    "You are a quality inspector on a liquid filling line. The strip at the top of the image is the "
    "manufacturing plan: for each tube position it shows the planned color. Below it, a rack of numbered "
    "tubes shows the actual fill result.\n"
    "Produce the inspection report as compact JSON with keys: tubes (list of {position, planned_color, "
    "actual_color, fill_pct, status}), deviating_positions, pass. status is one of OK, wrong_color, "
    "underfill, overfill, empty, contaminated. Color words: red, orange, yellow, green, cyan, blue, purple, "
    "magenta, brown, white, or none for an empty tube. Output JSON only."
)


def image_data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def extract_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def inspect(client: OpenAI, model: str, image: Path, max_tokens: int) -> tuple[dict | None, str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        max_tokens=max_tokens,
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    return extract_json(raw), raw


def score(pred: dict | None, gt: dict) -> dict:
    """Per-image metrics: color accuracy per tube, status accuracy, deviation-set exact match, pass/fail match."""
    n = len(gt["tubes"])
    out = {"tubes": n, "color_ok": 0, "status_ok": 0, "dev_exact": 0, "pass_ok": 0, "parsed": pred is not None}
    if not pred:
        return out
    pt = {t.get("position"): t for t in pred.get("tubes", []) if isinstance(t, dict)}
    for t in gt["tubes"]:
        p = pt.get(t["position"], {})
        out["color_ok"] += int(str(p.get("actual_color", "")).lower() == t["actual_color"])
        out["status_ok"] += int(str(p.get("status", "")) == t["status"])
    pred_dev = sorted(int(x) for x in pred.get("deviating_positions", []) if str(x).isdigit())
    out["dev_exact"] = int(pred_dev == sorted(gt["deviating_positions"]))
    out["pass_ok"] = int(bool(pred.get("pass")) == gt["pass"])
    return out


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="single image to inspect")
    ap.add_argument("--split", help="dataset split folder containing images/ and ground_truth.json")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--base-url", default=os.getenv("NVIDIA_CHAT_BASE_URL", "https://inference-api.nvidia.com/v1"))
    ap.add_argument("--model", default=os.getenv("NVIDIA_REASONER_MODEL", "nvidia/nvidia/cosmos3-super-reasoner"))
    ap.add_argument("--api-key", default=os.getenv("NVIDIA_API_KEY") or "none")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", default="results/inspection_results.jsonl")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"endpoint={args.base_url} model={args.model}", file=sys.stderr)

    if args.image:
        pred, raw = inspect(client, args.model, Path(args.image), args.max_tokens)
        print(json.dumps(pred, indent=2) if pred else raw)
        return

    if not args.split:
        ap.error("give --image or --split")

    split = Path(args.split)
    gts = json.loads((split / "ground_truth.json").read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    totals = {"images": 0, "tubes": 0, "color_ok": 0, "status_ok": 0, "dev_exact": 0, "pass_ok": 0, "parsed": 0}
    with open(args.out, "w") as fh:
        for rel, gt in list(gts.items())[: args.limit]:
            pred, raw = inspect(client, args.model, split / rel, args.max_tokens)
            s = score(pred, gt)
            totals["images"] += 1
            for k in ("tubes", "color_ok", "status_ok", "dev_exact", "pass_ok", "parsed"):
                totals[k] += int(s[k])
            fh.write(json.dumps({"image": rel, "pred": pred, "raw": raw, "score": s}) + "\n")
            print(f"{rel}: color {s['color_ok']}/{s['tubes']} status {s['status_ok']}/{s['tubes']} "
                  f"dev_exact={s['dev_exact']} pass_ok={s['pass_ok']}", file=sys.stderr)

    n, t = totals["images"], max(1, totals["tubes"])
    print(json.dumps({
        "model": args.model,
        "images": n,
        "parse_rate": totals["parsed"] / max(1, n),
        "tube_color_acc": totals["color_ok"] / t,
        "tube_status_acc": totals["status_ok"] / t,
        "deviation_set_exact": totals["dev_exact"] / max(1, n),
        "pass_fail_acc": totals["pass_ok"] / max(1, n),
    }, indent=2))


if __name__ == "__main__":
    main()
