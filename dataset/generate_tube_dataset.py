"""
Synthetic "tube filling line" inspection dataset for Cosmos3-Nano fine-tuning with TAO.

Scenario
--------
A filling station fills a rack of test tubes with colored liquid. Each production run has a
MANUFACTURING PLAN: the expected color for every tube position. The plan is printed as a
color-swatch strip at the top of the image (as a real station would show on its HMI).
The inspection task is to read the plan, look at every tube, and report:
  * the color actually in each tube
  * whether each tube matches the plan
  * the list of tube positions that deviate (wrong color, under-fill, over-fill, empty, contaminated)

Output layout (TAO cosmos-rl "vlm/llava" dataset)
-------------------------------------------------
out/
  train/
    images/*.png
    annotations.json      # LLaVA conversations, TAO fields: id, images, conversations, category, normalized_answer
    ground_truth.json     # per-image plan + actual state (for local scoring, not used by TAO)
    images.tar.gz         # TAO expects images.tar.gz + annotations.json inside the dataset folder
  val/ (same)

Usage
-----
  python dataset/generate_tube_dataset.py --out data/tube_inspection --train 800 --val 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- palette
# Name -> RGB. Names are the vocabulary the model must answer with.
PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (214, 40, 40),
    "orange": (244, 140, 6),
    "yellow": (245, 208, 30),
    "green": (46, 160, 67),
    "cyan": (30, 190, 215),
    "blue": (34, 90, 220),
    "purple": (130, 50, 190),
    "magenta": (215, 55, 170),
    "brown": (120, 72, 30),
    "white": (240, 240, 240),
}
COLOR_NAMES = list(PALETTE)

STATUS_OK = "OK"
DEFECTS = ["wrong_color", "underfill", "overfill", "empty", "contaminated"]


@dataclass
class Tube:
    position: int          # 1-based
    planned_color: str
    actual_color: str      # "none" when empty
    fill_pct: int          # 0..100
    status: str            # OK | wrong_color | underfill | overfill | empty | contaminated
    contaminant_color: str | None = None


# --------------------------------------------------------------------------- helpers
def jitter(rgb, amount=14, rng: random.Random | None = None):
    rng = rng or random
    return tuple(max(0, min(255, c + rng.randint(-amount, amount))) for c in rgb)


def darken(rgb, f=0.7):
    return tuple(int(c * f) for c in rgb)


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sample_run(rng: random.Random, n_tubes: int, defect_rate: float) -> list[Tube]:
    """Create a plan and the 'actual' filling result with random defects."""
    tubes: list[Tube] = []
    for pos in range(1, n_tubes + 1):
        planned = rng.choice(COLOR_NAMES)
        if rng.random() < defect_rate:
            defect = rng.choice(DEFECTS)
        else:
            defect = STATUS_OK

        actual, fill, contaminant = planned, rng.randint(72, 90), None
        if defect == "wrong_color":
            actual = rng.choice([c for c in COLOR_NAMES if c != planned])
        elif defect == "underfill":
            fill = rng.randint(15, 50)
        elif defect == "overfill":
            fill = rng.randint(96, 100)
        elif defect == "empty":
            actual, fill = "none", 0
        elif defect == "contaminated":
            contaminant = rng.choice([c for c in COLOR_NAMES if c != planned])
        tubes.append(Tube(pos, planned, actual, fill, defect, contaminant))
    return tubes


# --------------------------------------------------------------------------- rendering
def render(tubes: list[Tube], rng: random.Random, width=1024, height=640) -> Image.Image:
    img = Image.new("RGB", (width, height), (58, 62, 68))
    d = ImageDraw.Draw(img)
    f_med, f_big = load_font(22), load_font(28)
    f_small = load_font(18 if len(tubes) <= 8 else 14)   # keep "10: MAGENTA" inside its swatch

    # Station background: wall panel + conveyor
    d.rectangle([0, 0, width, height], fill=jitter((60, 64, 70), 6, rng))
    conveyor_y = height - 110
    d.rectangle([0, conveyor_y, width, height], fill=(38, 40, 44))
    for x in range(0, width, 48):  # conveyor slats
        d.rectangle([x, conveyor_y + 8, x + 40, conveyor_y + 14], fill=(70, 72, 76))

    # HMI plan strip at top
    strip_h = 92
    d.rectangle([16, 12, width - 16, 12 + strip_h], fill=(20, 22, 26), outline=(120, 124, 130), width=2)
    run_id = rng.randint(10000, 99999)
    d.text((28, 20), f"FILL PLAN  RUN-{run_id}   LINE {rng.randint(1, 6)}", fill=(230, 230, 230), font=f_med)
    n = len(tubes)
    slot_w = (width - 64) / n
    for t in tubes:
        x0 = 32 + (t.position - 1) * slot_w
        d.rectangle([x0 + 6, 52, x0 + slot_w - 6, 96], fill=PALETTE[t.planned_color], outline=(200, 200, 200))
        label = f"{t.position}: {t.planned_color.upper()}"
        tc = (20, 20, 20) if t.planned_color in ("white", "yellow", "cyan") else (245, 245, 245)
        d.text((x0 + 12, 64), label, fill=tc, font=f_small)

    # Rack
    rack_top = 150
    rack_h = conveyor_y - rack_top - 10
    d.rounded_rectangle([40, rack_top, width - 40, rack_top + rack_h], radius=14, fill=(92, 96, 104), outline=(30, 30, 34), width=3)

    # Nozzles + tubes
    tube_w = int(min(70, slot_w * 0.55))
    tube_h = int(rack_h * 0.80)
    tube_top = rack_top + int(rack_h * 0.12)
    for t in tubes:
        cx = int(40 + (t.position - 0.5) * (width - 80) / n)
        x0, x1 = cx - tube_w // 2, cx + tube_w // 2
        y0, y1 = tube_top, tube_top + tube_h

        # nozzle
        d.rectangle([cx - 8, rack_top - 26, cx + 8, rack_top + 4], fill=(150, 152, 158))
        d.polygon([(cx - 8, rack_top + 4), (cx + 8, rack_top + 4), (cx, rack_top + 16)], fill=(120, 122, 126))

        # glass body (slightly translucent look)
        d.rounded_rectangle([x0, y0, x1, y1], radius=tube_w // 2, fill=(180, 186, 194), outline=(235, 238, 242), width=2)
        inner = [x0 + 4, y0 + 4, x1 - 4, y1 - 4]

        # liquid
        if t.fill_pct > 0:
            liq_top = inner[3] - int((inner[3] - inner[1]) * t.fill_pct / 100)
            col = jitter(PALETTE[t.actual_color], 12, rng)
            d.rounded_rectangle([inner[0], liq_top, inner[2], inner[3]], radius=tube_w // 2 - 4, fill=col)
            # meniscus
            d.ellipse([inner[0], liq_top - 5, inner[2], liq_top + 5], fill=darken(col, 0.85))
            if t.status == "contaminated" and t.contaminant_color:
                band_h = max(10, (inner[3] - liq_top) // 5)
                by = rng.randint(liq_top + 6, max(liq_top + 6, inner[3] - band_h - 6))
                d.rectangle([inner[0] + 2, by, inner[2] - 2, by + band_h], fill=jitter(PALETTE[t.contaminant_color], 8, rng))
        # glare
        d.line([(x0 + 8, y0 + 18), (x0 + 8, y1 - 24)], fill=(250, 250, 250), width=3)
        # fill-level tick marks
        for pct in (25, 50, 75, 100):
            ty = inner[3] - int((inner[3] - inner[1]) * pct / 100)
            d.line([(x1 + 3, ty), (x1 + 10, ty)], fill=(220, 220, 220), width=1)

        # position label on conveyor
        d.rounded_rectangle([cx - 20, conveyor_y + 30, cx + 20, conveyor_y + 62], radius=6, fill=(230, 230, 230))
        d.text((cx - 7 if t.position < 10 else cx - 13, conveyor_y + 34), str(t.position), fill=(20, 20, 20), font=f_big)

    # subtle noise for realism
    px = img.load()
    for _ in range(width * height // 40):
        x, y = rng.randrange(width), rng.randrange(height)
        r, g, b = px[x, y]
        v = rng.randint(-10, 10)
        px[x, y] = (max(0, min(255, r + v)), max(0, min(255, g + v)), max(0, min(255, b + v)))
    return img


# --------------------------------------------------------------------------- annotations
SYSTEM_CONTEXT = (
    "You are a quality inspector on a liquid filling line. The strip at the top of the image is the "
    "manufacturing plan: for each tube position it shows the planned color. Below it, a rack of numbered "
    "tubes shows the actual fill result."
)


def deviations(tubes: list[Tube]) -> list[int]:
    return [t.position for t in tubes if t.status != STATUS_OK]


def report_json(tubes: list[Tube]) -> dict:
    return {
        "tubes": [
            {
                "position": t.position,
                "planned_color": t.planned_color,
                "actual_color": t.actual_color,
                "fill_pct": t.fill_pct,
                "status": t.status,
            }
            for t in tubes
        ],
        "deviating_positions": deviations(tubes),
        "pass": len(deviations(tubes)) == 0,
    }


def make_samples(image_rel: str, tubes: list[Tube], rng: random.Random) -> list[dict]:
    """Return several LLaVA-format QA samples for one image (TAO cosmos-rl schema)."""
    samples = []

    def add(question: str, answer: str, category: str):
        sid = hashlib.md5(f"{image_rel}|{question}".encode()).hexdigest()
        samples.append(
            {
                "id": sid,
                "images": [image_rel],
                "conversations": [
                    {"from": "human", "value": f"<image>\n{SYSTEM_CONTEXT}\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
                "category": category,
                "normalized_answer": answer,
            }
        )

    # 1) color of a random tube
    t = rng.choice(tubes)
    add(
        f"What color is the liquid in tube {t.position}? Answer with a single color word, or 'none' if the tube is empty.",
        t.actual_color,
        "color",
    )

    # 2) plan color of a random tube (reading the HMI strip)
    t = rng.choice(tubes)
    add(f"According to the manufacturing plan, what color should tube {t.position} contain? Answer with one word.",
        t.planned_color, "plan_color")

    # 3) yes/no match for a random tube
    t = rng.choice(tubes)
    add(
        f"Does tube {t.position} match the manufacturing plan (correct color and a normal fill level)? Answer yes or no.",
        "no" if t.status != STATUS_OK else "yes",
        "match",
    )

    # 4) list of deviating positions
    dev = deviations(tubes)
    add(
        "List the tube positions that do NOT match the manufacturing plan, as comma-separated integers in "
        "ascending order. Answer 'none' if every tube matches.",
        ",".join(map(str, dev)) if dev else "none",
        "deviation_list",
    )

    # 5) full structured report
    add(
        "Produce the inspection report as compact JSON with keys: tubes (list of {position, planned_color, "
        "actual_color, fill_pct, status}), deviating_positions, pass. status is one of OK, wrong_color, "
        "underfill, overfill, empty, contaminated. Output JSON only.",
        json.dumps(report_json(tubes), separators=(",", ":")),
        "report",
    )
    return samples


# --------------------------------------------------------------------------- main
def build_split(name: str, count: int, out_root: Path, rng: random.Random, n_tubes_choices, defect_rate):
    split_dir = out_root / name
    img_dir = split_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    annotations, ground_truth = [], {}
    for i in range(count):
        n_tubes = rng.choice(n_tubes_choices)
        tubes = sample_run(rng, n_tubes, defect_rate)
        img = render(tubes, rng)
        fname = f"tubes_{name}_{i:05d}.png"
        img.save(img_dir / fname)
        rel = f"images/{fname}"
        annotations.extend(make_samples(rel, tubes, rng))
        ground_truth[rel] = report_json(tubes)

    (split_dir / "annotations.json").write_text(json.dumps(annotations, indent=1))
    (split_dir / "ground_truth.json").write_text(json.dumps(ground_truth, indent=1))

    with tarfile.open(split_dir / "images.tar.gz", "w:gz") as tar:
        tar.add(img_dir, arcname="images")

    print(f"[{name}] {count} images -> {len(annotations)} samples at {split_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/tube_inspection")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--defect-rate", type=float, default=0.22, help="per-tube probability of a defect")
    ap.add_argument("--tubes", default="6,8,10", help="comma list of possible tube counts per rack")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_choices = [int(x) for x in args.tubes.split(",")]
    out = Path(args.out)
    build_split("train", args.train, out, rng, n_choices, args.defect_rate)
    build_split("val", args.val, out, rng, n_choices, args.defect_rate)
    print("Done. Upload each split folder (images.tar.gz + annotations.json) to the cluster/cloud storage "
          "referenced by TRAIN_DATASET_URI / EVAL_DATASET_URI.")


if __name__ == "__main__":
    main()
