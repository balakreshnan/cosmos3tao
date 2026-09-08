"""
Synthetic side-view dataset of a vial filling line (MagneMotion-style linear track) for Cosmos3-Nano
fine-tuning with TAO.

Modeled on the real plant camera: a horizontal stainless track with individual wheeled carriers, each
clamping one tall clear vial with a printed label ("20200232 VIAL 0019" + QR). Vials contain either
colored liquid filling roughly the bottom third, nothing (empty), or a stack of colored plastic cubes.
A glass safety shield with vertical frame bars sits between camera and track.

The MANUFACTURING PLAN is not visible in the plant image, so it is supplied as text in the prompt
("expected content per vial"), and the model must compare what it sees against that plan.

Output layout (TAO cosmos-rl "vlm/llava" dataset)
-------------------------------------------------
out/<split>/
  images/*.png
  annotations.json      # id, images, conversations, category, normalized_answer
  ground_truth.json     # per-image plan + actual state (used by inspect/inspect_tubes.py)
  images.tar.gz

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
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --------------------------------------------------------------------------- vocab
LIQUIDS: dict[str, tuple[int, int, int]] = {
    "orange": (232, 96, 10),
    "blue": (20, 110, 225),
    "yellow": (238, 200, 20),
    "red": (205, 30, 30),
    "green": (40, 165, 80),
    "purple": (120, 50, 180),
    "clear": (215, 225, 235),
}
CUBE_COLORS = {"red": (215, 45, 40), "blue": (35, 90, 210), "yellow": (240, 200, 40)}

# content words the model answers with
CONTENTS = list(LIQUIDS) + ["cubes", "empty"]
STATUS_OK = "OK"
DEFECTS = ["wrong_color", "underfill", "overfill", "empty", "wrong_content"]

NORMAL_FILL = (28, 42)      # percent of vial height for a good liquid fill (real line ~1/3)
LOT = "20200232"


@dataclass
class Vial:
    position: int            # 1-based, left to right
    vial_id: str             # e.g. "VIAL 0019"
    planned: str             # planned content: a liquid color, "cubes" or "empty"
    actual: str              # actual content word
    fill_pct: int            # liquid height %, 0 for empty/cubes
    status: str


# --------------------------------------------------------------------------- helpers
def jitter(rgb, amt=10, rng: random.Random | None = None):
    rng = rng or random
    return tuple(max(0, min(255, c + rng.randint(-amt, amt))) for c in rgb)


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sample_run(rng: random.Random, n: int, defect_rate: float) -> list[Vial]:
    start = rng.randint(1, 180)
    vials = []
    for i in range(n):
        pos = i + 1
        vid = f"VIAL {start + i:04d}"
        planned = rng.choices(["liquid", "cubes", "empty"], weights=[0.7, 0.15, 0.15])[0]
        if planned == "liquid":
            planned = rng.choice([c for c in LIQUIDS if c != "clear"] + ["clear"])

        status = rng.choice(DEFECTS) if rng.random() < defect_rate else STATUS_OK
        actual, fill = planned, 0
        if planned in LIQUIDS:
            fill = rng.randint(*NORMAL_FILL)
            if status == "wrong_color":
                actual = rng.choice([c for c in LIQUIDS if c != planned])
            elif status == "underfill":
                fill = rng.randint(4, 16)
            elif status == "overfill":
                fill = rng.randint(58, 80)
            elif status == "empty":
                actual, fill = "empty", 0
            elif status == "wrong_content":
                actual, fill = "cubes", 0
        elif planned == "cubes":
            if status in ("wrong_color", "underfill", "overfill"):
                status = STATUS_OK          # not meaningful for cubes
            elif status == "empty":
                actual = "empty"
            elif status == "wrong_content":
                actual = rng.choice([c for c in LIQUIDS if c != "clear"])
                fill = rng.randint(*NORMAL_FILL)
        else:  # planned empty (e.g. a placeholder / not yet filled)
            if status in ("wrong_color", "underfill", "overfill", "empty"):
                status = STATUS_OK
            elif status == "wrong_content":
                actual = rng.choice(["cubes"] + [c for c in LIQUIDS if c != "clear"])
                fill = 0 if actual == "cubes" else rng.randint(*NORMAL_FILL)
        vials.append(Vial(pos, vid, planned, actual, fill, status))
    return vials


# --------------------------------------------------------------------------- rendering
def render(vials: list[Vial], rng: random.Random, W=1400, H=700) -> Image.Image:
    img = Image.new("RGB", (W, H), (236, 236, 232))
    d = ImageDraw.Draw(img)
    f_lab = load_font(11)

    # ---- back wall + background clutter (cabinets / trays)
    d.rectangle([0, 0, W, int(H * 0.42)], fill=jitter((228, 228, 224), 5, rng))
    for _ in range(rng.randint(2, 4)):
        x0 = rng.randint(0, W - 200); w = rng.randint(140, 320); y0 = rng.randint(40, 170)
        d.rectangle([x0, y0, x0 + w, y0 + rng.randint(60, 120)], fill=jitter((205, 208, 210), 12, rng),
                    outline=(170, 172, 175))
    # yellow-capped tray in the back right (like the real photo)
    tx = rng.randint(int(W * 0.55), int(W * 0.8))
    d.rectangle([tx, 90, tx + 260, 140], fill=(40, 40, 44))
    for cx in range(tx + 14, tx + 250, 24):
        d.ellipse([cx, 100, cx + 16, 116], fill=(240, 205, 40))

    # ---- track body (horizontal bands): steel top rail, dark motor band, steel skirt
    track_top = int(H * 0.42)
    d.rectangle([0, track_top, W, track_top + 60], fill=(178, 182, 186))          # brushed top
    d.rectangle([0, track_top + 60, W, track_top + 110], fill=(48, 50, 54))        # dark band
    d.rectangle([0, track_top + 110, W, track_top + 190], fill=(150, 154, 158))    # rail housing
    d.rectangle([0, track_top + 190, W, track_top + 205], fill=(90, 92, 96))       # rail edge
    d.rectangle([0, track_top + 205, W, H], fill=(120, 124, 128))                  # skirt
    for y in range(track_top, track_top + 60, 4):                                  # brushed texture
        d.line([(0, y), (W, y)], fill=jitter((178, 182, 186), 6, rng), width=1)
    # bolt heads on housing
    for x in range(20, W, 46):
        d.ellipse([x, track_top + 150, x + 8, track_top + 158], fill=(80, 82, 86))
    d.text((int(W * 0.62), track_top + 74), "MM LITE", fill=(200, 200, 200), font=load_font(16))

    # ---- carriers with vials
    n = len(vials)
    base_y = track_top + 150            # top of carrier block
    pitch = (W - 120) / n
    vial_h = rng.randint(230, 270)
    vial_w = 48
    for v in vials:
        cx = int(60 + (v.position - 0.5) * pitch + rng.randint(-8, 8))
        # carrier block + wheels
        d.rectangle([cx - 46, base_y, cx + 46, base_y + 40], fill=(190, 194, 198), outline=(110, 112, 116))
        d.rectangle([cx - 30, base_y - 6, cx + 30, base_y + 6], fill=(205, 208, 212))
        for wx in (cx - 34, cx + 34):
            d.ellipse([wx - 11, base_y + 30, wx + 11, base_y + 52], fill=(30, 30, 32))
            d.ellipse([wx - 5, base_y + 36, wx + 5, base_y + 46], fill=(120, 122, 126))
        # vertical aluminum bracket behind vial + black clamp ring
        d.rectangle([cx - 52, base_y - vial_h + 40, cx - 42, base_y], fill=(200, 203, 207), outline=(140, 142, 146))
        d.rectangle([cx + 42, base_y - vial_h + 40, cx + 52, base_y], fill=(200, 203, 207), outline=(140, 142, 146))
        ring_y = base_y - vial_h + 60
        d.rectangle([cx - 60, ring_y, cx + 60, ring_y + 22], fill=(28, 28, 30))

        # vial body (clear): light translucent tube
        x0, x1 = cx - vial_w // 2, cx + vial_w // 2
        y0, y1 = base_y - vial_h, base_y + 8
        d.rounded_rectangle([x0, y0, x1, y1], radius=vial_w // 2, fill=(206, 214, 222), outline=(150, 158, 166), width=2)
        inner = [x0 + 3, y0 + 3, x1 - 3, y1 - 3]
        ih = inner[3] - inner[1]

        # contents
        if v.actual in LIQUIDS and v.fill_pct > 0:
            top = inner[3] - int(ih * v.fill_pct / 100)
            col = jitter(LIQUIDS[v.actual], 8, rng)
            d.rounded_rectangle([inner[0], top, inner[2], inner[3]], radius=vial_w // 2 - 3, fill=col)
            d.ellipse([inner[0], top - 4, inner[2], top + 4], fill=tuple(int(c * 0.8) for c in col))
            d.line([(inner[0] + 6, top + 8), (inner[0] + 6, inner[3] - 10)], fill=tuple(min(255, c + 60) for c in col), width=3)
        elif v.actual == "cubes":
            cube = vial_w - 12
            y = inner[3] - cube - 2
            names = list(CUBE_COLORS)
            while y > inner[1] + ih * 0.15:
                c = CUBE_COLORS[rng.choice(names)]
                d.rectangle([cx - cube // 2, y, cx + cube // 2, y + cube - 2], fill=jitter(c, 8, rng),
                            outline=tuple(int(k * 0.6) for k in c))
                d.polygon([(cx - cube // 2, y), (cx + cube // 2, y), (cx + cube // 2 - 6, y - 5), (cx - cube // 2 + 6, y - 5)],
                          fill=tuple(min(255, k + 40) for k in c))
                y -= cube + 1
        # glass highlights
        d.line([(x0 + 7, y0 + 14), (x0 + 7, y1 - 20)], fill=(245, 248, 250), width=3)
        d.line([(x1 - 8, y0 + 20), (x1 - 8, y1 - 30)], fill=(235, 238, 240), width=1)

        # label with lot / vial id and QR square
        ly = ring_y - 46
        d.rectangle([x0 + 4, ly, x1 - 4, ly + 40], fill=(245, 245, 242), outline=(180, 180, 176))
        d.text((x0 + 7, ly + 3), LOT, fill=(40, 40, 40), font=f_lab)
        d.text((x0 + 7, ly + 16), v.vial_id, fill=(40, 40, 40), font=f_lab)
        qx = x0 + 8
        for i in range(4):
            for j in range(2):
                if rng.random() < 0.6:
                    d.rectangle([qx + i * 3, ly + 30 + j * 3, qx + i * 3 + 2, ly + 30 + j * 3 + 2], fill=(30, 30, 30))

    # ---- safety shield: vertical frame bars + faint glass tint
    shield = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shield)
    for _ in range(rng.randint(2, 3)):
        bx = rng.randint(40, W - 40)
        sd.rectangle([bx, 0, bx + rng.randint(8, 14), H], fill=(210, 214, 218, 235))
    sd.rectangle([0, 0, W, H], fill=(255, 255, 255, 14))
    img = Image.alpha_composite(img.convert("RGBA"), shield).convert("RGB")

    # mild camera softness + noise
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    px = img.load()
    for _ in range(W * H // 60):
        x, y = rng.randrange(W), rng.randrange(H)
        r, g, b = px[x, y]
        k = rng.randint(-8, 8)
        px[x, y] = (max(0, min(255, r + k)), max(0, min(255, g + k)), max(0, min(255, b + k)))
    return img


# --------------------------------------------------------------------------- annotations
def describe_plan(vials: list[Vial]) -> str:
    parts = []
    for v in vials:
        what = "empty" if v.planned == "empty" else ("colored cubes" if v.planned == "cubes" else f"{v.planned} liquid")
        parts.append(f"position {v.position} ({v.vial_id}): {what}")
    return "; ".join(parts)


CONTEXT = (
    "You are a quality inspector watching a vial filling line from the side. Each wheeled carrier on the "
    "track holds one clear vial with a printed label. Vials are numbered by position from left to right. "
    "A correctly filled liquid vial is filled to roughly one third of its height."
)


def deviations(vials):
    return [v.position for v in vials if v.status != STATUS_OK]


def report_json(vials: list[Vial]) -> dict:
    return {
        "vials": [
            {"position": v.position, "vial_id": v.vial_id, "planned": v.planned, "actual": v.actual,
             "fill_pct": v.fill_pct, "status": v.status}
            for v in vials
        ],
        "deviating_positions": deviations(vials),
        "pass": not deviations(vials),
    }


def make_samples(image_rel: str, vials: list[Vial], rng: random.Random) -> list[dict]:
    plan = describe_plan(vials)
    samples = []

    def add(q, a, cat, with_plan=True):
        prompt = f"<image>\n{CONTEXT}\n" + (f"Manufacturing plan: {plan}\n" if with_plan else "") + q
        samples.append({
            "id": hashlib.md5(f"{image_rel}|{q}".encode()).hexdigest(),
            "images": [image_rel],
            "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": a}],
            "category": cat,
            "normalized_answer": a,
        })

    v = rng.choice(vials)
    add(f"What is in the vial at position {v.position}? Answer with one word: a liquid color "
        f"({', '.join(LIQUIDS)}), cubes, or empty.", v.actual, "content", with_plan=False)

    add(f"How many vials are visible on the track? Answer with an integer.", str(len(vials)), "count", with_plan=False)

    v = rng.choice(vials)
    add(f"Does the vial at position {v.position} match the manufacturing plan (correct content and a normal "
        f"fill level)? Answer yes or no.", "no" if v.status != STATUS_OK else "yes", "match")

    dev = deviations(vials)
    add("List the positions whose vial does NOT match the manufacturing plan, as comma-separated integers in "
        "ascending order. Answer 'none' if all match.", ",".join(map(str, dev)) if dev else "none", "deviation_list")

    add("Produce the inspection report as compact JSON with keys: vials (list of {position, vial_id, planned, "
        "actual, fill_pct, status}), deviating_positions, pass. status is one of OK, wrong_color, underfill, "
        "overfill, empty, wrong_content. Output JSON only.",
        json.dumps(report_json(vials), separators=(",", ":")), "report")
    return samples


# --------------------------------------------------------------------------- main
def build_split(name, count, out_root: Path, rng, n_choices, defect_rate):
    split = out_root / name
    img_dir = split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    annotations, gt = [], {}
    for i in range(count):
        vials = sample_run(rng, rng.choice(n_choices), defect_rate)
        img = render(vials, rng)
        fname = f"vials_{name}_{i:05d}.png"
        img.save(img_dir / fname)
        rel = f"images/{fname}"
        annotations.extend(make_samples(rel, vials, rng))
        gt[rel] = {"plan_text": describe_plan(vials), **report_json(vials)}
    (split / "annotations.json").write_text(json.dumps(annotations, indent=1))
    (split / "ground_truth.json").write_text(json.dumps(gt, indent=1))
    with tarfile.open(split / "images.tar.gz", "w:gz") as tar:
        tar.add(img_dir, arcname="images")
    print(f"[{name}] {count} images -> {len(annotations)} samples at {split}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/tube_inspection")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--defect-rate", type=float, default=0.25, help="per-vial probability of a defect")
    ap.add_argument("--vials", default="5,6,7,8", help="comma list of possible visible-vial counts")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n_choices = [int(x) for x in a.vials.split(",")]
    build_split("train", a.train, Path(a.out), rng, n_choices, a.defect_rate)
    build_split("val", a.val, Path(a.out), rng, n_choices, a.defect_rate)


if __name__ == "__main__":
    main()
