# 03 · The dataset

Source: [`dataset/generate_tube_dataset.py`](../dataset/generate_tube_dataset.py). Produced by README step 4
(inside `ptyche_setup.sh`) and regenerated locally for the report's failure gallery.

## From one photo to thousands of examples (the non-technical version)

We had **one** real photograph of the filling line. A model cannot learn a task from one example, and we could
not go back to the plant and photograph thousands of deliberately mis-filled vials. So we did what a film studio
does: we built a set.

1. **We studied the photo** and wrote down what a scene contains: a metal track, wheeled carriers, tall clear
   vials with printed labels, liquid that normally reaches about a third of the height, cube stacks, empties, a
   glass shield in front, the camera looking from the side.
2. **We wrote a drawing program** that paints such a scene from scratch. Every time it runs it rolls dice: how many
   vials (5 to 8), what each one should contain according to the plan, and whether something went wrong with it
   (wrong colour, too little, too much, empty, wrong kind of content). Then it paints exactly that.
3. **Because the program decides the contents, it also knows the correct answer** for every picture: the plan,
   what is really in each vial, how full it is, and which positions deviate. No human labelling was needed.
4. **We ran it 1,000 times** and asked **5 questions** about every picture, giving 5,000 question/answer pairs.

| From | To | How |
| --- | --- | --- |
| 1 real photo | a scene description | looked at it |
| a scene description | a drawing program | ~400 lines of Python using an image library |
| the drawing program | **800 training pictures** | run with seed 42, first 800 draws |
| the drawing program | **200 validation pictures** | next 200 draws; used to check learning during training and for the accuracy test |
| each picture | **5 question/answer pairs** | one per question type (what is in vial N, how many vials, does vial N match, which deviate, full JSON report) |
| 800 pictures | **4,000 training pairs** | 800 × 5 |
| 200 pictures | **1,000 validation pairs** | 200 × 5 |
| the 1,000 validation pairs | **300 graded test questions** | first 60 of each question type, used for the accuracy table |
| 1 real photo | **final exam** | never used in training; tried only at the very end in the web app |

Important honesty note: the 1,000 pictures are **not** edited copies of the real photo. They are drawings in the
same style. That is why the model's score on the drawings (96.7 %) is a measure of how well it learned the task,
while its result on the real photo (7 of 8 vials right) is a first, separate indication of how it copes with a
real camera. Chapter 12 explains how to add real photos to the training set to close that gap.

Because the dice are seeded (`--seed 42`), anyone who runs the program gets the identical 1,000 pictures, which is
how the laptop could recreate the exact validation images the cluster tested on.

**Was any AI model used to create the dataset? No.** The pictures are drawn by ordinary Python code with the
Pillow image library: rectangles for the track, rounded rectangles for vials and liquid, squares for cubes,
circles for wheels, text for labels, plus a little blur and noise. No image-generation model (Cosmos Predict,
diffusion, GAN), no vision model and no language model was involved, and no human labelled anything. The
answers are exact because the same code that chooses what to draw also writes the label. Models enter the
project only later: Cosmos3-Nano is the model being trained, and its base version draws the bounding boxes in
the web app.

## What the images show

A side-view camera on a MagneMotion-style linear track, modeled on the one real plant photo in
`data/real/plant.jpg`:

- stainless track with a dark motor band and bolted rail housing, "MM LITE" lettering, background cabinets and a
  tray of yellow caps
- 5–8 **wheeled carriers**, each with an aluminum bracket and a black clamp ring holding one tall clear vial
- every vial has a printed label: lot `20200232` and an ID such as `VIAL 0019`, plus a small QR block
- vial contents: **colored liquid** (orange, blue, yellow, red, green, purple, or clear) filling roughly the
  bottom third, **a stack of red/blue/yellow cubes**, or **nothing**
- a safety shield in front: faint vertical frame bars and a slight tint, plus camera blur and sensor noise

Everything is drawn with Pillow; no external assets. Image size 1400 × 700.

## What the labels encode

Each vial position gets a **planned** content (what the manufacturing plan expects) and an **actual** content
(what was drawn). With probability `--defect-rate` (default 0.25) a vial is given a defect:

| status | meaning | how it is drawn |
| --- | --- | --- |
| `OK` | matches plan; liquid fill 28–42 % | normal |
| `wrong_color` | liquid of a different color | different color |
| `underfill` | fill 4–16 % | short column |
| `overfill` | fill 58–80 % | tall column |
| `empty` | nothing where liquid or cubes were planned | empty glass |
| `wrong_content` | cubes where liquid was planned, or liquid where cubes/empty were planned | swapped content |

Some combinations are meaningless (an "overfill" of cubes) and are mapped back to `OK` in `sample_run`.

The **manufacturing plan is not drawn in the image** because the real plant photo contains no plan; it is
supplied as text in the prompt, e.g.
`position 1 (VIAL 0007): colored cubes; position 2 (VIAL 0008): colored cubes; position 3 (VIAL 0009): empty; …`

## The five questions per image

`make_samples` turns each image into five LLaVA-format training samples that share the picture:

| category | question (abridged) | answer | plan in prompt? |
| --- | --- | --- | --- |
| `content` | What is in the vial at position N? one word | `blue` / `cubes` / `empty` | no |
| `count` | How many vials are visible? | `5` | no |
| `match` | Does vial N match the plan? | `yes` / `no` | yes |
| `deviation_list` | List positions that do NOT match the plan, ascending, or `none` | `4` or `1,3,5` | yes |
| `report` | Produce the inspection report as compact JSON | full JSON | yes |

The JSON report has keys `vials` (list of `{position, vial_id, planned, actual, fill_pct, status}`),
`deviating_positions`, `pass`. Mixing short factual questions with the full report gives the model an easy
curriculum (perception first) and a strict target (the structured report the line controller would consume).

## File format (TAO `vlm` / `llava`)

`annotations.json` is a JSON array; each element:

```json
{
  "id": "0151686c12b50e0ba1ca3d8662f0023d",
  "images": ["images/vials_train_00000.png"],
  "conversations": [
    {"from": "human", "value": "<image>\n<role text>\nManufacturing plan: ...\n<question>"},
    {"from": "gpt",   "value": "<answer>"}
  ],
  "category": "report",
  "normalized_answer": "<answer>"
}
```

- `<image>` is the placeholder TAO's data loader replaces with the picture.
- `images` paths are relative to the split folder, which becomes `custom.dataset.media_path` in the spec.
- `id` is an MD5 of image + question so it is stable across regenerations.
- `category` / `normalized_answer` are the TAO/cosmos-rl evaluation fields; our own evaluator also uses them.

Each split folder also contains `images.tar.gz` (the layout TAO's FTMS expects when it ingests a dataset) and
`ground_truth.json`, a per-image record with `plan_text`, the vial list, `deviating_positions` and `pass`. The
ground truth is not used in training; the evaluator and the report use it to diagnose failures.

## Determinism

`--seed 42` seeds one `random.Random` used for everything, and samples are drawn sequentially, so the first N
images are identical no matter how many you generate. That is how the laptop reproduces the exact cluster
validation images for the failure gallery (`regenerated val matches cluster gold: True` in our check).

## Sizes

800 train images → 4000 samples, 200 val → 1000 samples, about 1 minute to generate. The cluster setup uses
these defaults; change with `N_TRAIN` / `N_VAL` env vars before running `ptyche_setup.sh`, or delete
`data/tube_inspection` and re-run to regenerate.

## Known limitations that showed up in evaluation

- **Clear liquid** is drawn in a light gray that is hard to distinguish from the glass; half of the fine-tuned
  model's misses involve clear liquid.
- **Fill level** has no printed scale on the vial; over-fill was the most under-detected defect. Adding tick
  marks or more overfill examples is the first thing to try for a second round.
- All images share one camera geometry; real photos vary. Mixing in labeled real frames is needed before
  trusting numbers on plant images.
