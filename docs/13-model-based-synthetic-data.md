# 13 · Model-based synthetic data: closing the realism gap

Chapter 03 explains that the current dataset is drawn by ordinary Python code with no model involved. That gave
perfect labels for free but stylised pictures: 96.7 % accuracy on the drawings, 7 of 8 vials on the one real
photo. This chapter surveys NVIDIA's model-based generators that can produce **photoreal** training images, and
lays out a concrete plan for the vial line.

## The options

| Tool | What it does | How it would apply to the vial line | Labels |
| --- | --- | --- | --- |
| **PAIDF AnomalyGen** — container `nvcr.io/nvidia/paidf-anomalygen`, model `nvidia/Cosmos-AnomalyGen-Metal-2B` | Diffusion pipeline on Cosmos-Predict2 Text2Image. Few-shot fine-tunes small adapters (anomaly tokens + a 2-layer MLP) on a handful of real defect examples, then inpaints new photoreal defects into good images at mask locations. Exports DAFT for TAO | Paint wrong-colour, over-fill, under-fill, empty and wrong-content states into real photos of correctly filled vials | **automatic**: the inpainting mask says which vial changed and how |
| **PAIDF Augmentation** — `nvcr.io/nvidia/paidf-augmentation` | Photoreal variation of images/video (lighting, weather, scene). Picks among Cosmos Transfer 2.5, Cosmos Predict 2.5 and an image-editing model | Turn 30 real frames into thousands across lighting, glare and shift conditions | **kept**: contents unchanged, existing labels still valid |
| **Cosmos Transfer 2.5** | Control-to-image/video: segmentation, depth or edge maps in, photoreal frames out | Use our Pillow renders (or a mask render of them) as control input; geometry and labels stay, appearance becomes photoreal | **kept**: same positions and contents as the render |
| **Cosmos Predict 2.5** — `tao-toolkit:7.0.1-cosmos-predict` | Text/image/video-conditioned generation | New scenes from a prompt or seeded by the plant photo | **manual** or via a VLM labeller |
| **Cosmos3 generator** (Cosmos3-Nano/Super Text2Image, Image2Video) | Generation half of the Cosmos 3 omni model | Same as Predict within the Cosmos 3 family | manual |
| **Omniverse Replicator / Isaac Sim** | 3D simulation with automatic ground truth; can pass frames through Cosmos for realism | Model the track, carriers and vials once; render any state from any camera | **automatic** |
| **TAO skill bank `tao-run-deft-aoi-cosmos3`** | End-to-end recipe: mine images → AnomalyGen → assemble training JSON → fine-tune Cosmos Reason 3 → evaluate | NVIDIA's own automation of what this repo does by hand | automatic |

## Recommended plan for the vial line

### Phase A · AnomalyGen on real frames (highest payoff)

1. **Collect** 20–50 real photos of the line with all vials correctly filled, same camera as `plant.jpg`, across a
   shift so lighting varies. Record the plan for each.
2. **Mask** each vial once (a rectangle per carrier position is enough; the base model's grounding from
   `apps/model_runtime.py` can propose them). Masks give AnomalyGen the paint region and give us the position
   label.
3. **Collect defect exemplars**: 5–10 real photos per defect type (wrong colour, over-fill, under-fill, empty,
   cubes where liquid belongs). If some never occur in production, stage them once on the line.
4. **Fine-tune AnomalyGen** on the exemplars (few-shot; minutes on one GPU), then **generate** several hundred
   images per defect type by inpainting into the good frames at random vial masks.
5. **Write annotations** with the same five templates as chapter 03: for each generated image we know the plan,
   which vial changed, to what, so `actual`, `status`, `deviating_positions` and `pass` follow. Fill % for
   over/under-fill comes from the mask height ratio.
6. **Mix** with the existing renders (e.g. 30 % real-derived, 70 % synthetic) and repeat README steps 6–10.

Expected effect: the real-photo result moves from "7 of 8, misses occluded cube stack" toward the synthetic
96.7 %, because the model now sees real glass, reflections, labels and the shield frame during training.

### Phase B · Cosmos Transfer over the existing renders (cheapest)

Our generator already produces perfect geometry and labels. Feed each render, or a flat-colour segmentation
version of it (one colour per vial, one per liquid, one per track band), to Cosmos Transfer 2.5 with a prompt
such as "photograph of a stainless-steel vial filling line, glass test tubes in wheeled carriers, industrial
lighting". Keep the annotations unchanged. Cost: one Transfer inference per image (seconds on a GB200), zero
labelling.

### Phase C · Augmentation for robustness

Run PAIDF Augmentation over the real frames from Phase A to vary lighting and reflections before inpainting.
This is what protects accuracy across a day of plant operation.

## Practical notes for this cluster

- The PAIDF containers are separate images from the TAO cosmos-rl one; import them to `.sqsh` the same way
  (`enroot import -o … docker://nvcr.io#nvidia/paidf-anomalygen:1.0.1`) and check they run on GB200. Like the
  TAO image, they will not run on GB300 nodes if their CUDA predates sm_103.
- Generated images must be reviewed by someone who knows the line. A diffusion model can paint plausible but
  physically impossible states (liquid floating above air, cubes melting into liquid). Reject those before
  training on them.
- Keep the real photos out of the generated set for evaluation: hold back some real frames untouched so the
  final accuracy number is measured on genuine camera images.
- DAFT is TAO's dataset format for these pipelines; our LLaVA `annotations.json` can be produced from DAFT masks
  with a short conversion script following the templates in chapter 03.

## Sources

- [PAIDF AnomalyGen on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paidf-anomalygen)
- [PAIDF Augmentation on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/paidf-augmentation)
- [nvidia/Cosmos-AnomalyGen-Metal-2B](https://huggingface.co/nvidia/Cosmos-AnomalyGen-Metal-2B)
- [NVIDIA Physical AI Data Factory](https://github.com/NVIDIA/physical-ai-data-factory)
- [tao-run-deft-aoi skill](https://www.skills.sh/nvidia/skills/tao-run-deft-aoi)
- [Cosmos synthetic data generation in Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_cosmos.html)
- [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/index.html)
