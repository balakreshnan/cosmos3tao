# 05 · Model conversion: Cosmos3-Nano (Omni) → Qwen3-VL

README step 5: `cluster/ptyche_prepare_model.sh`, which runs
[`cluster/convert_omni_to_qwen3vl.py`](../cluster/convert_omni_to_qwen3vl.py) inside the container.

## The problem

The first training attempt failed inside the container with

```
ValueError: The checkpoint you are trying to load has model type `cosmos3_omni` but Transformers does not
recognize this architecture.
```

`nvidia/Cosmos3-Nano` on Hugging Face is a single **Omni** checkpoint, architecture
`Cosmos3ForConditionalGeneration`, that bundles:

- the **reasoner**: a Qwen3-VL-8B language stack (36 layers, hidden 4096, 8 KV heads) and a Qwen3-VL vision
  tower (27 blocks, deepstack mergers)
- the **video generator**: extra "moe_gen" branches in every layer (`mlp_moe_gen`, `*_layernorm_moe_gen`,
  `self_attn.add_q/k/v_proj`, `to_add_out`, `norm_added_*`), `norm_moe_gen`, `time_embedder`, `proj_in/out`
- **action** and **audio** heads (`action_proj_*`, `audio_proj_*`, modality embeddings)

Transformers 4.57 in the TAO 7.0.1 image has no `cosmos3_omni` class. It does have `Qwen3VLForConditionalGeneration`,
and the reasoner is exactly that model. NVIDIA's own TAO skill for Cosmos3-Nano does the same conversion using
`Qwen/Qwen3-VL-8B-Instruct` as the "architecture donor", so this is the sanctioned path, not a hack.

## What the converter does

1. **Plan the key mapping** from `model.safetensors.index.json` without loading any tensors:

   | Omni key (transformer/ shards) | Qwen3-VL key |
   | --- | --- |
   | `layers.N.self_attn.to_q/to_k/to_v/to_out.weight` | `model.language_model.layers.N.self_attn.q_proj/k_proj/v_proj/o_proj.weight` |
   | `layers.N.self_attn.norm_q/norm_k.weight` | `…self_attn.q_norm/k_norm.weight` |
   | `layers.N.input_layernorm / post_attention_layernorm / mlp.*` | `model.language_model.layers.N.<same>` |
   | `embed_tokens.weight`, `norm.weight` | `model.language_model.embed_tokens.weight`, `model.language_model.norm.weight` |
   | `lm_head.weight` | `lm_head.weight` |
   | everything in `vision_encoder/model.safetensors` | `model.visual.<same key>` |
   | `*_moe_gen*`, `add_*`, `norm_added_*`, `to_add_out`, `action_*`, `audio_*`, `proj_in/out`, `time_embedder.*` | **dropped** |

   Result: keep 750 of 1165 tensors, drop 415 generator-only tensors.

2. **Verify** the resulting key set equals the donor's `model.safetensors.index.json` exactly (750 = 750). The
   script aborts on any mismatch, so a Transformers version with different naming would be caught here.

3. **Copy side files.** Tokenizer (`tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`,
   `chat_template.json`) and processor configs (`preprocessor_config.json`, `video_preprocessor_config.json`)
   come from the Cosmos repo root: they are already Qwen3-VL ones (`Qwen2Tokenizer`, `Qwen3VLProcessor`).
   `config.json` comes from the donor with `architectures=["Qwen3VLForConditionalGeneration"]`,
   `model_type="qwen3_vl"`, `torch_dtype=bfloat16`. The donor config matters for one thing the Omni config
   omits: `rope_scaling` with `mrope_interleaved` and `mrope_section=[24,20,20]`, which Qwen3-VL needs for
   multimodal positions. All dimension fields are identical between the two configs.

4. **Stream the weights.** Shard by shard with `safetensors.safe_open`, cast to bfloat16, rename, write new
   4 GiB shards `model-0000X-of-0000N.safetensors` and a new index. Peak RAM ≈ one output shard.

5. Write `CONVERSION.json` recording provenance.

Output: `models/Cosmos3-Nano-qwen3vl`, ~17 GB, loadable with `AutoConfig`/`Qwen3VLForConditionalGeneration`.

## The wrapper script

`ptyche_prepare_model.sh`:

1. downloads three small donor files with the login-node venv:
   `hf download Qwen/Qwen3-VL-8B-Instruct config.json generation_config.json model.safetensors.index.json`
   (filenames are positional arguments to `hf download`; an earlier version used `--include` wrongly)
2. probes the compute arch and picks the matching `.sqsh`
3. runs the converter under `srun` inside the container with `/opt/venv/cosmos_rl/bin/python` (which has torch
   and safetensors; the system python does not), then checks
   `AutoConfig.from_pretrained(...)` prints `qwen3_vl ['Qwen3VLForConditionalGeneration']`

Runtime ≈ 5–10 minutes, CPU only. Skipped if the output index already exists; delete the folder to redo.

## Is anything lost?

For the reasoning/vision task: no. The dropped tensors belong to the video generator and action/audio heads,
which are never used for image-plus-text question answering. The reasoner weights are copied bit-exact (already
bfloat16 in the source).

## Dry-run on a laptop

The mapping can be validated without the 33 GB download:

```powershell
python cluster\convert_omni_to_qwen3vl.py --src <dir with Cosmos index json> --donor <dir with donor index json> --dst x --dry-run
```

prints `keep 750 of 1165 … key set matches donor Qwen3-VL-8B index exactly (750 tensors)`.
