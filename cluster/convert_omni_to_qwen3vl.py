"""
Convert the nvidia/Cosmos3-Nano checkpoint (model_type=cosmos3_omni, Cosmos3ForConditionalGeneration) into a
plain Qwen3-VL Hugging Face checkpoint (model_type=qwen3_vl, Qwen3VLForConditionalGeneration) that the TAO
cosmos-rl container can load and LoRA fine-tune.

What Cosmos3-Nano actually contains
-----------------------------------
  transformer/diffusion_pytorch_model-*.safetensors  reasoner (Qwen3-VL-8B text stack) + generator ("*_moe_gen",
                                                     add_*/to_add_out attention branches, action/audio/time heads)
  vision_encoder/model.safetensors                   Qwen3VLVisionModel (blocks, merger, deepstack mergers, ...)
  tokenizer / preprocessor configs at repo root      already Qwen3-VL (Qwen2Tokenizer, Qwen3VLProcessor)

Mapping (verified against Qwen/Qwen3-VL-8B-Instruct's weight index; 750 tensors, exact key-set match):
  layers.N.self_attn.to_q|to_k|to_v|to_out   -> model.language_model.layers.N.self_attn.q_proj|k_proj|v_proj|o_proj
  layers.N.self_attn.norm_q|norm_k           -> model.language_model.layers.N.self_attn.q_norm|k_norm
  layers.N.{input_layernorm,post_attention_layernorm,mlp.*} -> model.language_model.layers.N.<same>
  embed_tokens.weight / norm.weight          -> model.language_model.embed_tokens.weight / model.language_model.norm.weight
  lm_head.weight                             -> lm_head.weight
  vision_encoder/*                           -> model.visual.<same>
  dropped: *_moe_gen*, self_attn.add_*, norm_added_*, to_add_out, action_*, audio_*, proj_in/out, time_embedder.*

Config: text/vision configs of the donor Qwen/Qwen3-VL-8B-Instruct (identical dims; supplies the mrope
rope_scaling the Omni config omits). Tokenizer + processor files are copied from the Cosmos3-Nano root.

Usage (inside the TAO container, which has torch + safetensors):
  python convert_omni_to_qwen3vl.py --src /tao-workspace/models/Cosmos3-Nano \
      --donor /tao-workspace/models/Qwen3-VL-8B-Instruct-donor --dst /tao-workspace/models/Cosmos3-Nano-qwen3vl
  python convert_omni_to_qwen3vl.py --src ... --donor ... --dst ... --dry-run    # key mapping only, no torch
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

TEXT_PREFIX = "model.language_model."
VIS_PREFIX = "model.visual."

ATTN_RENAME = {
    "to_q": "q_proj", "to_k": "k_proj", "to_v": "v_proj", "to_out": "o_proj",
    "norm_q": "q_norm", "norm_k": "k_norm",
}
DROP_PATTERNS = re.compile(
    r"(_moe_gen(\.|$))|(self_attn\.(add_|norm_added_|to_add_out))|^(action_|audio_|proj_in|proj_out|time_embedder)"
)
LAYER_RE = re.compile(r"^layers\.(\d+)\.(.*)$")


def map_text_key(key: str) -> str | None:
    """Map a transformer/ shard key to Qwen3-VL naming, or None if it belongs to the generator."""
    if DROP_PATTERNS.search(key):
        return None
    if key == "lm_head.weight":
        return key
    if key in ("embed_tokens.weight", "norm.weight"):
        return TEXT_PREFIX + key
    m = LAYER_RE.match(key)
    if not m:
        raise KeyError(f"unexpected reasoner key: {key}")
    n, rest = m.groups()
    if rest.startswith("self_attn."):
        sub = rest[len("self_attn."):]
        name, _, tail = sub.partition(".")
        if name not in ATTN_RENAME:
            raise KeyError(f"unexpected attention key: {key}")
        rest = f"self_attn.{ATTN_RENAME[name]}.{tail}"
    return f"{TEXT_PREFIX}layers.{n}.{rest}"


def map_vision_key(key: str) -> str:
    return VIS_PREFIX + key


def plan(src: Path) -> dict[str, tuple[str, str]]:
    """Return {new_key: (shard_path, old_key)} for every tensor to keep."""
    idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    out: dict[str, tuple[str, str]] = {}
    for old, shard in idx.items():
        if shard.startswith("vision_encoder/"):
            new = map_vision_key(old)
        else:
            new = map_text_key(old)
            if new is None:
                continue
        if new in out:
            raise KeyError(f"duplicate target key {new} from {old} and {out[new][1]}")
        out[new] = (shard, old)
    return out


def verify_against_donor(mapping: dict, donor: Path) -> None:
    di = donor / "model.safetensors.index.json"
    if not di.exists():
        print("WARN: donor index missing, skipping key-set verification")
        return
    donor_keys = set(json.loads(di.read_text())["weight_map"])
    ours = set(mapping)
    missing, extra = donor_keys - ours, ours - donor_keys
    if missing or extra:
        raise SystemExit(f"key-set mismatch vs donor: missing={sorted(missing)[:10]} extra={sorted(extra)[:10]}")
    print(f"key set matches donor Qwen3-VL-8B index exactly ({len(ours)} tensors)")


def copy_side_files(src: Path, donor: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json",
              "preprocessor_config.json", "video_preprocessor_config.json"):
        p = src / f
        if p.exists():
            shutil.copy2(p, dst / f)
    for f in ("generation_config.json",):
        p = donor / f
        if p.exists():
            shutil.copy2(p, dst / f)
    cfg = json.loads((donor / "config.json").read_text())
    cfg["architectures"] = ["Qwen3VLForConditionalGeneration"]
    cfg["model_type"] = "qwen3_vl"
    cfg["dtype"] = cfg.get("dtype", "bfloat16")
    cfg["torch_dtype"] = "bfloat16"
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    (dst / "CONVERSION.json").write_text(json.dumps({
        "source": "nvidia/Cosmos3-Nano (cosmos3_omni)",
        "architecture_donor": "Qwen/Qwen3-VL-8B-Instruct",
        "converter": "cosmos3tao/cluster/convert_omni_to_qwen3vl.py",
        "note": "reasoner + vision weights only; generator (moe_gen/action/audio) weights dropped",
    }, indent=2))


def convert(src: Path, donor: Path, dst: Path, mapping: dict, shard_bytes: int) -> None:
    import torch  # noqa: F401  (container has it)
    from safetensors import safe_open
    from safetensors.torch import save_file

    by_shard: dict[str, list[tuple[str, str]]] = {}
    for new, (shard, old) in mapping.items():
        by_shard.setdefault(shard, []).append((old, new))

    weight_map, buf, buf_bytes, shard_id, total = {}, {}, 0, 0, 0
    out_files = []

    def flush():
        nonlocal buf, buf_bytes, shard_id
        if not buf:
            return
        shard_id += 1
        name = f"model-{shard_id:05d}.safetensors"
        save_file(buf, str(dst / name), metadata={"format": "pt"})
        for k in buf:
            weight_map[k] = name
        out_files.append(name)
        print(f"  wrote {name} ({buf_bytes / 2**30:.2f} GiB, {len(buf)} tensors)")
        buf, buf_bytes = {}, 0

    for shard, pairs in sorted(by_shard.items()):
        print(f"reading {shard} ({len(pairs)} tensors to keep)")
        with safe_open(str(src / shard), framework="pt", device="cpu") as f:
            for old, new in sorted(pairs):
                t = f.get_tensor(old).to(torch.bfloat16).contiguous()
                buf[new] = t
                n = t.numel() * t.element_size()
                buf_bytes += n; total += n
                if buf_bytes >= shard_bytes:
                    flush()
    flush()

    # rename to HF style model-0000X-of-0000N and write index
    n = len(out_files)
    final_map = {}
    for i, name in enumerate(out_files, 1):
        newname = f"model-{i:05d}-of-{n:05d}.safetensors"
        (dst / name).rename(dst / newname)
        for k, v in weight_map.items():
            if v == name:
                final_map[k] = newname
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(final_map.items()))}, indent=2))
    print(f"done: {len(final_map)} tensors, {total / 2**30:.2f} GiB in {n} shards -> {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Cosmos3-Nano snapshot dir")
    ap.add_argument("--donor", required=True, help="dir with Qwen3-VL-8B-Instruct config.json (+ index for verification)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--shard-gib", type=float, default=4.0)
    ap.add_argument("--dry-run", action="store_true", help="compute + verify mapping only")
    a = ap.parse_args()
    src, donor, dst = Path(a.src), Path(a.donor), Path(a.dst)

    mapping = plan(src)
    kept = len(mapping)
    total = len(json.loads((src / "model.safetensors.index.json").read_text())["weight_map"])
    print(f"mapping: keep {kept} of {total} source tensors (dropped {total - kept} generator-only tensors)")
    verify_against_donor(mapping, donor)
    if a.dry_run:
        for k in list(mapping)[:8]:
            print(f"  {mapping[k][1]:60s} -> {k}")
        return
    if (dst / "model.safetensors.index.json").exists():
        print(f"{dst} already converted; delete it to redo."); return
    copy_side_files(src, donor, dst)
    convert(src, donor, dst, mapping, int(a.shard_gib * 2**30))


if __name__ == "__main__":
    main()
