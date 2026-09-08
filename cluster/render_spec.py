"""Render the TAO cosmos-rl train spec with cluster paths and GPU count.

Called by cluster/ptyche_train.sbatch inside the allocation. Paths are the *container* paths
(the sbatch mounts $WORK at /tao-workspace; never mount over /workspace, which holds cosmos_rl).

  python cluster/render_spec.py --template specs/cosmos3_nano_lora_sft.yaml --out /path/spec.yaml \
      --model /tao-workspace/models/Cosmos3-Nano --train-root /tao-workspace/data/tube_inspection/train \
      --val-root /tao-workspace/data/tube_inspection/val --results /tao-workspace/results/<job> --gpus 4
"""

import argparse
import yaml


def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    if v is None:
        return '""'
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def to_toml(d: dict, prefix: str = "") -> str:
    """Minimal dict -> TOML (matches what `toml.dumps` produces for TAO specs). No external deps."""
    out, tables = [], []
    for k, v in d.items():
        if isinstance(v, dict):
            tables.append((k, v))
        else:
            out.append(f"{k} = {_toml_scalar(v)}")
    text = "\n".join(out) + ("\n" if out else "")
    for k, v in tables:
        name = f"{prefix}.{k}" if prefix else k
        text += f"\n[{name}]\n" + to_toml(v, name)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--experiment-name", default="cosmos3_nano_tube_lora")
    a = ap.parse_args()

    spec = yaml.safe_load(open(a.template))
    spec["policy"]["model_name_or_path"] = a.model
    spec["policy"]["parallelism"]["dp_shard_size"] = a.gpus
    spec["policy"]["parallelism"]["dp_replicate_size"] = 1
    if a.epochs:
        spec["train"]["epoch"] = a.epochs
    spec["train"]["output_dir"] = f"{a.results}/output"
    spec["results_dir"] = a.results
    spec["logging"]["experiment_name"] = a.experiment_name

    custom = spec.setdefault("custom", {})
    # TAO 7.0.1 public container schema (custom.dataset). Newer 7.2 RC images use
    # custom.train_dataset / custom.val_dataset; we emit both so either schema finds its keys.
    custom["dataset"] = {
        "annotation_path": f"{a.train_root}/annotations.json",
        "media_path": a.train_root,
        "system_prompt": "",
    }
    custom["train_dataset"] = {"annotation_path": f"{a.train_root}/annotations.json", "media_path": a.train_root}
    custom["val_dataset"] = {"annotation_path": f"{a.val_root}/annotations.json", "media_path": a.val_root}

    with open(a.out, "w") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)
    print(f"rendered spec -> {a.out}")

    # TAO FTMS runs `toml.dumps(spec)` and launches `cosmos-rl --config spec.toml <hook>`; emit the same.
    toml_path = a.out.rsplit(".", 1)[0] + ".toml"
    with open(toml_path, "w") as fh:
        fh.write(to_toml(spec))
    print(f"rendered toml -> {toml_path}")
    print(yaml.safe_dump({"policy": spec["policy"], "custom": custom, "results_dir": a.results}, sort_keys=False))


if __name__ == "__main__":
    main()
