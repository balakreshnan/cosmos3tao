#!/usr/bin/env bash
# Runs INSIDE the TAO cosmos-rl container. Dumps how the cosmos-rl launcher is invoked, which TAO
# dataset hooks / converters are bundled, and how a spec is consumed. Read-only, no GPU work.
#
#   srun -A $ACCOUNT -p $PARTITION -N1 -n1 --time=00:10:00 --job-name=$ACCOUNT-cosmos3.probe \
#        --container-image=$LUSTRE_DIR/cosmos3tao/sqsh/nvcr.io_nvidia_tao_tao-toolkit_7.0.1-cosmos-rl-aarch64.sqsh \
#        --container-mounts=$LUSTRE_DIR/cosmos3tao:/tao-workspace,$LUSTRE_DIR/cosmos3tao/repo:/tao-repo:ro \
#        --no-container-remap-root bash /tao-repo/cluster/probe_container.sh 2>&1 | tee probe.txt
set +e
sec() { echo; echo "################ $*"; }

sec "python / launcher locations"
which python; python --version
ls -la /opt/venv/cosmos_rl/bin | grep -v -E "^total|activate|pip|python|wheel" | head -60

sec "cosmos-rl --help"
cosmos-rl --help 2>&1 | head -80

sec "cosmos-rl subcommands (if any)"
for sub in train sft evaluate inference; do echo "--- cosmos-rl $sub --help"; cosmos-rl $sub --help 2>&1 | head -15; done

sec "installed packages (tao / cosmos / daft / peft / transformers / vllm)"
pip list 2>/dev/null | grep -i -E "tao|cosmos|daft|peft|transformers|vllm|torch "

sec "site-packages layout"
SP=$(python -c "import cosmos_rl,os;print(os.path.dirname(cosmos_rl.__file__))")
echo "cosmos_rl at: $SP"
ls "$SP"
echo "--- tools:"; ls "$SP/tools" 2>/dev/null
echo "--- custom_hooks:"; ls "$SP/tools/custom_hooks" 2>/dev/null
echo "--- model_preparation:"; ls "$SP/model_preparation" 2>/dev/null || echo "(none)"

sec "TAO integration entrypoints (anything mentioning yaml spec / results_dir / status logger)"
grep -rl -E "results_dir|TAOStatusLogger|tao_status" "$SP" 2>/dev/null | head -20
echo "--- python modules with __main__ under cosmos_rl/tools:"
find "$SP/tools" -name "*.py" 2>/dev/null | head -40
echo "--- other TAO packages:"
python - <<'EOF'
import importlib, pkgutil
for m in ["nvidia_tao_core", "nvidia_tao_pytorch", "tao_core", "nvidia_tao_daft", "daft"]:
    try:
        mod = importlib.import_module(m); print(m, "->", getattr(mod, "__file__", "?"))
    except Exception as e:
        print(m, "-> not importable:", type(e).__name__)
EOF

sec "/workspace and any docs / examples"
ls -la /workspace 2>/dev/null | head; ls /workspace/cosmos_rl 2>/dev/null | head -30
find / -maxdepth 4 \( -iname "*tao*sft*" -o -iname "*example*.toml" -o -iname "*sft*.toml" \) -not -path "*/proc/*" 2>/dev/null | head -20

sec "how a hook consumes the yaml spec (first hook, head)"
H=$(ls "$SP"/tools/custom_hooks/tao*sft*.py 2>/dev/null | head -1)
[ -n "$H" ] && { echo "$H"; sed -n '1,120p' "$H"; } || echo "(no tao sft hook found)"

sec "/opt/cosmos_rl TAO hooks (what FTMS launches: cosmos-rl --config spec.toml /opt/cosmos_rl/tao_sft_example.py)"
ls -la /opt/cosmos_rl 2>&1
for f in /opt/cosmos_rl/tao_sft_example.py /opt/cosmos_rl/custom_sft.py; do
  [ -f "$f" ] && { echo "--- $f (custom.* keys it reads):"; grep -n -E "custom|annotation_path|media_path|system_prompt|total_pixels|fps|dataset" "$f" | head -60; }
done

sec "omni -> qwen3_vl converter present?"
python -c "import cosmos_rl.model_preparation.vlm_safetensors as m; print('converter:', m.__file__)" 2>&1 | tail -1
ls /opt/tao 2>/dev/null; cat /opt/tao/image-provenance.json 2>/dev/null | head -20

sec "checkpoint config"
python -c "import json;c=json.load(open('/tao-workspace/models/Cosmos3-Nano/config.json'));print({k:c[k] for k in c if k in ('model_type','architectures','text_config','vision_config')} if 'model_type' in c else c)" 2>&1 | head -20
echo "done."
