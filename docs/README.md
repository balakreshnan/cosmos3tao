# Documentation

This folder explains what actually happens when you run the steps in the top-level [README](../README.md).
Read it in order the first time; afterwards jump to the chapter for the step you are on.

| # | Chapter | Answers |
| --- | --- | --- |
| 00 | [The whole journey, start to finish](00-end-to-end-flow.md) | **Start here if you are not technical**: the story from one photo to a working inspection app |
| 01 | [Overview and architecture](01-overview.md) | What are the moving parts, how do they fit, what runs where |
| 02 | [Cluster environment](02-cluster-environment.md) | SLURM, Pyxis/Enroot, Lustre, `.sqsh` images, GPU architecture constraints, env vars |
| 03 | [The dataset](03-dataset.md) | How synthetic images and LLaVA annotations are generated, the five question types, ground truth |
| 04 | [Setup script](04-setup-script.md) | `cluster/ptyche_setup.sh` step by step |
| 05 | [Model conversion](05-model-conversion.md) | Why Cosmos3-Nano must be converted to Qwen3-VL layout and how the converter does it |
| 06 | [Training spec](06-training-spec.md) | Every field in `specs/cosmos3_nano_lora_sft.yaml` and what it controls |
| 07 | [Training job](07-training-job.md) | `ptyche_train.sbatch`, `train_in_container.sh`, `render_spec.py`, how cosmos-rl is launched |
| 08 | [Monitoring and results](08-monitoring-and-results.md) | Reading the log, SLURM commands, what lands in `results/` |
| 09 | [Evaluation](09-evaluation.md) | `eval_in_container.py`: adapter merge, inference, scoring, the metrics, the numbers we got |
| 10 | [Report](10-report.md) | `report/make_report.py`: parsing, analysis, the HTML page |
| 11 | [Troubleshooting](11-troubleshooting.md) | Every failure hit during the first run, its cause and fix |
| 12 | [Inference and next steps](12-inference-and-next-steps.md) | Using the adapter, real plant photos, second training round |
| — | [Local inferencing guide](../inferencing/README.md) | Running the web app on a laptop GPU, with screenshots |
| — | [Glossary](glossary.md) | TAO, cosmos-rl, LoRA, FSDP, Pyxis, sqsh, Omni, Qwen3-VL … |

Conventions used throughout:

- `$LUSTRE_DIR/cosmos3tao` is called **the work directory**. Inside the container it is mounted at `/tao-workspace`.
- The git checkout on Lustre is **the repo**; inside the container it is mounted read-only at `/tao-repo`.
- "Login node" means the SSH host (`login-lyris.nvidia.com`); "compute node" means a GPU node SLURM allocates.
- Shell snippets are meant for the login node unless marked PowerShell (laptop).
