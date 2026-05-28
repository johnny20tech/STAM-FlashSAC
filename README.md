# STAM-FlashSAC

Code and lightweight experiment artifacts for **Short-Term Action-Observation Memory for FlashSAC in Dexterous Manipulation**.

STAM-FlashSAC adds a short-term observation/action memory module to a FlashSAC-style feed-forward actor-critic. This repository is organized for reproducing the IsaacLab Shadow Hand cube reorientation experiments and for regenerating the released plots from lightweight CSV results.

## Repository Layout

```text
STAM-FlashSAC/
├── stam_flashsac/               # package code
│   ├── flashsac/                # FlashSAC-style models, agent, buffer, runtime helpers
│   ├── stam.py                  # STAM memory gate
│   ├── training.py              # training entrypoint
│   ├── experiment_runner.py     # multi-run launcher
│   ├── plotting.py              # result plotting
│   └── results.py               # lightweight result export helpers
├── scripts/                     # CLI entrypoints
├── results/
│   ├── experiment/<env_name>/<variant>_nstepN/seedX/
│   ├── figures/                 # current plotted experiment only
│   └── summaries/
├── checkpoints/                 # local training checkpoints, gitignored
├── pyproject.toml
├── uv.lock
└── README.md
```

Historical results are stored by variant, n-step setting, and seed, for example:

```text
results/experiment/Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0/memory_k16_action_nstep3/seed0/training_log.csv
```

## Environment

This repository uses `uv` and the IsaacLab-compatible dependency set inherited from FlashSAC.

```bash
cd STAM-FlashSAC
uv sync --extra isaaclab
```

Quick import check:

```bash
uv run python -c "import torch; print(torch.__version__)"
uv run python -c "import isaaclab; print('isaaclab ok')"
```

## Run Experiments

Dry-run the full suite:

```bash
uv run python scripts/run_experiments.py --suite all --dry-run
```

Run the main comparison:

```bash
uv run python scripts/run_experiments.py --suite main --gpus 0,1 --processes-per-gpu 1
```

Useful options:

```text
--suite main | ablations | all
--variants baseline,memory_k16_action
--seeds 0,1,2,3,4
--gpus 0,1
--processes-per-gpu 1
--n-step 3
--stam-context-len 16
--result-mode append | overwrite
--result-set my_run
--wandb
--wandb-project STAM-FlashSAC
--wandb-mode online | offline
```

Suites:

- `main`: runs the primary comparison, `baseline` and `memory_k16_action`.
- `ablations`: runs memory/action ablations, `memory_k4_action`, `memory_k8_action`, `memory_k16_action`, `memory_k8_obs`, and `memory_k16_obs`.
- `all`: runs every configured variant from `main` and `ablations`.

`append` is the default. It keeps existing `results/experiment/<env_name>/<variant>_nstepN/seedX` data and writes repeated runs under a result-set subfolder. `overwrite` replaces the existing data for a previously run seed/result-set.
Results are grouped by environment first, so different tasks can coexist under `results/experiment/<env_name>/...`.

## Weights & Biases Logging

TensorBoard logging is always written under each run's `tensorboard/` directory. To also stream scalar metrics and the run config to Weights & Biases, first log in:

```bash
wandb login
```

For a single training run:

```bash
uv run python scripts/train.py \
  --wandb \
  --wandb-project STAM-FlashSAC \
  --output-root checkpoints/20260528_211324
```

For the experiment runner:

```bash
uv run python scripts/run_experiments.py \
  --suite main \
  --gpus 0,1 \
  --processes-per-gpu 1 \
  --wandb \
  --wandb-project STAM-FlashSAC
```

On machines without stable network access, log locally and sync later:

```bash
uv run python scripts/run_experiments.py \
  --suite main \
  --wandb \
  --wandb-mode offline

wandb sync /tmp2/b12902071/STAM-FlashSAC/checkpoints/<result_set>/wandb
```

Useful W&B options:

```text
--wandb | --no-wandb
--wandb-project STAM-FlashSAC
--wandb-entity <team-or-user>
--wandb-mode online | offline
```

W&B records scalar metrics and config values only. Controller checkpoints, replay buffers, and other large artifacts remain local unless they are uploaded separately.

## Record MP4 Rollouts

Trained IsaacLab policies can be recorded as MP4 videos from an existing checkpoint run directory. The recorder uses IsaacLab's `render_mode="rgb_array"` path with headless offscreen video rendering, so run it on a machine with IsaacLab/Isaac Sim and an NVIDIA GPU available.

For a quick short test:

```bash
CUDA_VISIBLE_DEVICES=3 uv run python scripts/record_mp4.py \
  --run-dir checkpoints/20260528_224247/Isaac-Lift-Cube-Franka-v0_memory_k16_action_seed0 \
  --checkpoint best \
  --out checkpoints/20260528_224247/Isaac-Lift-Cube-Franka-v0_memory_k16_action_seed0/test.mp4 \
  --device cuda:0 \
  --steps 80 \
  --frame-stride 2 \
  --width 1280 \
  --height 720 \
  --fps 30
```

For a full rollout:

```bash
CUDA_VISIBLE_DEVICES=3 uv run python scripts/record_mp4.py \
  --run-dir checkpoints/20260528_224247/Isaac-Lift-Cube-Franka-v0_memory_k16_action_seed0 \
  --checkpoint best \
  --out checkpoints/20260528_224247/Isaac-Lift-Cube-Franka-v0_memory_k16_action_seed0/rollout.mp4 \
  --device cuda:0 \
  --steps 250 \
  --frame-stride 2 \
  --width 1280 \
  --height 720 \
  --fps 30
```

Useful MP4 options:

```text
--checkpoint best | latest | /path/to/controller.pt
--seed 42
--camera-eye 1.7 1.2 0.9
--camera-lookat 0.5 0.0 0.15
--rendering-mode performance | balanced | quality
--action-scale 2.75
```

`CUDA_VISIBLE_DEVICES=3 --device cuda:0` means the process sees physical GPU 3 as local `cuda:0`. The default output is `<run-dir>/rollout.mp4`. The recorder uses deterministic policy actions by default; `--action-scale` only overrides the environment action scaling used during recording. For `Isaac-Lift-Cube-Franka-v0`, the training wrapper scale is `3.0`, so lowering it too much can make the policy fail to lift the cube.

## Plot Results

```bash
uv run python scripts/plot_results.py
```

If you have multiple environments under `results/experiment/`, you can select which one to plot:

```bash
uv run python scripts/plot_results.py --figure-dataset paper_main --figure-env Isaac-Repose-Cube-Shadow-OpenAI-FF-Direct-v0
```

Plots are written to:

```text
results/figures/
```

`results/figures/` is intentionally reserved for the current plotted experiment. Historical CSV results may contain multiple n-step settings, but the default figures use the current paper setting, `n_step=3`, and therefore omit `n-step=3` from legend labels.

Summaries are written to:

```text
results/summaries/
```

## Historical Results

The included historical artifacts are lightweight:

- `config.json`
- `training_log.csv`
- summary CSVs
- plot PNGs

Large controller checkpoints, replay buffers, tensorboard event files, and process logs are intentionally excluded from `results/`. Different `n_step` settings are treated as different variants in the result layout.

To rebuild the historical layout from the original local experiment folders:

```bash
uv run python scripts/migrate_historical_results.py \
  --nstep1-archive-dir /path/to/nstep1/archive \
  --paper-main-dir /path/to/paper/main
```

## License

This repository is released under the MIT License. The minimal IsaacLab wrapper vendored under `stam_flashsac/vendor/flash_rl` follows the FlashSAC license included in that folder.

## Acknowledgements

This project builds on [FlashSAC](https://github.com/Holiday-Robot/FlashSAC) and uses a minimal vendored IsaacLab environment wrapper derived from the FlashSAC codebase.

## References

FlashSAC:

```bibtex
@article{kim2026flashsac,
  title={FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control},
  author={Kim, Donghu and Lee, Youngdo and Park, Minho and Kim, Kinam and Nahendra, I Made Aswin and Seno, Takuma and Min, Sehee and Palenicek, Daniel and Vogt, Florian and Kragic, Danica and Peters, Jan and Choo, Jaegul and Lee, Hojoon},
  journal={arXiv preprint arXiv:2604.04539},
  year={2026}
}
```
