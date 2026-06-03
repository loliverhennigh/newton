# Beam-Twist POD + Neural Latent ROM

This directory contains a narrow reduced-order modeling proof of concept for the
Newton implicit-MPM beam-twist example. It keeps the full Newton MPM solver as the
teacher, fits a POD decoder over particle positions, and trains a small neural
network to advance the POD latent coordinates on held-out controlled beam
rollouts.

The reduced model is geometry-specific. It assumes the same beam particle layout
and a small family of twist plus kinematic end-translation controls. It does not
change the core MPM solver and does not use a sampled MPM residual solve.

## End-to-End Run

From the repository root:

```bash
uv run python tools/mpm_beam_twist_pod_nn_rom.py run-smoke \
  --output-dir artifacts/beam_twist_pod_nn_smoke \
  --device cpu \
  --grid-type dense \
  --frames 24 \
  --rank 8 \
  --epochs 1000 \
  --voxel-size 0.75
```

The command writes:

- teacher rollouts under `teacher_rollouts/`
- POD basis and projected training coefficients under `pod_models/`
- linear and neural latent models under `latent_models/`
- autonomous held-out decoded rollouts under `reduced_rollouts/`
- GIF/MP4 visual comparison under `animations/`
- `beam_twist_pod_nn_report.md` with quality and runtime metrics

For a richer train/validation/extrapolation split, run:

```bash
uv run python tools/mpm_beam_twist_pod_nn_rom.py run-controlled-dataset \
  --output-dir artifacts/controlled_beam_dataset \
  --device cpu \
  --grid-type dense \
  --frames 96 \
  --rank 16 \
  --epochs 1500 \
  --voxel-size 0.75 \
  --hidden-dim 96
```

This command generates 12 training rollouts, 4 interpolation validation rollouts,
and 3 extrapolation rollouts. The right beam end is driven with combinations of
twist, lateral motion, vertical motion, and axial in/out motion. It trains both a
linear latent baseline and a neural latent model, then evaluates both on the
held-out cases and writes `controlled_dataset_report.md`.

## Individual Pipeline

Generate teacher rollouts:

```bash
uv run python tools/mpm_beam_twist_pod_nn_rom.py generate \
  --output-dir artifacts/beam_twist/teacher_rollouts \
  --run-name train_twist_0p9 \
  --frames 24 \
  --twist-speed-scale 0.9 \
  --voxel-size 0.75
```

Fit POD:

```bash
uv run python tools/mpm_beam_twist_pod_nn_rom.py fit-pod \
  --output-dir artifacts/beam_twist/pod_models \
  --run-name pod_rank8 \
  --rank 8 \
  artifacts/beam_twist/teacher_rollouts/train_twist_0p9 \
  artifacts/beam_twist/teacher_rollouts/train_twist_1p1
```

Train latent models:

```bash
uv run python tools/mpm_beam_twist_pod_nn_rom.py train-linear \
  --output-dir artifacts/beam_twist/latent_models \
  --pod-dir artifacts/beam_twist/pod_models/pod_rank8 \
  artifacts/beam_twist/teacher_rollouts/train_twist_0p9 \
  artifacts/beam_twist/teacher_rollouts/train_twist_1p1

uv run python tools/mpm_beam_twist_pod_nn_rom.py train-nn \
  --output-dir artifacts/beam_twist/latent_models \
  --pod-dir artifacts/beam_twist/pod_models/pod_rank8 \
  --epochs 1000 \
  artifacts/beam_twist/teacher_rollouts/train_twist_0p9 \
  artifacts/beam_twist/teacher_rollouts/train_twist_1p1
```

Roll out and render the held-out case:

```bash
uv run python tools/mpm_beam_twist_pod_nn_rom.py rollout \
  --output-dir artifacts/beam_twist/reduced_rollouts \
  --pod-dir artifacts/beam_twist/pod_models/pod_rank8 \
  --model-dir artifacts/beam_twist/latent_models/pod_nn_latent \
  artifacts/beam_twist/teacher_rollouts/val_twist_1p0

uv run python tools/mpm_beam_twist_pod_nn_rom.py render \
  --output-dir artifacts/beam_twist/animations \
  --prediction-dir artifacts/beam_twist/reduced_rollouts/heldout_pod_nn_rollout \
  artifacts/beam_twist/teacher_rollouts/val_twist_1p0
```
