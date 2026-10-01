# Clean-PhysicsBridge denoiser experiment

This project is the next low-SNR experiment for the FIM channel-estimation pipeline.
It keeps the existing PhysicsBridge, low-SNR-biased sampling, and FiLM noise conditioning,
but changes the **training supervision** so that the neural residual learns the noise-induced
PhysicsBridge error rather than simultaneously learning denoising + structural bridge bias.

## Core idea

The current hybrid predicts

`H_hat = H_phys,noisy + DeltaH_NN`.

For this experiment, the supervision target is the noiseless PhysicsBridge output generated
from the **same underlying scene**:

`H_phys,clean = PhysicsBridge(clean observations, zero noise variance)`.

The network therefore learns, implicitly,

`DeltaH_noise = H_phys,clean - H_phys,noisy`.

The final model is still evaluated against the true undeformed channel `H0`. The best
checkpoint is selected using **true-channel validation NMSE**, not the clean-physics target.
This prevents the denoising objective from hiding regressions in the actual end metric.

## What is retained from the previous experiment

- PhysicsBridge unchanged.
- Residual hybrid form unchanged.
- FiLM noise conditioning unchanged.
- Low-SNR-biased training distribution unchanged:
  - SNRs: `[-10,-5,0,5,10,15,20]` dB
  - probabilities: `[0.25,0.25,0.20,0.15,0.07,0.04,0.04]`
- 70% of training scenes are therefore at `-10/-5/0 dB`.
- Evaluation still measures true undeformed-channel NMSE.

## Main changed files

- `simulator.py`: adds `return_clean=True` without exposing all latent scene parameters.
- `train.py`: adds `--training-target clean_physics` and clean-physics supervision.
- `verify.py`: validates the clean-physics target and finite gradients.
- `kaggle_clean_physics_denoiser.py`: dedicated Kaggle launcher.
- `Kaggle_Clean_Physics_Denoiser.ipynb`: one-cell Kaggle notebook.
- `compare_denoising_runs.py`: paired old-FiLM vs new-denoiser comparison on identical scenes/noise.

## Recommended Kaggle run

Edit in `kaggle_clean_physics_denoiser.py`:

```python
REPO_URL='https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git'
LARGE_ALPHA=1.0
QUICK=True
```

Run once with `QUICK=True`. If it succeeds, set `QUICK=False` for the real experiment.

To automatically compare against your previous low-SNR+FiLM checkpoint, optionally set:

```python
BASELINE_RESULTS_DIR='/kaggle/input/.../previous_results_folder'
```

The baseline folder must contain `configuration.json`, `geometry.pt`, `bridge_large.pt`, and
`seed_11/large_hybrid/best.pt`.

## Manual smoke test

```bash
python verify.py
python train.py \
  --output smoke_clean_physics \
  --train-only \
  --regimes large \
  --model-kinds hybrid \
  --large-alpha 1.0 \
  --training-target clean_physics \
  --noise-conditioning film \
  --quick
```

Then evaluate:

```bash
python evaluate_sweep.py \
  --results smoke_clean_physics \
  --output smoke_clean_physics_eval \
  --snrs -10 -5 0 5 10 15 20 \
  --fixed-snrs -10 -5 0 10 20 \
  --allow-smoke
```

## Paired comparison against the previous FiLM model

```bash
python compare_denoising_runs.py \
  --baseline /path/to/old_film_results \
  --candidate /path/to/new_clean_physics_results \
  --output comparison \
  --regime large \
  --seed 11 \
  --snrs -10 -5 0 5 10 15 20 \
  --scenes 500 \
  --batch-size 32
```

The comparison reports both:

1. true undeformed-channel NMSE, and
2. NMSE to the clean-PhysicsBridge target.

The second metric verifies whether the new training objective actually made the model a
better denoiser, even if the final H0 improvement is modest.

## Important interpretation

This experiment is **not intended to break the high-SNR PhysicsBridge structural floor**.
Its narrow goal is to improve the low-SNR correction by removing the conflicting task of
learning PhysicsBridge structural bias at the same time. If this helps substantially at
`-10/-5/0 dB`, the later architecture can separately address the high-SNR structural floor
using raw multi-deformation observations + geometry.
