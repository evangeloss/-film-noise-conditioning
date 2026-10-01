# Low-SNR-biased + FiLM experiment

This project variant keeps the existing PhysicsBridge and residual hybrid estimator, but makes two low-risk changes aimed specifically at the noise-dominated regime.

## What changed

1. **Low-SNR-biased training sampling**
   - Training SNR support: `[-10, -5, 0, 5, 10, 15, 20]` dB.
   - Default probabilities: `[0.25, 0.25, 0.20, 0.15, 0.07, 0.04, 0.04]`.
   - Therefore 70% of training scenes are at `-10/-5/0 dB`.
   - Explicit-SNR evaluation is unchanged.

2. **FiLM noise conditioning**
   - The model still receives the original constant log-noise map.
   - A small MLP also converts normalized log noise variance into per-channel `gamma` and `beta` terms at the three encoder scales.
   - Feature modulation is `F' = F * (1 + gamma) + beta`.
   - FiLM output layers are zero-initialized, so the new branch starts as an exact identity.

3. **Experiment bookkeeping**
   - `configuration.json` records training SNRs, probabilities, test SNRs, `noise_conditioning`, and `architecture_version=low_snr_film_v1`.
   - Final evaluation supports `[-10,-5,0,5,10,15,20]` dB.
   - Old checkpoints remain evaluable: old configurations default to `noise_conditioning='none'` in inference/evaluation utilities.

## Recommended Kaggle run

Use `kaggle_low_snr_film.py`. Edit only:

```python
REPO_URL='https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git'
LARGE_ALPHA=1.0  # change if your target experiment uses another b/lambda
QUICK=True       # first run only; then set False
```

The launcher:

- clones the repository,
- installs requirements,
- runs `verify.py`,
- trains **only the large-deformation hybrid** with low-SNR-biased sampling + FiLM,
- evaluates Physics vs Hybrid at `-10,-5,0,5,10,15,20 dB`,
- creates a review ZIP,
- preserves the full checkpoint folder in `/kaggle/working`.

## Manual smoke test

```bash
python verify.py
python train.py \
  --output smoke_low_snr_film \
  --train-only \
  --regimes large \
  --model-kinds hybrid \
  --large-alpha 1.0 \
  --noise-conditioning film \
  --quick
```

Then evaluate:

```bash
python evaluate_sweep.py \
  --results smoke_low_snr_film \
  --output smoke_low_snr_film_eval \
  --snrs -10 -5 0 5 10 15 20 \
  --fixed-snrs -10 -5 0 10 20 \
  --allow-smoke
```

## Scientific comparison to run

For a clean ablation, compare three runs using the same seed/geometry/bridge settings:

1. Original sampling + no FiLM.
2. Low-SNR-biased sampling + no FiLM.
3. Low-SNR-biased sampling + FiLM.

The packaged project defaults to experiment 3. To isolate experiment 2, pass `--noise-conditioning none` while keeping the biased sampling arguments.
