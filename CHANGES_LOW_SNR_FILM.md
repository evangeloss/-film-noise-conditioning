# Changes in the low-SNR + FiLM project

## Modified files

### `simulator.py`
- Added configurable training SNR support/probabilities.
- `train.py` now passes the low-SNR-biased distribution:
  - SNRs: `[-10,-5,0,5,10,15,20]` dB
  - Probabilities: `[0.25,0.25,0.20,0.15,0.07,0.04,0.04]`
- Uses deterministic inverse-CDF categorical sampling when `snr=None`.
- Explicit fixed-SNR evaluation behavior is unchanged.
- Keeps `return_latents=True` support for oracle-floor diagnostics.

### `model.py`
- Added `NoiseFiLM`.
- Added FiLM conditioning at encoder scale 1, encoder scale 2, and bottleneck.
- Conditioning variable is normalized log noise variance: `log10(mean(variance)/scale^2)`.
- Retains the original constant log-noise input channel.
- FiLM output layers are zero initialized, so the new path begins as identity modulation.
- `Reconstructor(..., noise_conditioning='film'|'none')` keeps the same forward signature.

### `train.py`
- Added CLI arguments:
  - `--training-snrs`
  - `--training-snr-probs`
  - `--test-snrs`
  - `--noise-conditioning`
- Records `architecture_version='low_snr_film_v1'` in `configuration.json`.
- Passes the biased SNR distribution into `Simulator`.
- Uses the requested FiLM mode when constructing the model.
- Final test loops can cover the full `[-10,-5,0,5,10,15,20]` dB sweep.
- `load_state` allows zero-initialized FiLM parameters to be absent so older checkpoints can still be evaluated.

### `evaluate_sweep.py`
- Reads `noise_conditioning` from each run configuration.
- Uses each run's training SNR list when labeling trained/interpolated/extrapolated SNRs.
- Keeps compatibility with older non-FiLM checkpoints.

### `predict.py`
- Reconstructs the model with the correct `noise_conditioning` mode from `configuration.json`.

### `run_oracle_floor.py`
- Reconstructs the model with the correct `noise_conditioning` mode from `configuration.json`.

### `verify.py`
- Added a deterministic biased-SNR sampler check.
- Verifies the FiLM model still starts as the exact PhysicsBridge skip estimate and has finite gradients.

## New files

### `kaggle_low_snr_film.py`
Dedicated Kaggle launcher for this experiment. It trains only `large_hybrid`, then evaluates Physics vs Hybrid over the full SNR sweep.

### `Kaggle_Low_SNR_FiLM.ipynb`
Notebook version of the dedicated Kaggle launcher.

### `LOW_SNR_FILM_README.md`
Run instructions and suggested ablations.

## Validation performed before packaging

- `python -m py_compile *.py` — passed.
- `python verify.py` — passed.
- `train.py --quick --train-only --regimes large --model-kinds hybrid --large-alpha 1.0 --noise-conditioning film` — passed.
- `evaluate_sweep.py` on the quick checkpoint at `-10,-5,0,5,10,15,20` dB — passed.
- `run_oracle_floor.py --quick` on the new FiLM checkpoint — passed; perfect oracle remained at numerical precision (~`-143 dB`).
