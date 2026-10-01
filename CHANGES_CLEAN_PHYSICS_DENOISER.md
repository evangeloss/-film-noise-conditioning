# Changes: clean-PhysicsBridge denoiser

## 1. `simulator.py`

`Simulator.batch(...)` now accepts:

```python
return_clean=False
```

When enabled, the returned dictionary contains `clean`, the noise-free multi-deformation
observations. Existing calls are unchanged.

## 2. `train.py`

New option:

```bash
--training-target true_channel|clean_physics
```

For `clean_physics`, each noisy training sample is paired with:

```python
clean_x = pack(clean) / noisy_scale
clean_target = PhysicsBridge(clean_x, zero_variance)
```

The noisy and clean targets therefore use the same scene normalization. The model is trained
with NMSE between its hybrid output and `clean_target`, which is equivalent to learning the
noise-induced PhysicsBridge correction.

Checkpoint selection and LR scheduling still use **true H0 validation NMSE**. Training logs
also record `validation_clean_physics_nmse` as a denoising diagnostic.

Configuration records:

```text
training_target = clean_physics
architecture_version = low_snr_film_clean_physics_v1
```

## 3. FiLM + biased SNR sampling retained

No change to the previous FiLM architecture or SNR distribution. This isolates the effect of
changing the target.

## 4. `verify.py`

Adds checks that:

- clean-physics supervision can be constructed,
- it differs from noisy PhysicsBridge at low SNR,
- gradients through the denoising objective are finite.

## 5. New comparison utility

`compare_denoising_runs.py` evaluates old and new hybrid checkpoints on identical scenes and
noise draws, and reports:

- Physics true-channel NMSE,
- old Hybrid true-channel NMSE,
- new denoiser true-channel NMSE,
- new-vs-old improvement,
- old/new distance to the clean-PhysicsBridge target.

## 6. New Kaggle launcher/notebook

- `kaggle_clean_physics_denoiser.py`
- `Kaggle_Clean_Physics_Denoiser.ipynb`

The launcher trains only `large_hybrid` with `--training-target clean_physics`, evaluates the
SNR sweep, and can optionally run the paired comparison against a previous FiLM results folder.
