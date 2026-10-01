# Validation performed before packaging

The packaged project was checked locally with the following stages:

1. `python -m py_compile *.py` — passed.
2. `python verify.py` — passed, including:
   - reference steering agreement,
   - packing/unpacking,
   - full-band noise-power identity,
   - noise calibration,
   - PhysicsBridge algebra,
   - biased SNR sampler,
   - FiLM identity initialization,
   - clean-PhysicsBridge denoising target,
   - finite gradients.
3. Quick clean-physics training at `b/lambda=1` — completed for 2 epochs.
4. `evaluate_sweep.py` on the quick checkpoint — completed over `[-10,-5,0,5,10,15,20]` dB.
5. `compare_denoising_runs.py` against the previous FiLM smoke checkpoint — completed on paired scenes.
6. `run_oracle_floor.py --quick` on the new checkpoint — completed; perfect oracle remained approximately `-143.4 dB`.

Smoke-run numerical scores are software checks only because quick mode uses a tiny bridge/network.
