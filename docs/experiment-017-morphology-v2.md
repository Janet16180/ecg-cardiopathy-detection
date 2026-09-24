# Experiment 017 version 2: profile checkpoint device correction

**Frozen 24 September 2026, before a version 2 GPU profile.** The scientific protocol remains [Experiment 017](experiment-017-morphology.md): three arms at both label budgets, the same train-only template bank, 250 Hz native tokens, fixed five-epoch schedule, development-only selection, and the same whole-pilot two-hour gate.

The frozen first runner created its profile checkpoint roundtrip probe on the requested training device. During a CUDA profile, that meant the probe and AdamW slots were on CUDA while the saved checkpoint tensors were loaded on CPU, making its equality check device-dependent. Version 2 constructs the roundtrip probe on CPU, then verifies model tensors, optimizer tensors, exact epoch/batch position, and Python/NumPy/Torch/CUDA RNG state against the CPU-loaded checkpoint. The three real full-epoch profile arms still train and evaluate on the V100. This correction changes verification only; it does not alter the model, loss, optimizer, data, schedule, or training code.

The versioned entry point includes its own source and this addendum in the provenance fingerprint. It writes to `outputs/experiment017_morphology_templates_v2/`, preserving the first runner's CPU verification receipt. The complete waveform pool SHA-256 verification is reused via the immutable input file identities in Experiment 015's receipt. A new version 2 CPU check receipt and real-GPU profile are required before version 2 training. The active executable queue must be superseded with a separate immutable manifest; its sources and manifest are not edited in place.

```bash
.venv-pretrained/bin/python -m scripts.run_cpc_morphology017_v2 --stage check --device cpu --threads 1
.venv-pretrained/bin/python -u -m scripts.run_cpc_morphology017_v2 --stage profile --device cuda --threads 1
.venv-pretrained/bin/python -u -m scripts.run_cpc_morphology017_v2 --stage train --device cuda --threads 1
```
