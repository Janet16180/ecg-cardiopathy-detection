# Experiment 019: 25k CPC training result

The 25k source-stratified CPC continuation completed **902 optimizer updates**
and **115,359 record exposures** from 25,000 distinct training ECGs. It was
initialized from the same experiment-004 encoder and predictor seed as the
completed 115k continuation. Its mean CPC training loss was 2.42322, which
does not rank representation quality across different sampling streams.
The [checkpoint audit](../outputs/experiment019_cpc_25k_v1/training_audit.json)
passed: the checkpoint SHA-256 is
`1ac0035fe07aa421853056b99cf13300418546aa787cf8ded749c412cfd4d523`,
all 902 updates are saved with optimizer and RNG state, all 24 encoder tensors
changed, and the weights are finite.

The source-proportional selection contains 15,170 MIMIC, 3,746 PTB-XL, 2,072
Chapman, 2,066 Georgia, 1,334 CPSC 2018, and 612 CPSC Extra ECGs. Selection
seed 18046 and exposure-order seed 18047 reproduce the subset and cycling
schedule. Its canonical selected-index SHA-256 is
`43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791`.
No diagnostic labels were used in this CPC stage; raw and cached ECGs were
not modified.

The full-path 24-update V100 profile took 44.93 seconds after a 947.32-second
full-cache preflight. Its conservative total projection of 4,230.55 seconds
passed the 7,200-second gate. Production repeated the full-cache hash and
completed the 902 training updates in 319.30 seconds. Exact commands:

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_25k --stage profile
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_25k --stage train
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -m scripts.validation.audit_cpc_25k
```

The local [profile](../outputs/experiment019_cpc_25k_v1/profile.json),
[completion receipt](../outputs/experiment019_cpc_25k_v1/complete.json),
checkpoint and audit are the source evidence. This is **training-only**;
the frozen [development readout](experiment-019-cpc-25k-readout.md) is needed
before comparing model quality. Calibration and test patients remain closed.
