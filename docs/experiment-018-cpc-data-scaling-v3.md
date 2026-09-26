# Experiment 018 v3: matched CPC continuation from the verified cache

**Frozen 26 September 2026, before v3 profiling or training.** The user asked
to train on the new 115,359-record cohort. V1 direct reads failed their cost
gate. V2 built and independently verified a reusable local 250 Hz cache, then
measured both V100 arms. Its complete-study projection was **8,561.00 seconds**
against 7,200 because it charged the one-time 4,410.00-second cache build.
V2 did not start either full training arm. Its immutable
[profile receipt](../outputs/experiment018_cpc_data_scaling_v2/profile.json)
has SHA-256 `b8cb9b79f1ab23e3abe562f6ad68e0afe4d4daf3fe671eb81545127271a2ba81`.

V3 starts **after** cache completion and acceptance. Its 7,200-second ceiling
applies to new v3 executable work: two full-file preflights (profile and
production), real-data V100 profile updates, both training arms, and reporting.
The historical cache preparation and v1/v2 work remain disclosed above, not
charged again as unfinished work. This scope is fixed before v3 profiling and
does not change in response to model outcomes. The runner records the prior
4,410.00 seconds separately in its profile receipt.

The scientific arms, seed and optimizer are unchanged from v2: both start from
the experiment-004 ordinary compact CPC encoder with seed-18042 prediction
heads; batch 128, AdamW learning rate 1e-4, weight decay 0.01, the original
pool's train-only normalizer, ordinary CPC loss, and **115,359 exposures per
arm**. The old arm draws with replacement from the 56,875-record original
pool. The new arm traverses the already seed-shuffled 115,359-record manifest
once, using sequential reads from the verified cache. Four data-loader
workers are fixed for both arms. Every target is masked; no held-out patient
enters SSL. The intervention is continuation-data diversity, not an
architecture or label-budget change.

The local cache SHA-256 is
`df4a270e46a2870ac66b40bf73e8ad3c3dbf702aa08e98ac67de1393d772fef4`.
Its independent [verification receipt](../outputs/experiment018_cpc_data_scaling_v2/cache_verification.json)
rehashed the whole file, replayed 27 source waveforms and matched eight PTB
waveforms bit for bit to the historical CPC cache. The runner rehashes the
cache once on each v3 launch and binds source code, initial checkpoint,
normalization and the verification receipt. Local output is
`outputs/experiment018_cpc_data_scaling_v3/`; raw sources stay unchanged.

V3 first measures 24 real input-plus-GPU updates per arm. Its frozen gate is
`2 × measured preflight + profile elapsed + 1.25 × projected 902 updates per
arm + 180 seconds` ≤ 7,200. Stop if the gate fails or actual pace makes the
remaining work exceed the ceiling. Each training arm saves optimizer, model,
RNG and fixed-order progress every 100 updates. Completion of both arms and
their loss receipts is the training-only endpoint. SSL loss is not a model
quality or clinical result; a separately frozen development readout is needed
before comparing representations. Calibration and test remain closed.

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_scaling_pilot_v3 --stage profile
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_scaling_pilot_v3 --stage train
```
