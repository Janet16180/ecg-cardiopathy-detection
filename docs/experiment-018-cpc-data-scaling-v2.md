# Experiment 018 v2: cached continuation data-scaling pilot

**Frozen 26 September 2026 after the v1 profile cost stop and before v2 GPU
profile or training.** The v1 real-data profile passed 24 old-pool updates in
51.41 seconds and 24 direct new-cohort updates in 352.46 seconds. Its frozen
projection was 19,565.44 seconds, above the 7,200-second ceiling; no full
training arm started. That profile remains in
`outputs/experiment018_cpc_data_scaling/profile.json`.

V2 retains the v1 scientific comparison exactly: one ordinary compact CPC
continuation pass over 115,359 record exposures per arm, starting from the
same experiment-004 encoder with seed-18042 prediction heads; batch 128,
AdamW 1e-4 and weight decay 0.01, original train-only normalization. The old
control samples 115,359 exposures with replacement from the 56,875-record
pool. The new arm visits each record in the seeded 115,359-record cohort once.
All labels are masked. No development, calibration or test patient enters SSL.
The cache loader uses four workers for both profiled and training arms.
The new arm traverses the cohort manifest in its already seeded shuffled order;
this preserves the random selection order while making cache reads sequential.

The sole implementation change is a **local, derived 250 Hz cache** for the
new arm. The builder reads and hashes every original waveform through the
published manifest, applies the exact historical CPC resampling separately
to each five-second half, and writes one manifest-aligned float32 array. It
does not modify raw data or the frozen cohort. A completion receipt hashes
the full cache and binds source metadata, selection manifest, loader and
resampler code. The new runner rechecks the full cache hash before profiling
and training. The cache uses about 13.8 GB and stays local under
`data/processed/sampled_100k_cpc_cache_v1/`.

The local cache completed with 115,359 rows and SHA-256
`df4a270e46a2870ac66b40bf73e8ad3c3dbf702aa08e98ac67de1393d772fef4`.
Its completion receipt has SHA-256
`8ab358ee040cd876948e406067524e30dc20b4d10885c65d734f95889ab8e37b`.
The [independent replay receipt](../outputs/experiment018_cpc_data_scaling_v2/cache_verification.json)
has SHA-256 `5d8f8b13f1e535270a7974e54b3e12a31316b34ab56959cf61f7517a9a5dec92`:
it rehashed the complete cache, replayed 27 source rows, and found eight
bitwise matches to the historical PTB CPC cache. The builder counted 114
accepted MIMIC recordings with a constant lead. Resumable local preparation
took 4,410.00 measured process seconds, excluding the independent validation.

The v2 profile again measures 24 real batches and GPU updates per arm using
the actual cache loader. The 7,200-second gate includes measured cache-build
time, both profile and production preflight/hash passes, profile update time,
1.25 times projected full-training update time, and 180 seconds
for reporting. Stop if the gate fails; do not reinterpret the failed v1
projection as a pass. No downstream performance conclusion follows from SSL
training loss. A separate frozen development readout is required.

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.data.cache_sampled_cpc
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_scaling_pilot_v2 --stage profile
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_scaling_pilot_v2 --stage train
```
