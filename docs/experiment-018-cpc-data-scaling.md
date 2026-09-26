# Experiment 018: compact CPC data-scaling pilot

**Frozen 26 September 2026, before the real-data GPU profile.** The user
requested training on the new 115,359-record sampled cohort. This is a
training-only pilot; a downstream development readout requires a separately
specified analysis. It does not evaluate calibration or test patients.

## Question and controls

Does one additional ordinary CPC training pass over the new cohort learn a
different representation than the same number of updates on the original
56,875-record pool? Both arms start from the completed experiment-004 seed-42
CPC encoder, initialize prediction heads with seed 18042, and use the same
architecture, ordinary CPC objective, batch size 128, AdamW learning rate
1e-4 and weight decay 0.01. Both use the original pool's train-only per-lead
normalizer. The original arm samples 115,359 records with replacement from
56,875, while the new arm visits all 115,359 sampled records once in a seeded
order. This controls optimizer updates and record exposures while varying the
source pool and unique-record diversity. The new arm includes 15,359 PTB
records with available labels, but SSL masks every target. No held-out ECG is
used for training or fitting.

The new source is the verified [sampled 100k-plus-labels cohort](sampled-100k-plus-labels-v1.md)
with manifest SHA-256
`82a316a360a95dac3f24545c964afc1ad6b9654fa830541f6871b49371353288`.
Input waveforms are hash-checked on read, each five-second half is resampled
from 500 to 250 Hz by the historical CPC transform, and the raw files are
unchanged. The old arm reads the fixed experiment-004 cache. The runner hashes
small input manifests, normalization, initial checkpoint and source files in
its profile receipt and repeats that identity check before training. Its
output and checkpoint files remain local under
`outputs/experiment018_cpc_data_scaling/`.

The first profile exposed a loader/source-policy mismatch: the published MIMIC
audit accepted finite ECGs with a constant lead, while the generic training
loader rejected them. Two such records appeared in the first 318 sampled
profile reads. The loader now applies the MIMIC audit's finite-signal policy
to MIMIC rows and keeps the stronger variable-lead gate for other sources.
These records are a quality limitation to count in a later full-cohort EDA;
neither the raw source nor the frozen sample manifest was changed. The first
profile failed before a cost receipt or training arm and is not a result.

## Cost gate and interpretation

One real-data V100 profile takes 24 batches per arm using the exact loader,
resampling, optimizer update and GPU lock. It includes cold worker startup and
source-shard checks. Projected total is profile elapsed time plus 1.25 times
the measured per-batch times for 902 updates per arm, plus 180 seconds for
reporting. Training is admitted only if this is at most 7,200 seconds. This
projection can still miss later raw-file I/O variation; monitor actual pace
at each 100-update checkpoint and stop a run that exceeds the ceiling. A
passed profile is evidence of feasibility, not model performance.

The fixed endpoint of this pilot is completion of both SSL arms and their
training-loss receipts. Training loss alone does not establish downstream
quality. A subsequent frozen, paired development-only readout on the same
PTB patients and labels is needed before claiming an improvement; calibration
and test remain closed. Because the starting encoder already saw the old pool,
this is a continuation-data study, not a comparison of pretraining from
scratch. Cross-source patient overlap for Challenge/Chapman is unresolved.

Commands from the repository root:

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_scaling_pilot --stage profile
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_scaling_pilot --stage train
```
