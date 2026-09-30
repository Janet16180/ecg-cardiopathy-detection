# Experiment 012 v3: sealed cache and staged 25k training path

**Prospective implementation, 26 September 2026.** This is a new executable
successor to the [original Experiment 012 protocol](experiment-012-temporal-hybrid.md).
It has no V100 profile, training result, or development result yet. The v1 and v2
profile manifests and their sources remain historical. A new verified executable
manifest is required before any GPU stage.

The scientific comparison is unchanged: fresh GRU-CPC reference and the mixed
versus local temporal hybrid arms, with identical mixed/local parameters and
initial values. Keep seed 12012, AdamW learning rate `1e-4`, weight decay `0.01`,
the ordinary causal CPC objective at horizons 4/8/12, the same stem and
prediction-head initialization, and the same independent five-second halves.
The frozen Experiment 019 selection has 25,000 rows, selected-index SHA-256
`43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791`.
Seed 18047 gives the same 115,359 exposures, 902 updates at batch 128, and
short final batch. Training still writes resumable model, optimizer, RNG, and
identity checkpoints every 100 updates. The PTB-XL development readout retains
512 mean/max features, 15,359 and 1,518 clean training labels, fixed `C=0.01`
train-only logistic heads, 1,306 development ECGs, and paired patient bootstrap
seed 12045. Calibration and test remain closed.

The shared cache-session seal at
`outputs/cache_sessions/nlp25k_v1/seal.json` is created independently from
architecture code by a separate one-time fresh full-cache SHA-256 check;
Experiment 011 v1's hash pass was interrupted before completion. Creation binds
both waveform cache SHA-256 values to their source receipts. Every v3 stage
validates the seal digest, receipt hashes, file identity and bounded content
blocks, and pins the seal digest in its run identity. The v3 stages **do not
repeat a fresh full SHA-256 of either waveform cache on every launch**. This
trades per-stage complete payload rehashing for the explicit session-seal
integrity checks; any seal creation failure or subsequent validation mismatch
blocks the stage. Other source, label, manifest, and checkpoint hashes remain
stage-specific.

The profile and training stage each copy the selected 25,000 waveforms to V100
once as exactly normalized float32 tensors, 3.0 GB (about 2.79 GiB). A mapping
from frozen source rows to staged rows reproduces the entire original exposure
order for all three arms. The GPU profile measures that full staging pass, 24
updates per arm, checkpoint write/replay, and training-only PTB extraction. Its
7,200-second cost gate includes seal preflights, measured staging, an additional
1.25 times staging for production, 1.25 times projected updates and checkpoints,
1.5 times projected PTB extraction, the separate readout profile, 900 seconds
for CPU/readout work, and 180 seconds of training reserve. A passing profile
does not itself start training. The readout has its own V100 cost gate. GPU
stages use the shared lock and are sequential.

```bash
uv run --no-sync python -m scripts.experiments.run_cpc_temporal_hybrid012_v3 --stage check
uv run --no-sync python -m scripts.experiments.run_cpc_temporal_hybrid012_v3 --stage profile
uv run --no-sync python -m scripts.experiments.run_cpc_temporal_hybrid012_v3 --stage train
uv run --no-sync python -m scripts.experiments.run_cpc_temporal_hybrid012_readout_v3 --stage check
uv run --no-sync python -m scripts.experiments.run_cpc_temporal_hybrid012_readout_v3 --stage profile
uv run --no-sync python -m scripts.experiments.run_cpc_temporal_hybrid012_readout_v3 --stage run
```

These commands are prospective and require a separately frozen, verified
successor manifest and an explicit queue launch. A profile loss or checkpoint
is not a downstream performance result.
