# Experiment 019: 25k source-stratified CPC continuation

**Frozen 26 September 2026 before profiling, training or new development
predictions.** The user requested a 25k follow-up after the 115,359-record
mixed-cohort CPC continuation gave no measurable development improvement.
The hypothesis is that repeating a smaller, source-proportional training pool
could produce a better PTB frozen-feature readout at the same compute budget.
This is a data-diversity/sampling test, not a claim that more data is harmful.

Select **25,000 total ECGs** without replacement from the exact verified
115,359-record training manifest used in Experiment 018. Use source-stratified
largest-remainder quotas proportional to that manifest, deterministic sorted
source names, and NumPy seed **18046**. Shuffle within each source selection,
then seed-shuffle the combined indices. Construct 115,359 record exposures by
concatenating fresh full-subset permutations using seed **18047** and truncating
the last cycle. This visits every selected ECG once before reuse. All source
records are training split; no diagnostic labels are used during CPC. The
25k selection is a subset of the existing verified local 250 Hz cache, with
no new waveform copy or raw-data modification. Record exact source quotas,
selected-index hash, input hashes and seeds in the profile receipt. Known
cross-source identity uncertainty from the parent cohort remains.

Initialize exactly as the completed v3 arms: experiment-004 compact CPC
encoder, prediction-head seed 18042, ordinary CPC objective, batch 128,
AdamW at 1e-4 and weight decay 0.01, the historical train-only normalizer,
four workers and **902 updates / 115,359 exposures**. The already completed
115k mixed-cohort arm is the primary matched-update comparator; the old-pool
arm and unchanged starting encoder are descriptive references. Existing
checkpoints remain immutable. This single 25k arm writes to a new output
directory and must not be treated as a performance result until a separately
frozen development readout is complete. CPC losses from different sampling
streams are not directly comparable.

The primary downstream contrast will be 25k minus 115k development AUROC
using the *same* fixed 512-feature CPC encoder readout, all 15,359 clean PTB
training labels and 1,306 development ECGs/1,173 patients. The 1,518-label
budget is secondary. Use the fixed scaler/logistic settings and paired
patient bootstrap of Experiment 018's frozen readout, with no hyperparameter
or checkpoint selection on development. We will freeze a successor readout
runner and comparison receipt **before** viewing new development scores.
Calibration and test patients stay closed.

Admission requires a full SHA-256 of the reusable 13 GB cache, verification
receipt and source hashes on each launch. Profile 24 actual input-plus-GPU
updates on the V100 under the shared lock. The gate is two measured preflights
plus profile elapsed plus 1.25 times the projected 902-update training time
plus 180 seconds, capped at 7,200 seconds. Stop this arm if the gate fails or
observed cost becomes incompatible. Save model, optimizer, RNG and update
count every 100 updates for exact resumption. The existing cache preparation
is historical sunk work, disclosed in Experiment 018 and not charged again.

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_25k --stage profile
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_25k --stage train
```

Local output: `outputs/experiment019_cpc_25k_v1/`.
