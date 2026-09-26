# Experiment 019: 25k development readout result

The 25k source-stratified CPC continuation scored **0.91960 development
AUROC** with all 15,359 clean PTB training labels, versus **0.91795** for the
matched 115,359-record continuation. The prespecified 25k-minus-115k
difference was **+0.00164**, with a 95% paired patient-bootstrap interval of
**-0.00073 to +0.00396**. The interval includes zero, so this one subset and
seed do **not** establish that using less data improves CPC. At 1,518 labels,
the corresponding difference was +0.00208 (interval -0.00034 to +0.00443),
also inconclusive. The unchanged starting encoder retained the highest point
AUROC at both budgets.

| Training labels | Frozen encoder | Development AUROC | Average precision |
| ---: | --- | ---: | ---: |
| 15,359 | Unchanged starting CPC | 0.92079 | 0.96145 |
| 15,359 | Historical old-pool continuation | 0.91809 | 0.95998 |
| 15,359 | 115k mixed-cohort continuation | 0.91795 | 0.95986 |
| 15,359 | **25k source-stratified continuation** | **0.91960** | **0.96021** |
| 1,518 | Unchanged starting CPC | 0.91296 | 0.95762 |
| 1,518 | Historical old-pool continuation | 0.90564 | 0.95370 |
| 1,518 | 115k mixed-cohort continuation | 0.90658 | 0.95398 |
| 1,518 | **25k source-stratified continuation** | **0.90866** | **0.95478** |

Against the unchanged starting encoder, 25k differed by -0.00119 at full
labels (interval -0.00607 to +0.00366) and -0.00430 at limited labels
(interval -0.00961 to +0.00085). All intervals used 2,000 paired whole-patient
draws, with no invalid single-class draws. The primary contrast was fixed in
the [training protocol](experiment-019-cpc-25k.md) and [readout protocol](experiment-019-cpc-25k-readout.md)
before these new predictions were generated.

The 25k and 115k arms began from the same compact CPC encoder and prediction
head seed and each completed 902 updates/115,359 record exposures. The 25k
arm cycled through its 25,000 distinct ECGs; the 115k arm visited each ECG
once. Both received the same historical train-only waveform normalizer. This
isolates a practical sampling/diversity change under one training recipe,
but it does not separate source composition, repeated exposure, or stochastic
seed effects. Development patients have been repeatedly inspected in this
project, and this binary endpoint is an ECG diagnostic-annotation proxy. No
calibration/test patient was featurized or scored. A paper claim would require
fresh patients or tasks and independent seeds, not another favorable point
estimate on this development set.

The train-only V100 readout profile projected 1,668.85 seconds against the
7,200-second gate; full feature extraction took 31.38 seconds after production
input verification. The local [result receipt](../outputs/experiment019_cpc_25k_readout_v1/result.json)
has SHA-256 `afd880190501d356e53b06d2194f0783cf8347915a05c5826552dbc532f0e060`.
The independent [audit](../outputs/experiment019_cpc_25k_readout_v1/readout_audit.json)
passed saved feature/prediction hashes, exact development record/patient/label
alignment to Experiment 018, AUROC, average precision, and contrasts. Arrays
and checkpoint bytes remain local and outside Git.

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_25k_readout --stage profile
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_cpc_25k_readout --stage run
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -m scripts.validation.audit_cpc_25k_readout
```

No additional subset sweep, calibration/test evaluation, or GPU job is
scheduled by this result. The stronger paper-oriented priority remains the
[controlled representation-versus-readout investigation](experiment-016-paper-investigation.md).
