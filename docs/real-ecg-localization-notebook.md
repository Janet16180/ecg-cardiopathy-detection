# Real ECG localization notebook

The portable notebook explains the completed research with real patient recordings, saved screening
decisions, and expert-annotated comparisons. It uses no synthetic ECGs and runs no training or new
performance evaluation. Its Markdown sections explain the method, findings, and conclusions.

## Open the local artifacts

- Notebook: `outputs/real_ecg_notebook_2026_10_02/real-ecg-localization.ipynb`.
- Offline reading version: `outputs/real_ecg_notebook_2026_10_02/real-ecg-localization.html`.
- Executed verification: `outputs/real_ecg_notebook_2026_10_02/validation.json`.

The notebook embeds the selected original waveform samples and saved results. Restarting the kernel
and running all cells requires Python, NumPy, Matplotlib, and IPython. It needs no original data
folders, repository imports, model weights, GPU, or network access. The HTML embeds every figure and
needs no Python or external page assets. The source links are optional references.

The [repository blueprint](../notebooks/14-jr-real-ecg-localization.ipynb) excludes patient waveforms
and outputs. The builder below produces the complete local artifact; raw data remain outside Git.

## What the reader sees

Nine original 12-lead PTB-XL development ECGs illustrate normal annotations (47 and 91), a premature
ventricular beat (219), left and right bundle branch blocks (2135 and 1123), nonspecific T-wave changes
(127), combined infarction-pattern annotations (184), a probable old inferior infarction annotation
(1577), and an LVH annotation (30). These are label-based examples, not a new accuracy estimate.
Every code is retained, so a mixed label is not presented as a single confirmed diagnosis.

The full recordings and two-second close-ups show the original voltages and existing candidate
marks. The saved combined screening rule flags five examples and does not flag four. The T-change
and LVH examples are explicitly shown without referral or highlights. Voltage alone remains an
undefined clinical referral criterion; its unflagged example does not establish a clinical error.

Two INCART excerpts show a successful abnormal-beat selection and a failure of the newer method.
Two QTDB excerpts show improved QRS boundaries and a deliberately selected boundary failure.
Their legends distinguish algorithm candidates from expert annotation timing. The QRS truth is
joint timing across two channels, not a disease-location annotation for an individual lead.

## Findings and conclusions

The completed [INCART evaluation](experiment-056-incart-beat-localization-results.md) increased
patient-averaged ventricular ectopic beat selection from 74.1% to 89.6% in mixed segments across
30 eligible patients. The candidate still flagged 19.1% of expert-normal beats in the 31-patient
false-alarm comparison. Finding an unusual beat helps inspection but does not prove that its
highlighted segment is the pathological wave.

The completed [QTDB evaluation](experiment-059-hybrid-wave-boundaries-results.md) increased QRS
overlap from 68.2% to 79.4%, and the share of endpoints within 30 milliseconds of the expert mark
from 65.5% to 91.5%, across 72 records and 2,582 complete QRS intervals. This supports more precise
physiological timing. It does not validate disease localization or anatomical diagnosis. Patient
independence is unknown in QTDB, and T-wave improvement remains unproven.

The notebook keeps screening, abnormal-beat selection, and wave boundaries separate. It explains
validated components and their limits without promoting them into a fully validated clinical
localization system or changing the classifier or label definitions.

## Rebuild and verification

From the checkout with the already-completed outputs and original public recordings:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.reports.build_real_ecg_notebook
```

The builder imports existing shared helpers and replays cached scores; it does not load neural
encoders or refit thresholds. Combined decisions come from executed notebook 13 or are proved
unchanged over its recorded threshold rounding interval. The rule provenance stays in the payload.
The selected PTB records are development fold 9. No protected final-test cohort is accessed.

Executed on 2026-10-02: 13 real examples, 17 code cells, 24 inline figures, and zero errors.
All embedded waveform samples matched the original native WFDB samples exactly. A second complete
execution in an empty temporary folder reproduced all 24 PNG outputs identically. Saved cache and
selected input hashes were checked, and an independent scientific review passed. The repository
test suite passed all 1,372 tests; Ruff checks and formatting passed for the new Python files.
