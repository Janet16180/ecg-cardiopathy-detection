# MIMIC CPC flag audit

The read-only audit in `scripts/reports/count_mimic_cpc_flags.py` scores the
39,457 accepted MIMIC-IV-ECG records already present in the 250 Hz CPC pool.
It uses the full-label Experiment 004 CPC checkpoint, the original train-only
normalization and the PTB-XL calibration threshold 0.2422915399. The checkpoint
was fine-tuned on 15,360 labeled PTB-XL ECGs. CPC representation pretraining
included these MIMIC waveforms, but no MIMIC diagnosis labels were used.

The output `outputs/mimic_cpc_flag_audit/report.json` contains only aggregate
counts. `identified_scores.csv.gz` is a small **local-only** table with MIMIC
study and subject identifiers, scores, flags and score bands. Band indices 0–5
correspond to score intervals [0, .1), [.1, .25), [.25, .5), [.5, .75),
[.75, .9), and [.9, 1]. These bands group model confidence; they are **not**
clinical diagnoses or different anomaly types. The original MIMIC headers in
this download identify subjects and waveforms but do not supply a diagnosis
category for these rows. The script creates no new waveform copy and leaves
raw and processed source data unchanged.

The completed 26 September 2026 CPU pass flagged **34,675 / 39,457 ECGs
(87.88%)** and **6,699 / 7,892 patients (84.88%)**. The compressed identified
table is 474,015 bytes. Its 39,457 ECG IDs are unique; recomputing the ECG and
patient totals from that table exactly reproduces the aggregate report. The
full inference took 387.48 seconds. The initial 512-record profile projected
416.45 seconds, below the 7,200-second cost gate. Saved PTB-XL test logits
replayed with maximum absolute error 9.54e-7.

This is an exploratory diagnostic-annotation **proxy** audit, not a count of
confirmed heart problems or a prevalence estimate. Its threshold was calibrated
on PTB-XL patients and has not been validated on MIMIC. MIMIC is a hospital
cohort, so distribution and case mix can shift the flag rate substantially.
Repeated ECGs from the same patient are counted separately in the ECG count;
the patient count deduplicates subject IDs. Score bands should only be used to
prioritize review or to plan a later annotation-linked evaluation. They cannot
tell whether a flagged ECG has arrhythmia, ischemia, conduction disease or
another specific problem. Obtain linked diagnosis or clinician review labels
and validate them before reporting any condition-specific counts.

Run `uv run --no-sync python -m scripts.reports.count_mimic_cpc_flags --profile`
and then the same command without `--profile`. The profile replays 16 historical
PTB-XL test logits and times 512 cached MIMIC records. The full pass requires a
profile projection under 7,200 seconds and uses the shared GPU lock. The cache
receipt records the full waveform SHA-256; this audit checks the receipt and
rehashes the aligned row manifest, but does not reread all waveform bytes to
reconfirm the historic digest.
