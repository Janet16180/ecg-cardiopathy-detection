# Clean cohort cached probe rerun v1

25 September 2026. This CPU-only development study refits the released ECG-JEPA
and Experiment 009 ordinary local-plus-context CPC linear probes on the clean
PTB-XL labeled selections in `outputs/data_quality/clean_rerun_preflight_v1`.
The full budget has 15,359 labels after one waveform exclusion; the fixed
limited budget retains its 1,518 labels. Development has 1,306 fixed patients'
records. Calibration and test are not used for model selection or evaluation.

Both encoders and their historical preprocessing remain frozen. Feature caches
must match the hashes recorded by Experiment 014, and the CPC input audit is
part of this run's identity. This transfer comparison does not remove the
excluded record from the encoders' prior exposure or refit encoder normalization.
It is distinct from the expanded 76,598-record union and any Challenge scaling
study. The endpoint remains a diagnostic annotation proxy.

For each model and budget, join cached features to the original PTB-XL manifest
by ECG ID, verify patient/split identity, and require each clean label to match
its original row exactly. Fit `StandardScaler` on the retained labeled training
rows only. Fit `LogisticRegression(solver="lbfgs", max_iter=3000,
random_state=42)` for C = `.001, .01, .1, 1, 10, 100`; choose highest
development AUROC, with the first C winning ties. Save selected parameters,
each grid score, exact input hashes and source hashes. Report original development
scores beside the clean refits. No calibration or test scores are produced.

An optional repeat of the 014 fusion screen may use the newly fitted probes,
with the same five fixed weights, training-logit normalization, patient folds,
bootstrap and development gate as the original protocol. A positive screen
requires a separately frozen calibration/test follow-up; it does not trigger
one automatically.
