# KEY-04 secondary negative-pool control

Prospective addition during design review, 6 November 2026. Training-only CV has begun, but no
KEY-04 SPH prediction artifact exists and no new SPH model result has been inspected. The original
primary comparison, CV selection and ten reference draws remain frozen and unchanged.

Review identified a fairness distinction: KEY-01 keys contrast each condition only with normal
references, whereas the new classifiers learn distinctions among all annotation combinations.
Add one descriptive prototype control to isolate negative-key composition without fitting another
classifier: preserve the same training-fitted scaler, unit vectors and each positive key, but
average all training records lacking the requested group to make its negative key. An ECG with
co-occurring group c remains positive for c and never enters c's negative pool.

Score the same SPH evaluation rows with positive-minus-all-other-negative cosine margins. Report
four-group/macro AUROC and AP, the normal-only subsets, training positive/negative counts and a
paired 2,000-draw patient interval versus the frozen whole-record keys (seed 20261103). This is an
exploratory control, not another primary candidate and not a replacement for the original negative
prototype. No other choices or refits follow its result. Also place its point metrics beside both
logistic families, without declaring a new independently confirmed winner.

Implement in new files after preserving the original running sources. Require the positive keys
and whole-record comparator to reproduce KEY-01 within 1e-12; verify parent inputs and models
before reading its result. Save its own protocol/source/input/output bindings and private scores.
This control addresses negative-reference composition, not classifier capacity, co-annotation
modeling, pretraining exposure, clinical calibration or source invariance.
