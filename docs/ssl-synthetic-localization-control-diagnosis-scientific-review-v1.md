# Scientific review of the saved-map control diagnosis

19 October 2026. Scientific source review supports one bounded execution of the saved-map diagnostic.
This is a prospective review, not a diagnostic result. Protocol/config were frozen in `5f4d663`
before any new hotspot metric. Reviewed implementation commit: `b892774`.

| Reviewed artifact | SHA-256 |
|---|---|
| `ecg_experiment/ssl_synthetic_control_diagnosis.py` | `f7d024717a782bdd1bf666b9aa56d9d1e5a90894ab257fcd85ec919d975ec7ac` |
| `configs/ssl_synthetic_control_diagnosis_v1.json` | `62f2d40f906bf3f16cf817aef609af4e39275058a4c2c0a0cd327524c7a9f2b2` |
| `docs/ssl-synthetic-localization-control-diagnosis-v1.md` | `63406079c45ccff3482f0d42dc2844ce5bd3e5d5bdfab85aa7cd78fca4fd45b1` |

Read the complete module, thin command wrapper and seven artificial tests. The implementation first
verifies the explicit parent anchors, admits only evaluation numeric slices, and reproduces the old
case metrics, controls, contrasts and scientific gates exactly before computing new diagnostics.
Whole-file hashes authenticate the declared arrays but do not authorize numeric TRAIN-row analysis.
No checkpoint, native generator array, real ECG, model forward or optimizer update is used.

The primary matches the frozen definition: exactly 600 stable-ranked cells per map; first-sample
prepending gives an initial zero derivative; strict per-lead upper-decile smoothed derivative energy;
one common fixed temporal shift across leads, variants and arms for each global seed; equal sham/
nuisance weighting within seed; and three paired whole-seed bootstrap estimates. Secondary support,
matched-control persistence, lead allocation and fixed TRAIN-prior overlap retain their specified
denominators. The prior uses the same deterministic tie rule, with tie counts exposed in outputs.

The artificial tests cover tied and nonfinite scores, constant nonzero waveforms, the common shift,
known support denominators, paired resampling, complete CSV rows, corrupted-parent admission and
exact predecessor-metric corruption. They use artificial arrays, not actual diagnostic inputs.
Source snapshots, before/after input bindings, explicit CPU/wall deadlines and terminal ledgers
preserve the bounded accounting contract. Receipt writing/hashing precede terminal measurement;
final ledger serialization remains the declared reserved bookkeeping tail.

Scientific GO is limited to this source-frozen descriptive study after coordinator integration,
required checks and independent numerical review. A positive or negative alignment difference is
not a new scientific PASS/FAIL or a permutation-test result. The proxy does not establish QRS/P/T
boundaries, disease, healthy status or a causal explanation of the model. Stable tie selection can
produce apparent location preferences; lead allocation and prior overlap must be read alongside it.
The fixed-fit, spent-generator-seed intervals do not represent patient or training-seed uncertainty.
No prior learning verdict changes, and no new fit or threshold tuning follows from this review.
