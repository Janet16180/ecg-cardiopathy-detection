# Randomized localization independent audit results

19 October 2026. The independent saved-output audit passed with zero mismatches. It reproduced all
12 scientific gates and nine integrity gates, including direct localization gains over both
comparators. Supervised mean positive AP was 0.077538, versus 0.024522 for absolute amplitude and
0.021353 for the matched model trained on misplaced masks. Absolute AP remains limited, and many
unedited controls exceeded the descriptive 0.5 maximum-probability cutoff. This is learning of
introduced numerical changes on generated backgrounds, not clinical disease localization.

## Independently reproduced findings

| Complete-grid AP | Focal | Repeated | Equal-mode mean |
|---|---:|---:|---:|
| Supervised model | 0.053579 | 0.101497 | 0.077538 |
| Permuted-target model | 0.003642 | 0.039065 | 0.021353 |
| Absolute observed amplitude | 0.006846 | 0.042197 | 0.024522 |
| Training-only spatial prior | 0.011604 | 0.037987 | 0.024796 |

The recomputed pooled supervised-minus-amplitude gain was 0.053016084, with 95% seed-bootstrap
interval [0.027368532, 0.083052243]. Supervised minus permuted was 0.056184530
[0.031927903, 0.083606838]. Both means exceeded the frozen 0.05 margin and both lower bounds exceeded
zero. Every mode-specific contrast lower bound exceeded zero, satisfying the prospective −0.01
mode guard. No mode was omitted or substituted in the pooled comparison.

Supervised AP response against the paired nuisance map was 0.050292079 [0.015805639, 0.092149683]
for focal and 0.078692806 [0.041261280, 0.116510845] for repeated changes. Selectivity was
0.042188734 [0.014197113, 0.074748989] and 0.047279951 [0.023302521, 0.073294834], respectively.
These distinguish introduced-change response from pre-existing agreement with the future edit site.
All intervals use the same 2,000 whole-generator-seed resamples, seed 2026103503, with zero skips,
48 evaluation seeds and equal two-mode weight within seed. They are conditional on one fitted
randomness and this generator/edit family; they are not patient or training-seed intervals.

Descriptive supervised control maxima exceeded 0.5 for 23/48 sham and 25/48 nuisance records. The
permuted model exceeded it for 0/48 in each control mode. Supervised mean control probabilities
were 0.091465 and 0.091546. The fixed cutoff was uncalibrated and had no advancement gate. These
rates describe recordings with no added local-offset target, not clinical false-positive rates.
They limit interpretation of the dense-AP gain as a usable screening or referral system.

## Data and training integrity

The audit regenerated every lead, sign, amplitude, width and time draw and required exact bytes
for all 512 observed variants, 256 strict masks and 512 thresholded masks. It verified coupled limb
leads, no changes outside intended hulls, energy retention, zero control targets and all 128 rows.
TRAIN indices 0–79 and EVALUATION indices 80–127 were disjoint. It reproduced all 400-by-eight batches:
each of the 320 training variants appeared ten times, with zero evaluation indices. The donor map
was exactly `(i+1)%80`, preserving the positive-mask collection. Training positive/negative cells
were 114,068 / 19,085,932, yielding the shared capped positive weight 20. The saved spatial prior
matched only the 160 positive TRAIN masks.

The immediate predecessor's full 128-seed coherence comparison and interval reproduced within the
audit's numerical tolerance; the production predecessor check required exact equality. Old
fixed-position positives were not training inputs. Backgrounds had already been examined during
qualification; only their roles in this new fit were held out, with no untouched clinical-test claim.

Strict weights-only CPU loading independently regenerated initialization from seed 2026103501 and
verified all six named parameter tensors, both arm initial states, finite changed final states,
and full Adam options/moments with step 400 for every parameter. Each arm saved 400 finite losses.
Original and recovered profile step-two and step-three model and complete optimizer states were
byte-identical, and saved losses matched exactly. Fresh scientific initialization and runtime
optimization remain source-bound claims supported by these immutable states and reviewed code.

The checker verified 8,477 case scalars, 4,715 metric/decision scalars, full map geometry/ranges,
tie-block AP, paired responses, spatial prior and controls. It performed 103 source/input/output
binding checks before and after arithmetic and checked the three fixed figure files. It performed
zero model forwards or optimizer updates. Saved recovery equality is independently established;
the audit did not independently rerun training or inference.

## Accounting and evidence

Main terminal costs were 16.926504622 CPU, 39.979068580 wall and 24.868681350 GPU-lock seconds,
including ten profile updates and two TRAIN inference batches, 800 scientific updates and
48 EVALUATION inference batches. The audit added 3.505593613 CPU and 6.259298472 wall seconds,
with zero GPU cost. Its 180-second/16 MiB limits passed; peak audit RSS was 710,889,472 bytes.
Main bytes were 269,290,978; the audit occupied 36,233 bytes, giving 269,327,211 combined bytes,
below the separate 512/16 MiB allowances.

This main stage plus audit used 20.432098235 CPU seconds. Adding the preceding verified chain once
gives current-machine cumulative CPU 32.489216225 seconds and GPU-lock wall 38.433311437 seconds.
These totals include failed quality-v1 and its metadata audit. Missing historical factorial costs
remain unverified and excluded. Terminal measurements include receipt write/hash before ledger
serialization. No failed attempt was discarded or rerun by the auditor.

Protocol/checker `289e705` was source-reviewed against exact hashes, integrated and admitted for
one copied-script run. All ten integrated artificial tests and Ruff passed. No real ECG, patient ID,
protected dataset, external inference API, generator forward or clinical label change was used.
Arrays, maps, weights and figures stay local. No additional fit follows automatically from this PASS.

Evidence under `outputs/ssl_synthetic_localization/randomized_localization_v1/`:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `fc4bf8934c95456d35cff5cc697b8b8f9b7563d79018fbc940fd68feda72064f` |
| `independent_audit_v1/audit_script.py` | `559ba9dbf5a0911edc5109038b8e6dc436d12f46ee1b7694f0eafd62daf17340` |
| `independent_audit_v1/summary.json` | `b44e847a00a6e41587dab9d1d95097e750d3db7568c3a2968ca899f43e2a17ac` |
| `independent_audit_v1/prospective_ledger.json` | `383ab5707ecfeb28773d0617c7aaa8b8875cc50d77fa17bd82c1db299cf8f1a5` |
