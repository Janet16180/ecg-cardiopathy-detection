# Compact CPC local readout v1: proposed controlled protocol

**Status, 25 September 2026:** the user authorized and the CPU-only study
completed under a new immutable manifest. The prespecified development point
screen passed. See the [results and limits](cpc-local-readout-v1-results.md).
This was the single recommendation from the
[ranked investigation](cpc-improvement-research-2026-09-25.md).

## Hypothesis and comparison

At fixed classifier width, pooled local-token peaks add useful morphology
information beyond pooled GRU context in frozen compact CPC, especially with
1,518 labels. Use only the completed 009 feature cache from the historical
20-epoch, seed-42 **ordinary compact CNN/GRU CPC** encoder. Do not use released
S4 features, a different SSL checkpoint, the 010 partial checkpoint or a
development-selected fine-tuned encoder.

For cached branches `F[record, branch, coordinate]`, freeze exactly two arms:

- **A context control:** `F[:, 0, :]`, meaning 256 context means followed by
  256 context maxima.
- **B local replacement:** concatenate `F[:, 0, :256]` and `F[:, 1, 256:]`,
  meaning the same context means and 256 normalized local-token maxima.

Both have 512 features and 513 trainable logistic-head parameters. Both use
token positions 7 onward and identical half pooling inherited from 009. This
tests a specific feature replacement, not a pure claim about recurrence or a
new end-to-end architecture. No alternative pools, concatenation widths,
template tuning or residual branches are included.

## Patients, fitting and fixed choices

Use the [clean preflight](../outputs/data_quality/clean_rerun_preflight_v1/receipt.json)
to identify exactly 15,359 full-budget labels and the original 1,518-label subset.
Development stays at 1,306 records/1,173 patients. Join ECG ID, patient ID and
partition explicitly; verify unique IDs, nested labels and patient separation.
Historical encoder normalization remains fixed. Fit a separate StandardScaler
and logistic classifier **only on each arm's exposed training labels**.

Fix float64 logistic regression, L2 penalty, `C=0.01`, unweighted classes,
intercept, L-BFGS, `max_iter=5000`, `tol=1e-8`, seed 42, and one CPU/BLAS thread
for all four fits. C is a prespecified design choice, not selected by historical
test scores or a new development search. No epoch selection or hyperparameter
sweep. Stop as numerically unresolved if any fit fails convergence; do not
silently give one arm a larger budget. Repeat saved-parameter logits exactly or
within a documented float64 tolerance before scoring. A convex deterministic
head fit does not create a new independent encoder seed.

## Verification and cost gate before outcomes

Use new source files and a new immutable manifest; preserve all original
receipts. Bind the historical feature/row hashes, encoder and normalization
identities, clean manifests, software versions and new source hashes. The
recorded 009 feature hash is
`dc25c5b69661fe636bc27c4691254b702047c4623d8ade39dedd75d520a9caca`;
the row hash is
`36bd0e83c4d90901be4375a9e1e1e6c1f9e189783e801959d712c4dd929c364a`.
Rehash bytes as an integrity check, then read only the allowlisted train/dev
numeric rows. Reject mismatched shape/dtype, nonfinite selected features,
duplicate IDs or partition disagreement. Do not invoke 009's old all-stage
runner: it contains calibration/test evaluation paths.

Before development outcomes, measure cold hash/load time and a complete
training-only full-budget fit of each arm, saving these fits for reuse if the
gate passes. Account for preflight/profile work and retries. Bound reporting
with a synthetic paired-bootstrap timing measurement; no actual new model
development outcomes are needed for the gate. Require projected total
`elapsed + 1.5 × remaining_work` ≤7,200 seconds, where remaining work includes
both limited-budget fits, development inference, 2,000 paired patient draws,
parameter serialization/replay and final hashes. Reserve at least 120 seconds
for report and write overhead. If the projection or observed cumulative time
exceeds the ceiling, stop and preserve the failure receipt.

Planning estimate is **1–15 CPU minutes**, conditional on verified caches and
solver convergence; it is not a measured promise. The 117.5 MB cache and
~68 MB full train/dev float64 arm matrix suggest <2 GB RSS, including working
copies, but record peak RSS. No GPU or waveform extraction is needed. Any
future GPU successor needs its own complete-path V100 profile and the shared
lock; this CPU proposal does not authorize it.

## Analysis and stopping rule

Primary endpoint: B−A development AUROC at 1,518 labels. Secondary: the same
contrast at 15,359 labels, plus descriptive AP and uncalibrated log loss.
Compute a 2,000-draw paired bootstrap over development patients with seed
250925; retain each patient's ECGs together and report invalid single-class
draws. Keep predictions paired across arms and use the same draws at both
budgets. Do not optimize thresholds, fit calibration or evaluate test data.

The practical **point-estimate screen** passes only if limited-label ΔAUROC
≥+0.002 and full-label ΔAUROC ≥−0.002. Also report the full 95% interval; an
interval spanning zero leaves the positive direction uncertain even if the
point screen passes. A failed screen ends this candidate without a pool/C
sweep. A passed screen justifies discussing an independently frozen replication,
not calibration/test promotion. Report the repeated-development-exposure
limitation and historical pretraining exposure explicitly. The endpoint is an
ECG annotation proxy, not established person-level health or referral utility.
