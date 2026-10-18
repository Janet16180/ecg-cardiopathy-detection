# Existing-map counterfactual-response audit

18 October 2026. Commit this protocol before deriving these new diagnostics. The original
[pilot result](ssl-synthetic-localization-pilot-results.md) is closed: its advancement rule failed.
This follow-up uses only the existing saved synthetic banks and R/S/F native probability maps.
It performs no training, new model inference, GPU work, threshold fitting or generator adjustment.
Profile B is now development evidence, not a fresh confirmation set. All work remains local.

## Admission and matching

The final independent original-metric reproduction audit must pass before execution, and its receipt
must bind the exact original evaluation summary. The new runner binds that audit, the original
training/evaluation/calibration receipts, all used source bytes and all saved map/bank artifacts.
Sources, checkpoint/inference-input provenance and saved synthetic artifacts are freshly checked
before and after the audit, except the unused real-training waveform payload: its declared digest
is compared with the frozen cache receipt without opening that payload, as in the original evaluation.
No real engineering-test/excluded waveform, model checkpoint deserialization or new inference occurs.

Keep all 256 subjects, both positive modes and every R/S/F map for B (primary descriptive domain)
and A (secondary): 1,024 positive cases and 3,072 arm/case comparisons. Before any new diagnostic,
verify every positive baseline is byte-identical to a saved observed control from the same latent
seed, subject, profile and background parameters. Match the sham control when that positive has
nuisance=False and the nuisance control when True. Require exactly one matching control and exact
source metadata. Its existing same-arm map supplies p_baseline; p_edited is the positive-case map.
Reject a missing/mismatched pair, invalid/nonfinite map, incomplete subject, source/hash change or
empty region. Retain a source-bound failure rather than excluding cases, substituting controls or
rerunning inference. Verify the complete pairing set before deriving any diagnostic metric.

## Fixed diagnostics

For each positive case and arm, define disjoint regions on the original native grid:

- T: its exact intervention support, abs(actual edited-baseline)>0.005 mV;
- U: the original broad all-beat/all-lead candidate domain excluding T;
- O: outside that candidate domain.

Require T nonempty and contained in the candidate domain, and U/O nonempty. Report the counts of
all three regions for every case. Let delta=p_edited-p_baseline in float64. Compute mean(delta) in
T, U and O, and the descriptive selectivity contrast mean(delta_T)-mean(delta_U). A positive contrast
denotes larger average response at the intervention than other candidate cells; it has no pass cutoff
and does not establish disease causality or complete evidence localization.

Use the unchanged `positive_map_metrics` on the SAME fixed T for both maps. Report edited AP,
baseline-map AP at those future intervention sites, and edited-minus-baseline AP. Baseline-site AP
is an agreement/phase-prior diagnostic, not pathology accuracy in a recording without the intervention.
Also report edited/baseline top-1%-area precision/recall and their changes, using the original 300-cell
expected-tie handling. No candidate restriction, mask expansion or new score transformation is allowed.

For each arm, average focal and persistent values equally within each subject, then summarize over
all 256 subjects. Give whole-subject percentile 95% intervals using the shared `patient_groups`/
`patient_resample` helpers, 2,000 shared draws, seed 20261019, NumPy linear percentiles. Report focal
and persistent separately. Compute S-minus-R contrasts in AP change and probability-response/selectivity
changes with the same paired subject draws, explicitly exploratory. Recording thresholds are fitted separately across arms; probability values themselves are not
probability-calibrated. Absolute probability differences/differences-of-differences do not imply a
common clinical effect scale. F is a lower-adaptation-capacity diagnostic. Invent no advancement rule.

For each profile, subject and arm, also compare the existing nuisance and sham maps: report full-map
mean(abs(p_nuisance-p_sham)), both original recording flags and their difference. Use the ORIGINAL
arm threshold with strict greater-than; do not refit. Summarize the 256 paired latent subjects and
bootstrap as above. These nuisance inputs combine the original noise, drift and gain stream; the
response is not attributed to one of those components separately.

## Execution and interpretation

Commit new implementation/tests before execution. Use one CPU numerical thread, CUDA hidden,
and no GPU lock/model API. Save only aggregate results, local per-case numeric diagnostics and source/
artifact hashes under a new `outputs/ssl_synthetic_localization/pilot_v1/counterfactual_audit_v1/`.
Refuse overwrite. Retain measured CPU cost and enforce the original 14,400-second cumulative
protocol CPU ceiling, including the completed pilot and independent reproduction audit. Limit new output to 100 MiB; do not duplicate probability maps or waveform arrays.
Keep the original evaluation, calibration and advancement verdict unchanged. No patient data,
weights or individual predictions leave the machine. Write results from all live outputs, including
any failure. Diagnosis validity, clinical pathology support, real-data transfer and fresh confirmation
remain unestablished regardless of these exploratory diagnostics.
