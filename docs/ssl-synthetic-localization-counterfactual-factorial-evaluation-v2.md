# Forward scoring correction before factorial evaluation

18 October 2026. The independent auditor found an argument-order error in the committed v1
`case_metrics` caller before ANY factorial calibration, evaluation inference or performance scoring.
The helper contract is `response_metrics(truth, prior, edited, baseline)`; the caller instead passed
`(scores, baseline, target, candidate)`. Existing artificial tests did not exercise this real-helper
integration. This is an implementation error, not an observed model result. The training objective,
training data, resource profile and model updates do not call this scoring helper and are unaffected.

Preserve the v1 evaluation module, tests, profile and training source bindings unchanged. Add a NEW
`ssl_synthetic_counterfactual_factorial_evaluation_v2.py` module, report runner and artificial tests.
Its explicit case-metric code calls `response_metrics(target, candidate, scores, baseline)`. It may
reuse unchanged v1 pure prediction, inference, calibration, model admission, aggregation and decision
helpers. It must use its own explicit orchestrator and destination, never monkeypatch imported
globals or edit a source already pinned by the actual profile. New output destination is
`outputs/ssl_synthetic_localization/counterfactual_factorial_v1/evaluation_v2`; the original
`evaluation` directory has not been created and must remain absent.

All four model states, new 64/128 calibration/evaluation patients, A/B profiles, primary direct-AP
contrast, secondary comparisons, twelve gates, bootstrap seed/draws, calibration-before-B order,
exact BCE closure and resource/storage limits remain those frozen in the original factorial
protocol. The four prediction banks occupy their original planned space; do not allocate a second
set. Bind unchanged v1, actual profile, final training, the original protocol/config and this policy
in the new receipt. Record that no v1 evaluation ran and that no performance score informed this fix.

Require a meaningful artificial integration test using the REAL scoring helper with distinct
boolean support/candidate masks and float32 probability maps. Check manual AP, target/unedited
candidate/outside response and selectivity, and exercise full new case_metrics index/pairing and
four-arm row accounting. This test must fail under the old argument order. Freeze, root-integrate
and independently review all new source/tests before the one evaluation execution.

Independent scientific review confirms no new GPU profile is needed for this CPU scoring-only
correction, provided the exact already-profiled v1 predict_batch/_infer/model path is reused. If the
GPU input/model/inference path changes, stop and obtain a new actual profile before evaluation.
No optimizer rerun, checkpoint selection, patient replacement, threshold tuning or successor fit is
introduced. Existing earlier failed advancement decisions remain failed. An executed results report
must disclose the correction and the independently verified outcome. All work remains local without
pushes, uploads or closed-test access.
