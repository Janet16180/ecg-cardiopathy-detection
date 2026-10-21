# Counterfactual audit v2: explicit reproduction binding

18 October 2026, before deriving any counterfactual diagnostic. This forward version follows the
[retained v1 schema-admission failure](ssl-synthetic-localization-counterfactual-results.md).
No metric, threshold, subject, profile, counterpart rule, region, bootstrap or interpretation changes.
The complete [original counterfactual protocol](ssl-synthetic-localization-counterfactual.md) still
specifies every diagnostic. The original pilot advancement remains failed and unchanged.

The new independent `final_audit_v1/reproduction_binding_closure.json` must pass with zero mismatches,
explicitly bind the exact original `evaluation/summary.json` in `input_sha256`, and bind the passed
original independent audit `final_audit_v1/summary.json`. This closure freshly verifies prior bindings
and matches previously independently recomputed original aggregates/thresholds/decision. It is an
additive provenance artifact; it does not edit the original audit or rerun model inference.
The gate must also require the current original failed advancement and retain the original
counterfactual-v1 failure receipt as an input. Verify every declared source/input binding before and
after this audit, retaining the explicit no-read boundary for the unused real-training waveform cache.

A minimal new orchestrator imports the unchanged v1 admission, complete matching, saved-map metric
and bootstrap helpers. It does not monkeypatch their globals or edit any receipt-pinned v1 file.
New outputs go to `pilot_v1/counterfactual_audit_v2/`, refusing overwrite; failures are retained there.
Commit this amendment, then commit the forward implementation/tests, and obtain source review before
executing once after the closure is ready. Preserve the 1.689-second stopped attempt and all measured
closure/supplement costs in cumulative CPU accounting. Keep one numerical CPU thread, CUDA hidden,
no model checkpoint deserialization/inference/GPU use, and at most 100 MiB of new output.

Profile B remains spent development evidence. These diagnostics describe response at programmed sites,
including baseline-site agreement and nuisance responses; they do not supply fresh confirmation,
clinical causality, disease validity or a new advancement rule.
