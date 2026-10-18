# Counterfactual audit v1 stopped before diagnostics

18 October 2026. The first counterfactual audit stopped at its original-reproduction admission gate.
No counterfactual diagnostic was derived: verified positive pairs are zero and neither profile
completed. The passed independent original audit had not placed the original evaluation summary's
SHA-256 in its `input_sha256` mapping, although the handoff had specified that schema. Its
`all_passed=True` and zero mismatch count were intact. The new guard correctly rejected that missing
explicit provenance binding; this is a schema-admission failure, not evidence of changed model results.

The [prospective policy](ssl-synthetic-localization-counterfactual.md) was committed as `cd7b1eb`.
Implementation `26a0651`, binding snapshot `d4081d5` and pre-execution wording/CPU clarification
`a6138db` all preceded this attempt. Nine artificial CPU tests and Ruff passed before execution.
The scientific source review found no numerical blocker. The original pilot's failed advancement,
calibration, native maps and independent audit remain unchanged.

The failed receipt is
`outputs/ssl_synthetic_localization/pilot_v1/counterfactual_audit_v1/failure.json`.
It records `ValueError: Original independent reproduction is absent, failed or bound to another
evaluation`, 1.690 wall seconds, 1.689 CPU seconds, source hashes and zero completed diagnostics.
No new inference, checkpoint deserialization, real waveform read or GPU operation occurred.
The failed directory and all four source/policy files it pins are retained without edits or restart.

A forward v2 admission may reuse the unchanged metric/matching helpers after a new independent
binding-closure receipt explicitly connects the passed original reproduction to the exact original
evaluation summary. That successor needs its own policy, implementation/tests, fresh output directory
and scientific review. Diagnostic endpoints and original thresholds must remain unchanged, with this
failed attempt's cost retained. This report contains no counterfactual performance result.
