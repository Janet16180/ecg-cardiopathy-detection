# Saved-map diagnostic v1 admission stop

18 October 2026. The single authorized diagnostic attempt stopped during its parent-integrity
admission, before producing any new control or positive-case statistics. No retry occurred and the
implementation and failure receipt remain unchanged. This is an admission failure, not a completed
scientific diagnostic or a change to the failed parent advancement.

The stop was caused by applying the scientific absolute tolerance of 1e-12 to a rounded CPU-cost
literal. The SHA-anchored parent audit records cumulative CPU as 545.1449704129989 seconds; the
protocol's displayed value was 545.144970413 seconds. Their absolute difference is
1.1368683772161603e-12 seconds, narrowly exceeding that tolerance. The source used the same generic
metadata comparator for scientific reproduction and this accounting check.

Read-only JSON investigation by the running agent and independent reviewer found that all eight
parent summary blocks and the frozen R/S thresholds match. The stop therefore does not identify a
model-metric discrepancy. Future accounting should use the exact SHA-anchored receipt values, with
rounded prose treated as display only; scientific comparisons must retain their original tolerance.
That is a prospective forward-adapter recommendation, not an alteration of this receipt or source.

The frozen run opened only allowed saved probability/mask arrays and metadata during admission.
It produced zero control rows and zero positive rows. Raw/processed waveform payload reads,
checkpoint payload reads, model inference, optimizer updates and GPU use were all zero. No
threshold was changed or new advancement criterion introduced. Original simulator and complete
paired-study advancement remain failed. Native findings remain unknown.

Actual stopped-attempt cost was 1.387159193 wall seconds and 1.386906838 CPU seconds, on one
numerical thread with CUDA hidden. Adding that exact CPU charge once to the parent gives
546.5318772509989 cumulative CPU seconds and 112.398946179 paired-stage CPU seconds. No new
scientific statistic was saved. The failure receipt retains 99 source and 229 input bindings.

The protocol/admission addendum were committed as `5245d2e` and `35882b3`. Source, runner and twelve
artificial CPU tests were committed as `b24c3ce`, then integrated as `a5e21a4`; tests passed in
0.49 seconds and Ruff passed. Independent source review and explicit coordinator execution READY
preceded this one attempt. The pinned source is not edited to fix the accounting issue.

Local immutable failure: `outputs/ssl_synthetic_localization/paired_background_v1/diagnostic_v1/failure.json`.

SHA-256: `3d41af8ff1c592ee1b7f0a1ac89d9744e796aef3b72cee64b71155151565bb67`.

A forward run requires a new committed policy/adapter, new destination, retained prior cost and
fresh source review/integration. No saved-map independent diagnostic audit was launched for this
stopped attempt. Patient arrays, parameters and identifiers remain local and untracked; all work
remains local without remote actions.
