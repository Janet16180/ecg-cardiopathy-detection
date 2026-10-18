# Matched SSL pilot training execution

18 October 2026. All frozen fits completed; this report contains optimization/accounting evidence only,
not held-out localization performance. Evaluation is a separate source-bound stage.

The original [protocol](ssl-synthetic-localization-pilot.md), [bounded generator amendment](ssl-synthetic-localization-pilot-generator-v2.md)
and [deterministic output amendment](ssl-synthetic-localization-pilot-deterministic-output.md) fix the
experiment. The successful [GPU profile](ssl-synthetic-localization-profile-v2-results.md) admitted the
actual batch-16 data path. Profile weights were discarded. The full-run implementation is `bb9b847`;
subsequent queue/report commits change no executed scientific source.

| Phase | Updates | Record presentations | Independent labeled subjects | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| Real masked SSL | 1,000 | 16,000 | No localization labels | 27.195 |
| R | 600 | 9,600 | 500 | 15.865 |
| S | 600 | 9,600 | 500 | 15.885 |
| F | 600 | 9,600 | 500 | 8.868 |

All 2,048 selected real training patients were cached in physical mV. The stochastic SSL schedule
actually presented 2046 distinct cache rows. The fixed synthetic
training bank contains 500 independent latent subjects/examples (375 positive edits and 125 controls);
presentation counts do not create additional independent labels. No real validation or protected
waveform entered optimization. No disease annotation was used in SSL.

The exact initialization checks pass: R starts from the fresh encoder used before SSL; S and F start
from the same final SSL encoder; all localization heads start from identical fresh bytes; F's final
encoder hash is identical to its initial hash. Every optimizer is fresh for its objective/arm, and F's
optimizer contains only its head. Localization batches are identical under seed 20261020, and the
training-only positive class weight is 20.00000000. Fixed final checkpoints are
used; no validation or held-out model score selected a checkpoint.

The full fitting stage consumed 68.472596 seconds wall and 68.392428 seconds
process CPU. Peak allocated CUDA memory was 103,609,856 bytes. The pretraining,
R, S and F losses are retained locally; they are optimization diagnostics, not disease accuracy or
cross-profile evidence. Shared GPU lock, one numerical/interop thread, deterministic algorithms,
per-update cumulative deadlines and a reporting/evaluation reserve were enforced.

Local evidence: `outputs/ssl_synthetic_localization/pilot_v1/training/summary.json`, SHA-256
`9c855f74e98ae5f1706ec9baf3fe46b1f104f1e2c24b936b011b96cb00f4b265`. It binds initialization/final component hashes, every checkpoint, complete update counts/losses,
the matched schedule, source files, dependencies and actual cache/bank payload hashes. A post-fit
check verified all recorded sources, inputs and outputs and every matching/frozen-state assertion.
`admission.json` records the before-fit bindings; failures would remain in `failure.json`. No failure
receipt was produced. Weights and per-subject artifacts remain local.

The next stage calibrates strict maximum-map thresholds on A controls before held-out profile B
inference. Training success alone supplies no localization or clinical performance result. Existing
historical experiment scores remain unreproduced motivation, not numerical comparators.
