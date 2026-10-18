# Counterfactual-loss factorial actual-path profile results

18 October 2026. The single four-arm actual-path profile passed before scientific fitting. This
establishes resource and recovery admission, not localization performance or benefit of the paired
objective.

RB/SB use unchanged weighted BCE, with original random/SSL encoder initialization. RP/SP use the
same respective states plus the prospectively fixed per-cell probability hinge margin 0.20 and
outside-support consistency, weighted 1.0. All have the original common fresh localization head and
one observed input per forward pass. The exact same fixed TRAIN batches and native class weight 20
were used.

| Arm | Mean training batch wall seconds | Mean training batch CPU seconds | Mean calibration-A inference wall seconds | Mean inference CPU seconds |
|---|---:|---:|---:|---:|
| RB | 0.038580112 | 0.038480944 | 0.026592032 | 0.026605057 |
| SB | 0.038470562 | 0.038380967 | 0.026559800 | 0.026576401 |
| RP | 0.039230919 | 0.039130661 | 0.027352913 | 0.027369586 |
| SP | 0.039461842 | 0.039371886 | 0.027084717 | 0.027106861 |

For each arm, ten warm-up and twenty timed training batches exercised actual regeneration, all byte
checks, transfers, objective, backward and fresh AdamW. A separate one-update model/AdamW/RNG disk
save followed by two direct and restored next updates reproduced every loss component, complete
model tensor state and optimizer state exactly, at optimizer step three. Repeated control indices in
the paired probability gather also passed deterministic CUDA backward. Each arm then used ten
warm-up and twenty timed actual calibration-A inference batches; model states were unchanged.

There were 140 disposable update calls (120 timing-path and 20 recovery) plus 120 calibration
inference batches. Probe probabilities were discarded. No B inference, threshold, localization
metric or scientific fitted checkpoint was produced. The unique rendered input set was 736 cases;
the class-weight calculation additionally verified all 2,000 TRAIN support rows. No TRAIN cache was
duplicated.

Actual profile wall/CPU was 11.100679235/11.076360575 seconds, including recovery and verification.
The prospective forecast counts twice 600 times each of the four measured training means and twice
48 times each of the four inference means: 192 scientific inference batches in total, then 120
seconds reporting/evaluation reserve. It adds actual profile and, for CPU, 2.988186446 seconds
preparation. Forecast stage cost was 328.321390369 GPU-wall and 330.837056492 CPU seconds, below
900/1,800. Scientific training preserves 130.335158851 seconds for evaluation/reporting.

Peak allocated CUDA memory was 103,091,712 bytes. Actual stream storage at completion was
3,553,658,222 bytes. The prospective complete new-study allowance is 504,869,760 bytes: 437,760,000
array payload, seven 128-byte headers and a shared 64 MiB metadata/checkpoint reserve. Projected
complete-stream size before preparation was 3,984,290,138 bytes, below 4 GiB, with no cleanup of old
artifacts. All numerical and interop threads were one, under the shared nonblocking GPU lock.

The profile freshly bound all 126 sources and 28 inputs; five outputs contain four recovery states
and hash-only regeneration checks. Three whole predecessor array payloads were authority-deferred;
only actually needed TRAIN row/mask bytes were read and checked on access. Original initial/SSL and
required predecessor checkpoint bytes were verified. No old validation waveform rows entered
inference or optimization.

Code was prospectively frozen as `462eb09`, integrated as `cfd50db`; final shared source hardening
`e44cf34` was integrated as `e9c2d82`. The combined artificial suite passed 38 tests in 1.23 seconds
and Ruff passed. Independent source review and explicit coordinator PROFILE GO preceded this one
execution from the coordinator worktree.

Local receipt:
`outputs/ssl_synthetic_localization/counterfactual_factorial_v1/profile/summary.json`.

SHA-256: `4007840a2363359567627025f37143c01119ce2e1668d99b154f9297907fd87f`.

The new-validation qualification receipt is
`498251dc05013ece766d3877755fe0a000db5b8a1db8dc2ffdcddad4bbefcdba`: all 192 backgrounds/768 cases
passed, with 768 exact replays, 384 matched baselines and zero failures. Native findings remain
unknown, and controlled pulses are not expert clinical annotations. The [prospective
factorial](ssl-synthetic-localization-counterfactual-factorial.md) and all earlier failed decisions
are unchanged.

