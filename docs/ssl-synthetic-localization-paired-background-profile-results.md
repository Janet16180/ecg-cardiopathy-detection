# Paired-background actual-path profile results

18 October 2026. The single committed and reviewed resource profile passed. This is an
execution/recovery and resource-admission result, not localization performance or evidence of an SSL
benefit.

The actual TRAIN path regenerated qualified inputs, checked all four frozen input/target hashes,
copied only the edited waveform and support to CUDA, and ran the existing weighted-BCE
backward/AdamW update. R and S each used ten warm-up and twenty timed batch-16 updates from their
respective original initial states with the same original localization head. All parameters were
trainable and the actual-schedule positive weight was 20. Neither optimizer inherited prior state.

| Profile path | Mean timed batch wall seconds | Mean timed batch CPU seconds |
|---|---:|---:|
| R training | 0.037723819 | 0.037668373 |
| S training | 0.037931957 | 0.037900809 |
| Fresh original R, calibration-A inference | 0.025845101 | 0.025892953 |

For each arm, a separate disposable model performed one update, saved model/AdamW/torch RNG state to
disk, then took two further fixed scheduled updates. A fresh model and optimizer restored that
artifact and reproduced both next losses, the complete final model state, optimizer state and step
count three exactly. There were ten recovery update calls plus sixty timing-path update calls:
seventy disposable updates in total. The profile did not create scientific fitted states.

Fresh original R additionally processed ten warm-up and twenty timed calibration-A batches through
the exact public prediction function shared with evaluation. Its state digest was unchanged. All
probability maps were discarded; no threshold, AP, flag rate or B inference was computed. The
profile regenerated 736 unique cases, checked every rendered use, and saved only local recovery
states and hash-only regeneration metadata.

Actual profile wall time was 12.165993124 seconds and CPU time 12.152899710 seconds, including
recovery and input/source verification. Peak allocated CUDA memory was 101,732,352 bytes. One
numerical and PyTorch interop thread were enforced under the shared nonblocking GPU lock.

The frozen forecast adds twice each arm’s mean batch time times 600 updates, twice the measured
inference batch time times 96 batches, the actual profile, and 120 seconds of reporting/evaluation
reserve. CPU additionally includes the 11.330755370-second data preparation charge. Forecast
new-stage cost was 227.915182883 GPU-wall seconds and 239.138119460 CPU seconds, below 900/1,800.
Training preserves 124.971446880 seconds for inference and reporting. Cumulative ceilings remain
7,200 GPU-wall/14,400 CPU seconds.

The five-array plus shared 64 MiB metadata/checkpoint storage forecast was 3,531,478,209
whole-stream bytes, below 4 GiB; actual stream size at profile completion was 3,290,756,370 bytes.
Array accounting counts existing study data once. Cumulative charged cost at profile completion was
148.188247838 GPU-wall seconds and 457.616586152 CPU seconds.

Admission freshly bound the passed 692-background/2,768-case qualification, all 1,384
matched-control baselines, the original initial/SSL checkpoint bytes, and the predecessor’s
independent numerical completion. All 91 source and 192 input bindings were verified again
afterward. The three adjacent output artifacts are hashed in the profile receipt. The original
toy-pilot advancement remains failed. No native clinical finding or physiological ST boundary was
validated.

Implementation `1ba72d4` was committed and integrated before execution. The exact shared inference
path additionally includes the raw-logit finiteness fix `a8da92e` and actual-storage guard
`7f0e2dd`. The combined artificial CPU suite passed 55 tests in 1.15 seconds; Ruff passed.
Independent source review cleared the profile before the coordinator’s PROFILEONLYREADY. Full
fitting required its own subsequent authorization.

Local profile receipt:
`outputs/ssl_synthetic_localization/paired_background_v1/profile/summary.json`.

SHA-256: `07cdf8c7517aa686337b16cb9128102c6e3d1c9954cc1c1edea26f6fc0658d76`.

Qualification receipt SHA-256: `38a9c533be62157d0696c14468da9f4d908338069b54c6fe3bbe2790bdca2f18`.

No failed profile attempt occurred. Recovery tensors, patient arrays and parameters remain local and
untracked. This report does not change the [prospective
study](ssl-synthetic-localization-paired-background-study.md) or its later decision rule.
