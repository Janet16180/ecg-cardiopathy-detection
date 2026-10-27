# Paired-background fixed training results

18 October 2026. Both prospectively fixed real-background localization fits completed once. R and S
each made exactly 600 optimizer updates, with the same ordered 9,600 case presentations, 500
patients and 2,000 unique inputs. This closes the training stage only. Training losses are
optimization accounting, not localization performance, and did not select checkpoints or alter the
recipe. Calibration and engineering evaluation are separate stages owned by the coordinator.

Each batch contained four scheduled TRAIN patients and all four adjacent variants per patient:
focal, repeated, unchanged and added nuisance. Each arm therefore saw 4,800 positive and 4,800
zero-mask presentations. The fixed default_rng(20261024) schedule concatenated independent
500-patient permutations: 400 patients appeared five times and 100 four times. The saved patient and
case schedules are identical across arms.

| Arm | Updates / optimizer step count | Presentations | Unique patients / cases | First weighted BCE | Last weighted BCE | Actual arm wall / CPU seconds |
|---|---:|---:|---:|---:|---:|---:|
| R | 600 / 600 | 9600 | 500 / 2000 | 1.034773350 | 0.562532187 | 22.468905 / 22.500432 |
| S | 600 / 600 | 9600 | 500 / 2000 | 1.032525897 | 0.323518187 | 22.657648 / 22.635865 |

All 1,200 losses were finite. Every final model tensor was finite; both plain CPU model-state
checkpoints reloaded to their exact final model digest. No early stopping, additional seed,
checkpoint search, retry, additional SSL, F arm or new optimizer-state transfer occurred. Both
encoders and heads remained fully trainable, with 45,868 trainable parameters per model.

R started from the original saved initial model. S started from that same state and replaced only
its encoder with the original final SSL encoder. Neither inherited a previously localization-trained
state. The localization head digest was identical initially; fresh independent AdamW optimizers used
learning rate 0.001 and weight decay 0.0001. Existing localization_update applied full-map weighted
BCE with positive weight 20, computed and capped from actual scheduled TRAIN mask cells before
profiling. Fixed model entry retained physical mV inputs divided by the original 2 mV scale, without
fitted normalization.

Only the regenerated observed edited waveform entered the model, with exact support supplied solely
as the target. Paired baseline, candidate windows, nuisance/edit parameters and patient metadata did
not enter network features. Every use rerendered and checked all four frozen array hashes; all 2,000
unique TRAIN cases matched the admitted truth. Zero support denotes absence of the added
perturbation, not native clinical normality.

## State and schedule provenance

| Component | SHA-256 |
|---|---|
| Common initial head | `e4498c214a55ec4875d023813a6d168dabd055cb94123982a28092e0c29bcbb7` |
| R initial encoder | `b65ea644feca87d9e02a749b56f4996845af4416b40c29e08b0dab69595a7cf7` |
| S initial encoder | `607e26ec2e405d3349220f1269003e7b8b874bcd42c4861f71f47abdd61d0e87` |
| Native patient-schedule bytes | `3729fb2311bf99110185f2baef0180d0d1b467d4a7dea205ab4fd997fa3c3d9c` |
| Native case-schedule bytes | `e14fd2aa4939051723cf5f7cf19fcd47b408871f520c5997fb63362ad914010d` |
| R final encoder | `c80ceb6b03cd3036b315daa6e09963a5e914486f25d39bf631848e0d6fbfb41d` |
| R final head | `340a0625c4be78cd0320cb509a2283cc662476f68fb6fbcf14ec6ade82bf16c8` |
| R final full model state | `de6c3013963fa79fbae2797f00c97cf2b349ca0b3590a7a2802941f2331cc14a` |
| R checkpoint file | `b66c6ac6eca74e4e40051ecaf07c4c1656aa9b2fcaaaa1fa1013a0a397d78ad9` |
| S final encoder | `e8da8d68783399b758a747b0499511ef03bc3cd930d29e257f6c42ea892f69a5` |
| S final head | `597f04a5704905e5cf9e1e208c453c87cf0ed3e57339e8aa9ac5223a5648f0d1` |
| S final full model state | `8dc7ca581ea498958ddf83330fc291fdd6833341006f4e5a2965a63f068c6d58` |
| S checkpoint file | `b06b70895bd257b540046bdb4af0f7df188e19b0d469885f3f2a97a386fc26b0` |

The prospective protocol/configuration were committed before selection or data reads. Data
qualification retained every selected background and all cases, with no replacements or generation
adjustment. Original checkpoint bytes, data/source hashes, the predecessor numerical completion and
original failed toy-pilot advancement were freshly bound before fitting. The unchanged source-bound
actual-path profile passed, including exact model/AdamW next-update recovery and the exact shared
calibration-A prediction path. Independent source review and the coordinator’s separate TRAINREADY
preceded this execution.

Implementation commit `1ba72d4` was integrated as `8500b4b`. The shared inference source fixes
`a8da92e` and `7f0e2dd` were integrated before profiling. The combined artificial CPU suite passed
55 tests in 1.15 seconds and Ruff passed. These are implementation checks; they do not replace the
separate engineering evaluation.

## Costs and integrity

| Cost | GPU-lock wall seconds | CPU seconds |
|---|---:|---:|
| Fixed R/S training stage, including verification/checkpoint work | 55.856909117 | 55.853009972 |
| New study through training, including preparation and profile | 68.022902241 | 79.336665052 |
| Cumulative charged history through training | 204.045156955 | 513.469596124 |

The first row includes admission, actual regenerated-input updates, state/checkpoint checks and
final source/input verification. The individual arm times above are not substituted for total stage
accounting. Training preserved 124.971446880 seconds for evaluation/reporting and stayed below all
live stage and cumulative limits. All numerical/PyTorch interop threads were one. Peak allocated
CUDA memory was 100,419,584 bytes, and whole-stream storage at completion was 3,292,010,774 bytes,
below 4 GiB.

All 91 source and 196 input bindings were verified again at completion; seven adjacent output files
are hashed in the training receipt. The unused original real-TRAIN cache payload remained delegated
to its frozen metadata digest rather than reopened; the newly qualified 500-patient training cache
was the authorized optimizer input. No protected real engineering-test payload, closed test set or
new cohort was read. No profile or fit failure occurred in this new stage; retained predecessor
failure/audit history remains unchanged.

Local training receipt:
`outputs/ssl_synthetic_localization/paired_background_v1/training/summary.json`.

SHA-256: `1896e9c638f27590d4a90c582536926dfbc85ffe57db1f08da12f7e3da8bf44c`.

The receipt binds R/S checkpoints, native patient/case schedules, aggregate progress, admission and
hash-only regeneration checks. Patient IDs, observed arrays, masks, weights and per-case parameters
stay local and untracked. No remote action occurred.

Actual profile receipt SHA-256: `07cdf8c7517aa686337b16cb9128102c6e3d1c9954cc1c1edea26f6fc0658d76`;
see the [profile results](ssl-synthetic-localization-paired-background-profile-results.md). The
[prospective study](ssl-synthetic-localization-paired-background-study.md) freezes the later primary
response comparison and all advancement gates. No training-loss difference establishes an SSL
benefit, native pathology localization or screening utility.
