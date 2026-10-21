# Original real-SSL and synthetic localization pilot: results

18 October 2026. The frozen pilot did not show the required localization benefit from real-ECG
self-supervised pretraining. On held-out edit profile B, mean subject-macro AP was 0.64060 with
pretraining and 0.63796 from scratch: a paired gain of 0.00264, 95% interval [-0.00355,0.00833].
This misses both the required 0.05 gain and positive lower bound. Both control-rate gates passed.
The complete conjunctive engineering advancement rule therefore failed; no clinical promotion follows.
No tuning, checkpoint replacement, seed retry or evaluation restart followed these results.

## What ran and what the comparison means

The original [pilot protocol](ssl-synthetic-localization-pilot.md) was frozen before scores. The
[bounded-generator amendment](ssl-synthetic-localization-pilot-generator-v2.md), committed as `847ac1e`,
and [deterministic-output amendment](ssl-synthetic-localization-pilot-deterministic-output.md), committed
as `831f7bf`, preceded full training and localization scoring. They retain the original label budget,
subjects, seeds, native grid, objectives, batch size, exposure and advancement rule.

The real cache contains 2,048 technically admitted PTB-XL training patients selected without diagnosis
labels. SSL's 16,000 random presentations actually included 2,046 of those cache rows. No real
engineering-test waveform, source fold 9–10 waveform, reserved Challenge/EchoNext test or clinical
annotation was opened by this evaluation. Only the simulated observed signal enters inference;
paired baselines, masks, candidate windows, latent landmarks and generator parameters are truth or
diagnostic artifacts, never extra network channels.

Training uses 500 independent simulated latent subjects/examples: 375 positives and 125 zero-target
controls. Validation uses 128 separate A subjects with four variants each. Evaluation uses all 256
separate subjects with focal, persistent, sham and nuisance variants, rendered by both B and secondary
A: 1,024 recordings per profile, including 512 positives. There were no dropped, replaced or
unscorable cases. Every variant of a latent subject stays in one role. B changes the edit profile
while retaining the same background simulator and timing distribution; this is held-out edit-profile
transfer, not independent clinical-domain validation.

| Arm | Initialization and localization adaptation | Trainable parameters |
| --- | --- | ---: |
| R | Original random encoder and common fresh head; both adapted | 45,868 |
| S | Final real-SSL encoder and same fresh head; both adapted | 45,868 |
| F | Same real-SSL encoder held fixed; same fresh head adapted | 396 |

SSL ran exactly 1,000 updates, batch 16. Each localization arm ran exactly 600 updates and 9,600
presentations with the identical ordered 12-positive/2-sham/2-nuisance schedule, presenting all 500
training subjects. Fresh optimizers were used. Exact state hashes establish the common initial head,
R's original pre-SSL encoder, S/F's same final SSL encoder and F's unchanged final encoder. F has the
same 45,868 total parameters but a lower adaptation capacity; it is a diagnostic rather than a
matched-capacity competitor.

## Primary held-out-profile localization

AP is computed on the complete 12-by-2,500 grid. Focal and persistent AP are averaged equally within
each latent subject before the primary mean and paired bootstrap. The 2,000 shared whole-subject
draws use seed 20261019, with no skipped draws. The intervals describe background-subject variation
conditional on one training seed; they do not measure variation across independently trained models.

| Method | B subject-macro AP | Focal AP | Persistent AP |
| --- | ---: | ---: | ---: |
| R: from scratch | 0.63796 | 0.62974 | 0.64618 |
| S: real SSL, adapted | 0.64060 | 0.62921 | 0.65199 |
| F: real SSL, frozen | 0.15787 | 0.05383 | 0.26192 |
| Uniform / mask prevalence | 0.03867 | 0.00604 | 0.07130 |
| Raw absolute voltage | 0.06954 | 0.01324 | 0.12585 |
| Oracle broad phase prior | 0.09968 | 0.01643 | 0.18293 |

The phase prior highlights every broad candidate ST window on every beat and all leads, without
knowing which beat, lead or width was edited. It has privileged simulated landmarks and is not a
deployable competitor. Both adapted models outperform these contextual maps on this engineering
intervention; that observation does not establish real pathological evidence localization.

| B S-minus-R comparison | AP difference | Paired 95% interval |
| --- | ---: | ---: |
| Primary subject-macro | +0.00264 | [-0.00355,+0.00833] |
| Focal secondary | -0.00053 | [-0.01015,+0.00790] |
| Persistent secondary | +0.00581 | [+0.00016,+0.01080] |

The small persistent secondary gain does not rescue the failed primary comparison. The primary
point estimate is about 0.26 AP percentage points, versus the prospectively required five points.
These are engineering intervention labels, not clinically verified ST disease boundaries.

Fixed-area results credit exactly 300 cells, the top 1% of the full map, with fractional expected
inclusion at tied cutoffs. B macro precision/recall are 0.58938/0.38668 for R, 0.58847/0.38757 for S
and 0.21689/0.05760 for F. For S, focal precision/recall are 0.46193/0.68466, whereas persistent
precision/recall are 0.71500/0.09048. A concentrated highlight captures only about 9% of persistent
support despite relatively high precision; AP or a correct highlighted point does not establish
complete support recovery. The encoder has 20 ms latent spacing and interpolates to the 4 ms scoring
grid; this does not create independent 4 ms model resolution.

## Calibration and recording flags

For each arm, validation calibration takes the larger sham/nuisance map maximum within each of the
128 A subjects, then its empirical 95th percentile with `method="higher"`. Recording flags use
strictly greater-than; threshold ties are negative. The final fixed checkpoints were used without
selection. Thresholds R=0.8796171546, S=0.9392395020 and F=0.7021850944 were written to a source-bound
calibration file and hashed before any B rendering or inference. That hash remained unchanged.

| B arm | Sham flags / 256 | Nuisance flags / 256 | Positive flags / 512 |
| --- | ---: | ---: | ---: |
| R | 2 (0.78%) | 19 (7.42%) | 474 (92.58%) |
| S | 1 (0.39%) | 23 (8.98%) | 455 (88.87%) |
| F | 0 (0.00%) | 17 (6.64%) | 19 (3.71%) |

Focal/persistent positive flag rates are 89.84%/95.31% for R, 86.72%/91.02% for S and 3.52%/3.91%
for F. Thus F's low sham rate accompanies very few positive flags; it cannot be read as useful
screening sensitivity. All of these flags concern generated intervention controls, not patients.

Paired S-minus-R sham flag-rate change is -0.00391, 95% interval [-0.01172,0.00000]. Nuisance change
is +0.01563, interval [-0.02344,+0.05469]. Both point changes meet the frozen maximum +0.05 veto.
The nuisance upper interval exceeds +0.05; the committed rule uses the point increase, and was not
changed to an interval criterion after seeing this result. The absolute nuisance rates remain
visible regardless of whether that comparative veto passes.

| Frozen advancement gate | Result |
| --- | --- |
| B subject-macro S-minus-R AP at least 0.05 | Fail: 0.00264 |
| Paired primary AP lower bound above zero | Fail: -0.00355 |
| Sham flag-rate increase at most 0.05 | Pass: -0.00391 |
| Nuisance flag-rate increase at most 0.05 | Pass: +0.01563 |
| All gates together | Fail |

## Same-profile secondary and retained preparation failures

On secondary A, R/S/F subject-macro AP is 0.64904/0.65651/0.17533. S-minus-R is +0.00747,
95% interval [+0.00254,+0.01238], still below the required primary gain and not a substitute for B.
A focal AP is 0.64353/0.64927/0.07114 and persistent AP is 0.65456/0.66376/0.27953 for R/S/F.
A uniform/raw-voltage/phase-prior AP is 0.04058/0.09419/0.10465. A positive flags are 481/512 for R,
469/512 for S and 30/512 for F. Its zero-target controls are identical to B controls and give the same
sham/nuisance flag counts. Complete A/B mode metrics, fixed-area scores and flag denominators remain
in the aggregate receipt and local per-case tables.

The earlier [v1 generator design grid](ssl-synthetic-localization-generator-results.md) passed, but
its independently seeded [pilot admission](ssl-synthetic-localization-generator-pilot-v1-results.md)
failed 28 of 3,060 examples on the unchanged occupancy limit, reaching 22.96%. The prospectively
specified forward v2 narrowed duration and coupled lead extent, then [passed all 3,060 fixed cases](ssl-synthetic-localization-generator-pilot-v2-results.md) without replacements. V1 was not trained as
a comparator. Support remains the actual postprocessing paired difference strictly above 0.005 mV,
including induced limb changes and retaining the reported subfloor/tail limitations.

The [first actual GPU profile](ssl-synthetic-localization-profile-results.md) failed because the
original linear-interpolation backward operation lacked the required deterministic CUDA implementation.
It stopped before timed training phases or any full fit, after 0.945 seconds wall/0.887 seconds CPU.
The prospectively amended factor-five fixed transposed-convolution output implements the same
real-valued interpolation without new trainable parameters or inputs; the numerical summation order
may differ. Its actual successor profile passed, taking 5.531 seconds wall/5.496 seconds CPU. Both
attempts and the generator admission failure are retained rather than presented as successful old runs.
Experiments 058/064 motivated the work; their missing local artifacts were not reproduced and their
scores are not numerically compared with this new architecture/training/generator combination.

## Measured execution and reproducible artifacts

The evaluation implementation was committed as `49ea2e6` before model scores and independently reviewed.
The execution checkout was `3dd8473b877461b0c8c52c8562ae68767e4d114d`, after synchronizing the coordinator's
committed training/profile successor. One reviewed frozen evaluation ran to completion with no failure
receipt or restart. It retained 7,680 native model probability maps: 512 validation recordings and
1,024 recordings per A/B profile, each scored by all three arms. Full-map metrics cover 6,144 positive
method maps, including the three contextual methods. Every case and denominator is retained.

| Executed phase | Wall seconds | CPU seconds |
| --- | ---: | ---: |
| SSL, 1,000 updates | 27.195 | 27.148 |
| R localization, 600 updates | 15.865 | 15.828 |
| S localization, 600 updates | 15.885 | 15.877 |
| F localization, 600 updates | 8.868 | 8.887 |
| Complete training, including setup/checkpoints | 68.473 | 68.392 |
| Complete evaluation, including generation/metrics/hashing | 34.474 | 34.217 |

The receipted profile-plus-training-plus-evaluation wall charge is 109.424 seconds against 7,200;
the preparation/profile/training/evaluation CPU charge is 337.067 against 14,400. Evaluation projected
all nine float32 probability files and fourteen bank arrays from their exact native shapes/dtypes/NPY
headers, with a separate 64 MiB JSON/report reserve, before scoring. Its recorded entire-stream size
is 2,769,515,545 bytes (about 2.58 GiB), below the 4 GiB ceiling. This includes the retained original
qualification examples, real-training cache, synthetic banks and probability maps. The actual bank/map
NPY files total 1,536,166,784 bytes; weights and examples remain local.

Training peak allocated CUDA memory was 103,609,856 bytes; evaluation peak was 45,752,832 bytes.
Both used the shared nonblocking GPU lock, deterministic float32 operations and one numerical/inter-op
CPU thread on the NVIDIA GeForce GTX 1060 3GB. The evaluation receipt records Python 3.11.13,
NumPy 2.4.6, scikit-learn 1.9.1, PyTorch 2.6.0+cu124 and CUDA 12.4. No dependencies changed.
Thirteen artificial-only CPU evaluation tests passed in 0.58 seconds; new evaluation files pass Ruff.
They cover signal-only inference, strict calibration, chronology, full-map/tied-area scoring,
partial-training/source rejection, state hashes, storage projection and retained failures.

The preparation [independent audit](../outputs/ssl_synthetic_localization/pilot_v1/prep_audit_v1/summary.json)
replayed all 1,012 train/validation cases and their qualified array hashes, with zero support or
metadata mismatches; it also reconstructed and tested intervention-independent candidate priors.
For completed evaluation, all 50 source bindings and 36 input bindings were checked before and after
execution. The one unused real-training waveform cache digest was compared with its frozen metadata
receipt, explicitly without reopening that real payload; all other inputs were freshly hashed.
All 29 output artifact hashes were freshly verified again after completion. The native probability
maps, A/B banks and local per-case metric JSON permit independent recomputation without new inference.

Key local receipts, relative to `outputs/ssl_synthetic_localization/pilot_v1/`:

| Receipt | SHA-256 |
| --- | --- |
| `training/summary.json` | `9c855f74e98ae5f1706ec9baf3fe46b1f104f1e2c24b936b011b96cb00f4b265` |
| `evaluation/calibration.json` | `c7a915fa644e40d912537af8f9a160a44420e5b87f59a014563a9b3c4aa20b96` |
| `evaluation/summary.json` | `118b0a84324c6ba2b7a77a6761c79abca3c9d4b5e743673d2ec452a33d762aaf` |

## Limits and consequence

This single-seed engineering pilot does not meet its own advancement rule. It provides no qualifying
benefit for this particular real-SSL initialization over matched training from scratch, while retaining
similar absolute localization performance for both adapted arms. It does not show that all real-ECG
self-supervision is ineffective. The frozen-encoder diagnostic is weak under this recipe and budget;
its smaller adaptation capacity prevents a general representation-quality conclusion.

The simulator, lead/morphology ranges, beat timing and editing rules define a narrow engineering
domain. Held-out B shares the background generator, and candidate priors have privileged landmarks.
Physical-floor masks represent added interventions, not complete expert pathology or clinical ST
boundaries. Neither synthetic AP, low sham flags nor this subject bootstrap establishes disease
validity, student referral utility, real-data localization or independent clinical confirmation.
The failed advancement is retained; any successor must prospectively define its question, recipe,
resources and evaluation rather than selecting a favorable secondary result or retuning on B.
