# Real ECG self supervision and synthetic localization supervision

18 October 2026. This research stream tests whether representations learned from real ECGs improve
localization learned from a small set of controlled synthetic examples. The user authorized planning and
implementation in this session while another session investigates No Time to Train in a separate PR.
The first deliverable is a reproducible readiness check. Training follows a committed executable protocol
and a real GPU profile; the plan itself is not a performance result.

## Research question and claims

Does self-supervised pretraining on real ECGs improve localization of a specified waveform alteration,
compared with the same network trained from scratch on the same synthetic examples?

The network outputs a score per lead and time interval. The initial target is a controlled ST waveform
alteration; QRS morphology is a separately reported transfer target once its generator is qualified.
These are engineering intervention labels. A changed waveform region is not automatically a verified
disease manifestation, and a diagnosis does not automatically identify its complete supporting region.
No new clinical label definition or student-screening operating point is introduced.

Keep three evidence types separate:

| Evidence | What it can establish |
| --- | --- |
| Exact counterfactual difference between baseline and edited signal | Location of the generated voltage alteration |
| Published expert beat, boundary or episode annotation | Performance on that annotation's specific task |
| ECGAD abnormal-region mask | Agreement with supplied masks, with annotation uncertainty retained |

The existing ECGAD holdout is spent. It cannot confirm a new method. Earlier public INCART, QTDB, LUDB
and European ST-T evaluations also informed the project, so rereading them is development evidence.
A new confirmation study requires an unused compatible source or annotation subset identified before
method development. If none is available, report an exploratory transfer study rather than inventing
an independent clinical validation.

## Separate development and shared resources

- Branch: `research/ssl-synthetic-localization`, based on `main`; every PR targets `main`.
- Worktree: `/tmp/ecg-wt-ssl-synthetic`; data, outputs, environment and third-party assets link to the main
  checkout. Set `PYTHONPATH` to this worktree whenever running project code.
- Own new files prefixed `ssl_synthetic_` and this plan. The other session owns its No Time to Train
  implementation. Shared queue/backlog edits must be reconciled before PR handover.
- PR #61 is prior evidence, not a branch dependency. Read sources at pinned commit
  `8902c1306d090e6fd8d338a47bab5d45965ef52d`. Reuse shared helpers after they are available on `main`, or
  bring a minimal, explicitly attributed dependency into this PR; do not silently copy a second framework.
- Run at most two CPU numerical threads during parallel work. GPU work must acquire
  `ecg_experiment.gpu.gpu_lock`; do not stop another session's processes.
- No patient records, raw plots, weights or individual predictions leave the machine. No dependency or
  pinned source changes are part of setup. Keep existing Challenge and EchoNext closed sets closed.

## Stage 0 Readiness and benchmark audit

1. Record exact source revisions and local prerequisite availability. Inspect paths only; the readiness
   command must not open waveform or label payloads, install packages, download data or use the GPU.
2. Restore the pinned Python 3.11 environment after checking `uv sync --locked --inexact --dry-run`.
   Check disk headroom before accepting installation or caches; avoid duplicating waveform archives.
3. Build a new patient partition from permitted training data, initially PTB-XL folds 1-8, and commit
   its policy before selecting any model. Reserve an internal validation patient group before SSL or
   generation. Every edited version, baseline and repeat ECG of a patient stays in one partition.
4. Where data have no patient IDs, group exact duplicates and source families and disclose the remaining
   patient-overlap uncertainty. Exclude all evaluation records from SSL even when labels are unused.
5. Audit ECGAD development only when its local arrays and match table are restored: record matching,
   lead order, sample rate, time alignment, preprocessing and mask geometry. Preselect 20 development
   records by seeded sampling before looking at model wins or losses. Retain all audit failures.
6. Describe annotation completeness and agreement documentation. Use development-only boundary
   perturbations to quantify score/ranking sensitivity. Do not repair clinical labels by engineer opinion
   or by model predictions, and do not use a second model as an independent clinician.
7. Reproduce predecessor numbers exactly before comparing against a frozen predecessor. Existing reports
   or hashes without their artifacts do not count as reproduction. Rebuild missing artifacts from pinned
   sources only on permitted data; record the cost and any inaccessible inputs.

Exit: a readiness report, training/validation manifest, annotation audit, and an explicit map from every
planned endpoint to its available truth. Missing clinical truth can narrow the endpoint; it cannot be
filled by a synthetic accuracy score. Stage 0 may finish while later prerequisites remain missing.

## Stage 1 Qualify the intervention generator

Begin with the controlled infrastructure of Experiment 058 where reusable sources are available. Its
negative localizer-robustness result does not qualify a generator as a clinical disease model.
Start with one ST alteration family and two independently implemented rendering/editing mechanisms.
At least one test mechanism must be absent from training and model selection.

- Save baseline and edited signals, the same latent/background identity, all parameters and the actual
  noiseless difference after the scoring preprocessing. Recompute support after filters and resampling.
- Freeze a physical amplitude floor and support rule using generator-design data before training. Report
  mask occupancy, candidate-domain chance, change energy, and any long filtering/ODE tails.
- Check unchanged counterparts, sham edits, channel algebra, mV units and deterministic replay. Reject
  boundary seams, marker leakage, clipping or normalization differences that reveal the label trivially.
- Vary background morphology, rate, edit size, duration and eligible leads within the generator's
  constraints. Preserve all coupled lead changes. Never move a disease label to an arbitrary location.
- Include persistent changes across beats and transient changes, plus noise, drift and gain controls.
  Nuisance parameters must not encode the intervention class or location.
- For edits of real backgrounds, the mask denotes the added intervention, not exhaustive pathology.
  Score only explicitly defined targets; do not call unedited background clinically healthy. Record
  presumed-normal source-label uncertainty. Treat the real-background version as a separate domain.

Exit: deterministic qualified intervention data and artifact controls. Pure simulation establishes
engineering truth; semi-synthetic data adds realistic backgrounds but retains background-label uncertainty.
Train no clinical disease localizer merely because these checks pass.

## Stage 2 Small matched training pilot

Use a compact waveform encoder that keeps lead and temporal coordinates, plus a dense localization head.
Start with a reusable ECG encoder only if its feature resolution and pretraining exposure are verifiable.
Otherwise use the same small architecture with fresh SSL. Choose the architecture and SSL objective
before reading localization scores. Existing pooled embeddings alone do not establish dense resolution.

Real-data SSL predicts masked waveform samples or latent representations, using training patients only.
Use views that preserve target-relevant amplitude, timing and lead information. The encoder sees normal
and abnormal training ECGs without their diagnosis labels; MIMIC and CODE-15 stay unlabeled-only sources.

| Arm | Initialization and localization supervision | Purpose |
| --- | --- | --- |
| R | Random initialization, synthetic masks | Synthetic supervision without SSL |
| S | Real-data SSL, same synthetic masks | Primary test of SSL initialization |
| F | Same SSL encoder frozen, synthetic masks train the head | Does the head suffice without encoder adaptation? |

Match architecture, head initialization, synthetic examples, localization optimizer steps, selection
rule and seed between R and S. Count SSL cost separately. F is a diagnostic, not an equally trainable
parameter comparison. Record parameter counts, exposures, steps, wall time and peak memory.
Keep simple amplitude/QRS-prior maps as contextual baselines, not equivalent supervised competitors.

Freeze one pilot seed and 500 paired synthetic examples as initial design defaults, contingent on
available independent backgrounds. Do not count correlated renders as independent patients. Train and
validation backgrounds and intervention draws must be disjoint. Check a nested 100-example budget only
if the real profile admits it. The pilot is exploratory; its single seed is not a confirmation result.

A real GPU profile of the actual data path must project the entire pilot, preparation and reporting
within two GPU hours and four CPU hours, with no more than 4 GiB of new cache. These are ceilings, not
measured costs. If the projection fails, reduce the prospective scope before training or scoring.
Do not change the pinned dependency files to force the pilot through.

Exit: a committed protocol, qualified generator, exact predecessor integrity, admitted GPU profile,
matched pilot receipt and results report. No run is launched by editing this plan or the queue.

## Stage 3 Evaluation and decisions

The initial engineering primary is S minus R on subject-macro average precision for exact change support
on the held-out rendering mechanism. Report positive-mask prevalence alongside AP. Bootstrap whole
background patients or latent subjects with shared paired draws using `ecg_experiment.intervals`
(2,000 draws; freeze the seed in the executable protocol). Keep variants of a background in one group.
Never bootstrap samples or edited beats as independent people.

Before any score, freeze support generation, output resolution, masking, interpolation, AP aggregation,
valid/unknown regions, tie handling, validation selection and missing-case penalties. Include all eligible
cases. Report fixed-area localization, false flags on shams/nuisances and results separately for focal
and persistent changes. A broad target mask and a correct top point do not prove precise localization.

Pilot advancement requires a point AP gain of at least 0.05 with a positive paired 95% lower bound, and
no more than a five-percentage-point increase in nuisance/sham flags at validation-frozen thresholds.
These are proposed engineering decision thresholds, not clinical utility thresholds. Report absolute
performance and failures even when the comparative rule passes. An unused renderer is an engineering
transfer test; it is not independent clinical confirmation.

If the pilot passes, freeze a three-seed replication and nested label budgets before further scoring.
Study partial encoder fine-tuning or continued real-data SSL as separately controlled successors.
Use existing expert annotations to study real-data transfer with their original scope: beat identity,
joint wave boundaries, ST episode time/channel, and coarse abnormal-region agreement remain separate.
For a confirmatory real-data claim, first secure unused compatible expert truth and freeze the method.
No reserved final test opens without the user's explicit final-test approval.

If SSL improves only same-generator scores, inspect artifact reliance and generator transfer before
scaling. If freezing the encoder works as well as adaptation, keep the simpler recipe. If real-data
transfer fails, retain the engineering result and report the failed transfer.

## Stage 4 Decide whether learned generation is needed

A GAN or diffusion generator is a successor hypothesis: does learned conditional editing improve
real-data transfer over the qualified simple edits at the same synthetic label budget? Preserve the
intervention, background and mask controls; prevent a realism-refinement step from changing untracked
regions. Compare with the same SSL encoder and localization head. Synthetic-only improvement or visual
realism cannot establish clinical gain. More data and a new generator are justified by measured failure
modes, not prerequisites for the first experiment.

## First implementation and sources

The initial command is `PYTHONPATH=. python3.11 -m scripts.validation.check_ssl_synthetic_readiness`.
Its local JSON report records availability only. It reads the versioned prerequisite specification,
not ECG payloads. A successful command does not mean training is ready or the labels are correct.
The command's exit status distinguishes incomplete prerequisites from malformed configuration.

The readiness report is summarized in [ssl-synthetic-localization-readiness.md](ssl-synthetic-localization-readiness.md).
The later [setup report](ssl-synthetic-localization-setup-results.md) records the completed patient
partition, declared-checksum duplicate audit and restored CPU runtime. Those preparations do not admit
training or complete the annotation/waveform-quality gates.
The shared [queue](experiment-queue.md) and [catalog](experiment-queue.json) track this stream.
Experiment numbers will be reserved when the executable training protocol is frozen, avoiding collisions
with the other session. Follow-up studies enter [the ranked backlog](experiment-priorities.md).

Prior evidence: [PR #61](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/61), particularly
Experiments 058, 064, 067 and 073 at the pinned revision above. These reports have not been reproduced
in this checkout. [Self-supervised ECG learning](https://github.com/tmehari/ecg-selfsupervised) supports
the representation-learning motivation; [SSSD-ECG](https://arxiv.org/abs/2301.08227) and
[SimGANs](https://proceedings.mlr.press/v119/golany20a.html) motivate the generation comparisons. None
of those sources establishes this proposed real-SSL plus synthetic-mask localization pipeline.
