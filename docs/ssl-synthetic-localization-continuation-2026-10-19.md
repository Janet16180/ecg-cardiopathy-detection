# Synthetic-generation and localization continuation

19 October 2026. Start here instead of treating the 18 October machine handoff as current evidence.
PR #67 remains draft, based on main including PR #70 at 41871e3. Current research uses released-generator
signals and exact introduced-change masks; the complementary screening studies do not complete it.

## Executed progress

| Stage | Actual finding | Report |
|---|---|---|
| GPU inventory |Both RTX 3090s execute PyTorch operations on the host; the restricted namespace hides device nodes. GPU0 ran the studies under the shared lock.|`outputs/ssl_synthetic_localization/continuation_inventory_v1.json`|
| Released checkpoint probe |Pulse2Pulse public weights/source downloaded and hash-verified; six CPU forwards, four seeds and exact replay; seven technical gates pass.|[Probe](ssl-synthetic-localization-pretrained-generator-standalone-v1-results.md)|
| Qualityv1 profile |Stops before the bank:1,550.645s forecast exceeds1,200s. Failure, noise, profile and costs retained; independently verified.|[Retained failure](ssl-synthetic-localization-generated-quality-v1-results.md)|
| Qualityv2 |Separately frozen1,800s resource amendment;128native signals/512canonical variants,15operational gates pass; coherence difference0.809425[0.803893,0.814856].|[Bank results](ssl-synthetic-localization-generated-quality-v2-results.md)|
| Randomized support learning |80TRAIN/48EVALUATION generator seeds, matched400-update supervised/permuted-target models, observed signal only. All12scientific+9integrity gates pass.|[Learning results](ssl-synthetic-localization-randomized-learning-v1-results.md)|
| Independent verification |Probe, retained failure, quality bank and learning all verified. New learning audit checks8,477case/4,715metric scalars and 103bindings with zero mismatches.|[Learning audit](ssl-synthetic-localization-randomized-localization-independent-audit-v1-results.md)|

The new model's pooled full-grid AP is 0.077538 versus 0.024522 for amplitude and 0.021353 for permuted
supervision. Gains are+0.053016[0.027369,0.083052] and+0.056185[0.031928,0.083607]. Absolute AP
is modest and the fixed gallery shows diffuse background activity. Supervised sham/nuisance maxima
exceed 0.5 in 23/48 and 25/48records; this is an uncalibrated engineering diagnostic, not clinical false
positives. Only one initialization/schedule and smooth-offset family were studied. The frozen point-estimate margin
passes narrowly; the interval does not establish a gain of at least 0.05.

No real ECGs, closed sets, clinical labels, external inference APIs or historical SSL checkpoints
entered these new studies. No real disease localization or SSL advantage is demonstrated. The
generator is unconditional; its native output units/cohort and license limitations remain unresolved.
Known masks locate separately added numerical changes, not generator-controlled pathology.

## Local artifacts and identities

All new scientific artifacts are under`/home/janetrivera/baseline/outputs/ssl_synthetic_localization/`:

- `pretrained_generator_standalone_v1/`:public weights, reviewed source/license, noise, generated traces,
  replay and independent audit.
- `generated_quality_v1/`:immutable failed profile and metadata audit; no bank exists here.
- `generated_quality_v2/`:128native8-lead signals,512canonical12-lead variants, strict/thresholded
  supports, fixed first-eight quality figures, metadata and independent audit.
- `randomized_localization_v1/`:512new randomized variants, masks, schedules, both fitted model/Adam
  states,384held-out model maps, fixed galleries, metrics and independent audit.

| Anchor | SHA256 |
|---|---|
| Public checkpoint |`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`|
| Qualityv2 summary |`0067dc16c609a7209cb626e8d31600f6b654d192e3b0803968ca3275db1660b2`|
| Qualityv2 independent audit |`5fc4ceb2e5fd2a405010e6527c555c1a675c56991e755531e02aabd67eaabb1f`|
| Learning summary |`fc4bf8934c95456d35cff5cc697b8b8f9b7563d79018fbc940fd68feda72064f`|
| Learning independent audit |`b44e847a00a6e41587dab9d1d95097e750d3db7568c3a2968ca899f43e2a17ac`|
| Latest cumulative-cost ledger |`383ab5707ecfeb28773d0617c7aaa8b8875cc50d77fa17bd82c1db299cf8f1a5`|

Eight current-machine scientific stages total 32.489216 CPU and 38.433311 GPU-lock seconds, including
failed attempts and audits once. Implementation, source review and artificial tests are setup.
The old reported 745.814295 CPU/382.224159 GPU seconds are historical context, not locally verified
costs. The old localization output tree was absent; it was neither recovered from the other checkout
nor fabricated. Its saved-map mode diagnosis remains blocked on the original permitted artifacts.

Each protocol, source, test, receipt and downloaded asset bound by these runs is now frozen. Use
forward-only successor files and fresh attempt directories, prospective budgets and source review.
Do not overwrite attempts, reuse profile-fitted weights, or reinterpret failed gates.

## Next research

The ranked backlog records saved-map background-activation diagnosis, independent fit/seed replication,
and public wave-annotation qualification. None is running. Existing48evaluation seeds are now spent
engineering development data. A future model/objective selection needs a prospectively stated new
evaluation policy. Exact fixed-position quality fixtures must never become a localization benchmark.
Wave-boundary annotations can validate timing anchors without recruiting a cardiologist; they do
not automatically supply regional disease masks. LUDB's permitted local raw directory remains empty.

Read [data locations and file ownership](ssl-synthetic-localization-data-ownership.md). Another agent
owns`~/ecg-cardiopathy-detection`; only permitted raw paths are shared for read-only use. Its Git state
and outputs remain closed. Baseline raw links sit beside tracked DVC pointers; never replace their
directories or stage the links. All weights, maps, generated signals, checkpoints and figures remain
local. Preserve Challenge/EchoNext closed-test and final-freeze restrictions.

## Delivery validation

Final local CPU suite: 1,793 passed, six skipped in 72.57 s. These are repository test skips,
not experimental results. Repository-wide Ruff passes. Source and
wheel builds pass, and their contents exclude raw datasets, outputs, weights, arrays and environments.
All64 main/failure source bindings still match after assembly; no frozen scientific source changed.
Logs are `outputs/ssl_synthetic_localization/continuation_cpu_tests_final_v3.log` and
`continuation_package_build_final_v3.log`. Read live PR checks for the final GitHub head rather than
assuming a local test log establishes remote CI completion. PR #67 stays draft and is not merged.
