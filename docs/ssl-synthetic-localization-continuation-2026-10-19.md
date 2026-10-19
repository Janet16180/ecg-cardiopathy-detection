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
| Independent fit replication |Three new matched pairs on64fresh generated backgrounds; all18scientific+11integrity gates pass. Mean supervised AP0.114598; gains amplitude0.090876/permuted0.096733. Exactold400-step losses/model/Adam replay.|[Replication](ssl-synthetic-localization-fit-replication-v1-results.md)|
| Independent replication audit |Zero mismatches;264bindings, all14saved panels and all18+11gates verified. Main+audit145.681616CPU/140.128610GPU-locks.|[Audit](ssl-synthetic-localization-fit-replication-independent-audit-v1-results.md)|
| Saved-map hotspot diagnosis |Exact predecessor closure; supervised activity alignment excess 0.200347 [0.118436,0.271866], sham/nuisance hotspot persistence 87.8%; focal/repeated hotspot support precision only 4.1%/13.5%.|[Diagnostic](ssl-synthetic-localization-control-diagnosis-v1-results.md)|
| Independent diagnostic audit |19,735 diagnostic / 22,656 CSV / 2,701 predecessor scalars and 52 bindings; zero mismatches, no new model operations.|[Diagnostic audit](ssl-synthetic-localization-control-diagnosis-independent-audit-v1-results.md)|
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
- `generated_control_diagnosis_v1/`:48 complete diagnostic seed rows/384 CSV rows, exact predecessor
  closure, fixed figures, source/input identities, measured ledger and independent audit.

| Anchor | SHA256 |
|---|---|
| Public checkpoint |`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`|
| Qualityv2 summary |`0067dc16c609a7209cb626e8d31600f6b654d192e3b0803968ca3275db1660b2`|
| Qualityv2 independent audit |`5fc4ceb2e5fd2a405010e6527c555c1a675c56991e755531e02aabd67eaabb1f`|
| Learning summary |`fc4bf8934c95456d35cff5cc697b8b8f9b7563d79018fbc940fd68feda72064f`|
| Learning independent audit |`b44e847a00a6e41587dab9d1d95097e750d3db7568c3a2968ca899f43e2a17ac`|
| Latest cumulative-cost ledger |`f17f0c112a3eeed8b5f46be389c2a06ff8dcb8c2dd6169b1833a398f20a0d264`|

Twelve current-machine scientific stages total 190.681380 CPU and 178.561921 GPU-lock seconds, including
failed attempts and audits once. Implementation, source review and artificial tests are setup.
The old reported 745.814295 CPU/382.224159 GPU seconds are historical context, not locally verified
costs. The old localization output tree was absent; it was neither recovered from the other checkout
nor fabricated. Its saved-map mode diagnosis remains blocked on the original permitted artifacts.

Each protocol, source, test, receipt and downloaded asset bound by these runs is now frozen. Use
forward-only successor files and fresh attempt directories, prospective budgets and source review.
Do not overwrite attempts, reuse profile-fitted weights, or reinterpret failed gates.

## Current autonomous continuation

The user authorized continued autonomous research, including recommended successor steps. Independent
fit replication is complete and audited; its fresh64 panel is now spent engineering evidence.
The four-arm P×K background-suppression protocol was committed as4500999 before the coordinator
opened those new scores. Identity admission, actual-path I/O-aware profiles and source review gate
its execution on a different64-background panel. No clinical mask or label definition changes.
Public wave-annotation qualification remains ranked, without a download or real-data read yet.

The active coordinator worktree is `/tmp/ecg-synthetic-autonomous-run`, backed by its own temporary
bare Git metadata. This preserves protocol commits despite timeouts in approval review for protected
baseline Git metadata. It shares only permitted baseline output/environment/source links and never
modifies the other checkout. New artifacts remain under baseline/outputs. Baseline's unrelated
untracked `instruction.md` remains untouched. Keep PR67 draft; sync/push only authored source and
aggregate reports, never arrays/weights/predictions. Existing original worktrees remain retained.

Read [data locations and file ownership](ssl-synthetic-localization-data-ownership.md). Another agent
owns`~/ecg-cardiopathy-detection`; only permitted raw paths are shared for read-only use. Its Git state
and outputs remain closed. Baseline raw links sit beside tracked DVC pointers; never replace their
directories or stage the links. All weights, maps, generated signals, checkpoints and figures remain
local. Preserve Challenge/EchoNext closed-test and final-freeze restrictions.

## Delivery validation

After replication, the local CPU suite passed1,844 tests, six skipped, two warnings in104.51s.
The17 main and12 independent audit tests also passed separately. Repository-wide Ruff including
conftest.py, source and wheel builds pass; archives exclude private scientific artifacts. All94
current main/failure source bindings match. Logs: `outputs/ssl_synthetic_localization/fit_replication_full_cpu_tests_v1.log`
and `fit_replication_package_build_v1.log`. Check live GitHub status for the pushed head. This
validation predates the successor implementation and does not certify future code or scores.
