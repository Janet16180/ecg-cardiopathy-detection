# Independent fit replication on a fresh generated panel

19 October 2026. Prospective continuation after the independently audited saved-map diagnosis.
This study tests repeatability of the introduced-change learning result across three newly fixed
initialization/schedule pairs and fresh generated backgrounds. It preserves the model, optimizer,
training examples and edit family. It does not test a new loss, real disease localization or SSL.
All numerical choices and input identities are frozen in
`configs/ssl_synthetic_fit_replication_v1.json` before generation, profiling, fitting or new scores.

## Roles, seeds and immutable inputs

Keep the original 80 TRAIN backgrounds and all 320 saved variants/targets unchanged. Indices 80–127
are the spent 48-background diagnostic panel, used only for secondary reporting. Do not select a
model, objective, threshold or hyperparameter from either evaluation panel during this study.

| Fit | Initialization seed | Schedule seed | Role |
|---|---:|---:|---|
| 0 | 2026103501 | 2026103502 | Exact predecessor training replay and fresh-panel reference |
| 1 | 2026103801 | 2026103901 | New independent fitted randomness |
| 2 | 2026103802 | 2026103902 | New independent fitted randomness |
| 3 | 2026103803 | 2026103903 | New independent fitted randomness |

Generate exactly 64 new backgrounds from CPU torch.Generator uniform[-1,1] float32 inputs, seeds
2026104000–2026104063. Use global editor indices 1000–1063 in the unchanged frozen `make_case` recipe:
noise seed =2026103000+index, edit RNG seed =2026103400+index and nuisance RNG seed =2026103100+index.
Thus fresh edit seeds are 2026104400–2026104463 and nuisance seeds 2026104100–2026104163. Four variants,
randomization draw order, robust scaling, lead reconstruction, strict/0.01 support definitions and
all support/energy qualification gates remain identical to the original learning protocol.

Reserve indices 2000–2063 and generator seed IDs 2026105000–2026105063 for a separately committed
successor. They are identifiers only: do not generate, inspect or score that future panel now.
Fresh-panel qualification requires finite/nonflat valid geometry, exact limb equations and supports,
no changes outside intended hulls, at least99% retained edit energy, and exact same-input replay.
Fresh raw outputs and canonical shams must be distinct within64; fresh canonical sham hashes must
also be disjoint from all128 old canonical shams. On failure, retain the entire attempt and stop;
there is no sample exclusion, replacement or favorable-seed search. Exact nonduplication is not a
privacy, near-duplicate or patient-independence claim.

The config's explicit `parents` catalogs bind the local public generator, original learning outputs
and latest diagnosis. Read only those permitted local files and new outputs. Old numeric TRAIN rows
are admitted for unchanged training/replay and duplicate comparison; old evaluation rows remain
secondary development data. Use the existing public checkpoint/source, never a download or external
inference service. No raw clinical ECG, closed set, age analysis or clinical label is admitted.
The original generator's unresolved cohort, license and physical-calibration limitations persist.

## Closure before new science

Verify every allowed input hash, passed parent receipts and current/snapshotted source bindings.
Exactly reproduce the old learning per-case AP, amplitude/prior AP, nuisance-site AP, response,
selectivity, control summaries, contrasts/intervals and twelve gates from saved old maps. Also
recompute the latest diagnosis's three primary estimates and intervals exactly using its frozen
pure helpers; do not claim reconstruction of its other secondary rows/CSV/figures. No model forward
is needed for these closures. Bind the diagnosis's passed audit/ledger as the latest accounting anchor.

Load the released generator strictly with the reviewed architecture and weights-only tensor route.
Replay its four original CPU probe inputs, requiring exact raw, author-export and canonical output
agreement. All four forwards are charged. After the training profile passes, replay both original
400-update arms from recreated initialization/schedule, requiring every saved loss, each named final
model tensor and the complete Adam state/options to match the original exactly. Require recreated
initialization, schedule, target donor mapping and class weight to match their original artifacts.
A mismatch stops before the three new fits and before any fresh localization score. Successful replay
states provide fit0's fresh-panel reference; do not refit fit0 again or infer its old48 maps again.

## Unchanged learning and operation allocation

Use the same observed-only three-layer Conv1d network, ReLUs, padding/dilation, default initialization,
float32 BCEWithLogitsLoss, capped TRAIN positive weight and full Adam specification as the frozen
randomized-learning protocol. All three new pairs use the same80 TRAIN examples, supervised masks
and fixed same-mode donor `(i+1)%80` for the permuted arm. No label or waveform augmentation changes.
For each fit's schedule seed, draw ten successive default_rng permutations of80 backgrounds; expand
consecutive pairs into the four variants in sham/nuisance/focal/repeated order. Each arm gets400
updates, batch8 and3,200 presentations. On every batch update supervised then permuted, with identical
initial model tensors and fresh separate Adam states. Save all losses, initial/final model states,
final optimizer states, schedules and counts. Require step400 for each parameter's Adam state.

Train all pairs completely before fresh-panel inference. Evaluate every fit0–3 arm on all64 fresh
backgrounds: 8 states×256 cases /batch8 =256 inference batches. Evaluate the six new states on the
spent48 panel: 6×192/batch8 =144 batches. Total scientific evaluation is400 batches. Save native
float32 probability maps and all case/control metrics. Fit0's old panel uses its existing saved maps.
No best-fit selection or ensemble replaces the fit-specific comparisons.

| Operation | Count |
|---|---:|
| CPU generator predecessor forwards | 4 |
| GPU generator profile forwards | 4 |
| GPU fresh-bank generator forwards | 64 |
| Disposable localizer profile optimizer updates | 10 |
| Exact old-pair replay optimizer updates | 800 |
| Three new pairs' optimizer updates | 2400 |
| Disposable profile localizer inference batches | 2 |
| Scientific localizer inference batches | 400 |

## Actual generation/training profiles and admission

Use CUDA0 under the shared literal `cuda` lock, deterministic operations, disabled matmul/cuDNN TF32,
CUBLAS_WORKSPACE_CONFIG=:4096:8 and one numerical/intraop/interop thread. Require at least2GiB free
memory and peak allocated memory below2GiB. Leave unrelated GPU processes untouched. No CPU fallback.

Profile the actual generator-to-qualified-case path on the first three fresh inputs, recording the
cold first and recurrent second/third full-path CPU/wall costs separately. Repeat the first input
once and require exact raw/derived-case equality. Retain all profile outputs/costs and unchanged
model-state evidence. These four forwards do not populate the scientific bank. After admission,
generate all64 bank cases, requiring its first three outputs to equal their profile counterparts;
no additional bank replay is needed. The generator state must remain unchanged throughout.

Profile localizer fit0's actual TRAIN path using the frozen recovery procedure: three updates per
arm, save full model/Adam after step1, restore it and replay steps2/3, requiring exact losses and
complete model/optimizer equality at both recovered steps. This consumes five updates per arm,
ten total. Profile one batch8 inference per arm using the first scheduled TRAIN pair, never fresh
panel scores. Scientific fits start from untouched saved initialization and fresh optimizers.

After both profiles, let G be the maximum recurrent generation step2/3 full-path time, R the maximum
original training step2/3 full-path time across both arms, and E the maximum profile inference-batch
time. Require a single combined forecast:

`already_spent + 2*64*G + 2*3200*R + 2*400*E + 90 seconds reporting reserve`.

Apply it to process CPU<1800seconds, whole-stage wall<1800seconds and GPU-lock wall<1200seconds,
using the appropriate elapsed CPU/wall/lock measurements. Cold setup, all original/replayed profile
work and CPU closures are charged once in elapsed work; no favorable timing is selected or cost
omitted. Main generation, old replay and new fits begin only after the combined gate passes.
Forecast all retained bytes, including all maps/checkpoints and a32MiB reporting/audit-independent
reserve, against1GiB. Record all measured components, three forecasts and peak memory. Preserve any
failed profile and stop this version without automatic retry or relaxed post-result thresholds.

## Fresh-panel primary and uncertainty

Use the same complete12×5000-grid positive AP, amplitude comparator, fixed TRAIN spatial prior,
paired nuisance-site AP, AP response, selectivity and descriptive control summaries as before.
Each new fit contributes a64-background×2-mode result. The primary population is the three new
fits on the fresh64 only. First average focal/repeated equally within each fit/background, then
average backgrounds and fits equally. The two primary contrasts are supervised minus amplitude
and supervised minus the matched permuted arm. Fit0 fresh results and all spent-panel results are
secondary and excluded from these primary aggregates.

The same64 backgrounds occur in every fit, so uncertainty is crossed, not192 independent observations
and not independently nested backgrounds within fits. For each of2,000 draws from
`default_rng(2026104501)`, first draw `rng.integers(0,3,3)`, then one64-background resample using shared
`patient_groups`/`patient_resample`. Apply that same background vector to all selected fit rows and
all paired arms/modes/contrasts. Compute each drawn mean and linear2.5/97.5percentile bounds. No draw
is skipped. Always report all three individual fit estimates, their range, and that only three fit
randomness pairs support these exploratory crossed intervals; they are not reliable large-sample
training-seed, patient or clinical confidence statements.

For each fit on fresh64, report conditional background-only intervals using2,000 shared draws from
default_rng(2026104502). These condition on that fixed fit. Spent48 conditional intervals use the
original seed2026103503 and48-background grouping. Never pool the old and fresh panels. The experiment
varies initialization and schedule together, so it does not separately identify their effects.

The engineering replication conjunction has18 gates:

- For each aggregate pooled primary contrast: mean gain≥0.05 and crossed lower bound>0 (four gates).
- For each comparator and each mode: aggregate crossed direct-AP gain lower bound>−0.01 (four gates).
- Aggregate supervised AP-response lower bound>0 separately for both modes (two gates).
- Aggregate supervised selectivity-response lower bound>0 separately for both modes (two gates).
- Each of the three new fits has pooled mean gain>0 versus each comparator (six direction guards).

All numerical/replay/profile/input/resource integrity checks must also pass. The six direction guards
do not require every fit individually to pass the complete original margin rule. Report fit-specific
mode tradeoffs even if the aggregate passes. The0.05 point margin does not require the interval lower
bound to exceed0.05. With only three fitted pairs, a PASS is a bounded repeatability result, not robust
training-distribution characterization or promotion of the clinical screening pipeline.

## Controls, reporting and continuation

Report absolute AP, every fit/mode contrast, spatial-prior baseline and all model controls on each
panel. Keep the strict record maximum>0.5 diagnostic uncalibrated and descriptive. Replicate the
saved-map top600 stable C-index tie rule for sham/nuisance persistence, positive/nuisance persistence,
positive support precision/recall, same-mask nuisance precision and precision response. These are
secondary descriptive means per fit/arm/panel; no activity-shift analysis or new diagnostic threshold
is added. A positive primary must remain alongside modest absolute AP or persistent background
responses. No clinical normality, wave-boundary truth or regional disease masks are inferred.

Figures show all fit-specific primary/mode values and control distributions. Fixed waveform/map
examples use new fit1, fresh global indices1000–1007, both modes and both arms; no favorable fit or
example selection. The coordinator writes the actual results from saved artifacts, obtains an
independent saved-output audit and updates both queues/backlog.

Fresh64 becomes spent engineering evidence after this study. The user separately authorized continued
research, but a paired-background suppression objective comparison still requires its own committed
protocol, matched controls, actual-path profile and the separately reserved future panel. A failed
scientific replication conjunction can motivate that new hypothesis if technical integrity passes;
retain the failure and do not call it confirmed replication. Do not tune/retry this recipe on fresh64.
No successor fit or reserved-seed generation occurs within this replication executable.

## Storage and additive costs

New path: `outputs/ssl_synthetic_localization/generated_fit_replication_v1`. Main caps: CPU1800seconds,
wall1800seconds, GPU-lock1200seconds, outputs1GiB, with1MiB terminal-receipt reserve and2seconds for
failure reporting. The independent audit has separate CPU300seconds, wall300seconds and32MiB, with
zero model/generator forwards or optimizer steps. Main and audit preserve failures and refuse
existing attempt paths. Check live limits throughout; retain completed arrays/counts before rejection.
Reuse old arrays and weights by reference; never copy full predecessor output trees. Hash sources,
inputs and snapshots before/after. The reviewed executable is
`scripts/experiments/run_ssl_synthetic_fit_replication.py` with new reusable modules in `ecg_experiment/`.

The pinned latest diagnosis audit ledger
`a2e2c31b6eda8a5aaed2498e69f6ee2bed97bd90d6b8b3d133e6c15a0f617f77` contains exact current-machine
cumulative CPU44.99976453399999seconds and GPU-lock38.433311437256634seconds. Add each new stage's own
terminal cost once, including old-pair replay and profile work. Do not add earlier nested cumulative
values again. Receipt writing/hashing precedes terminal measurement; ledger serialization is the
reserved unmeasured tail. Missing historical factorial costs remain unverified. All arrays, weights,
optimizer states and maps stay local. PR #67 remains draft, with no automatic merge.
