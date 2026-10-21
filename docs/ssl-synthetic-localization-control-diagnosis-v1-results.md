# Background hotspots in released-generator localization maps

19 October 2026. The saved-map diagnostic found that the correctly supervised model's highest
ranked control cells align with derivative activity in generated backgrounds, and mostly persist
between paired controls. This supports a background-response concern alongside the earlier
introduced-change learning result. It does not identify physiological waves or disease, establish a
causal mechanism, or change the predecessor's twelve-gate PASS.

The coordinator executed one CPU run after the [protocol](ssl-synthetic-localization-control-diagnosis-v1.md)
and implementation were committed and scientific/numerical source review gave GO. The run completed
integrity checks, exactly reproducing all predecessor case metrics, control summaries, contrasts,
bootstrap intervals and scientific gates before computing the new diagnostics. Independent saved-output verification passed with zero mismatches; see the
[independent report](ssl-synthetic-localization-control-diagnosis-independent-audit-v1-results.md).

## Measurements from saved outputs

The study used both fitted arms' saved maps on all 48 already-spent evaluation generator seeds,
indices 80–127, without selection or exclusion. Each map contributed exactly its highest-ranked
600 of 60,000 cells, with deterministic ascending-flat-index tie breaking. This forced ranked area
is separate from the predecessor's uncalibrated maximum>0.5 diagnostic.

Activity was the strict upper decile of each canonical lead's float64 squared first differences,
smoothed over 20 samples. Actual occupancy was 0.1 in these cases. One prospectively seeded common
time shift per background displaced each activity mask while preserving lead occupancy and internal
cross-lead timing. The reference is one fixed alignment comparison, not a permutation test.

| Primary control activity overlap difference | Mean | 95% whole-seed interval |
|---|---:|---:|
| Supervised native minus shifted |0.200347|[0.118436, 0.271866]|
| Permuted native minus shifted |0.017083|[-0.041778, 0.071928]|
| Paired supervised minus permuted |0.183264|[0.094637, 0.273739]|

Each seed averages sham/nuisance differences equally. Intervals use 2,000 shared whole-seed draws,
RNG 2026103701, with zero skipped draws. They condition on the two fixed fits, spent generated
backgrounds, activity proxy and chosen reference; they do not measure training-seed or patient
uncertainty. There was no new scientific advancement threshold or model selection.

| Secondary control measurement | Supervised | Permuted |
|---|---:|---:|
| Sham activity overlap among hotspots |0.307951|0.159965|
| Sham shifted activity overlap |0.107118|0.143611|
| Nuisance activity overlap among hotspots |0.304861|0.148125|
| Nuisance shifted activity overlap |0.105000|0.130313|
| Sham/nuisance hotspot persistence |0.878160|0.685451|
| Sham TRAIN-prior hotspot overlap |0.007500|0.025313|
| Sham shifted-prior overlap |0.011042|0.044028|
| Nuisance TRAIN-prior hotspot overlap |0.008160|0.030694|
| Nuisance shifted-prior overlap |0.011840|0.041528|

Supervised control hotspots are enriched near this derivative proxy, but roughly 69% remain outside
its activity mask. The 87.8% paired-control persistence shows substantial stability of ranked
background locations under the small nuisance perturbation; it does not establish invariance of
all scores. Exact predecessor closure retains 23/48 sham and 25/48 nuisance supervised records with
maximum>0.5, versus zero in each permuted group. These are not clinical false-positive rates.

Lead allocation is uneven: supervised sham/nuisance hotspots allocate 35.080%/34.767% to V1;
permuted hotspots allocate 43.413%/45.316% to III. These are descriptive associations. The fixed
TRAIN-prior hotspot overlap is below its shifted counterpart in both controls/arms, so this specific
ranked-prior reference provides no positive alignment explanation. It does not rule out other
position or lead biases. The prior cutoff is 0.04375: 436 cells are strictly above it and 796 tie at
it, of which 164 are selected by flat index. That tie rule can impose apparent spatial preferences.

## Introduced-change ranking remains limited

| Arm / mode | Hotspot support precision | Support recall | Matched nuisance precision | Precision response | Positive/nuisance hotspot persistence |
|---|---:|---:|---:|---:|---:|
| Supervised focal |0.041146|0.071728|0.000000|0.041146|0.889375|
| Supervised repeated |0.134826|0.040075|0.012118|0.122708|0.769549|
| Permuted focal |0.000451|0.000999|0.000000|0.000451|0.969549|
| Permuted repeated |0.036389|0.006844|0.019549|0.016840|0.841215|

These secondary quantities are 48-seed means, without additional intervals. The supervised model
moves some ranked area into known added support, especially for repeated offsets, while most of its
ranked cells remain outside that support. High positive/nuisance persistence, low precision/recall
and modest predecessor pooled AP 0.077538 limit usefulness as a localizer despite the frozen
supervision-versus-controls gains. These support masks locate externally added numerical changes;
they are not regional pathology annotations.

## Artifacts, identities and resources

All new files are local under
`outputs/ssl_synthetic_localization/generated_control_diagnosis_v1/`: `cases.jsonl` contains all 48
seed rows, `cases.csv` all 384 arm/variant rows, `predecessor_closure.json` the exact integrity closure,
`summary.json` aggregates and source/input/output bindings, and `prospective_ledger.json` terminal
costs. The coordinator viewed both fixed figures: `fixed_control_hotspots.png` retains indices
80–87 in four model/control columns; `diagnostic_summary.png` shows all seed values, lead allocation
and persistence. Gray marks derivative activity, red ranked hotspots and green their intersection. Their pixels are
source-bound renderings, not independent physiological evidence.
The gallery's tight title spacing is retained rather than rerendering frozen outputs.

| Identity | Value |
|---|---|
| Protocol/config commit |`8d1c4f8` (same bytes as original `5f4d663`)|
| Implementation commit |`359e352` (same bytes as original `b892774`)|
| New numerical module |`f7d024717a782bdd1bf666b9aa56d9d1e5a90894ab257fcd85ec919d975ec7ac`|
| Config |`62f2d40f906bf3f16cf817aef609af4e39275058a4c2c0a0cd327524c7a9f2b2`|
| Predecessor learning summary |`fc4bf8934c95456d35cff5cc697b8b8f9b7563d79018fbc940fd68feda72064f`|
| New main summary |`8c5c4615ae3a478b322eabd2006f0ed8dafc7750228f65a034623054a525365f`|
| New main terminal ledger |`e0628d5b8f0ee6ec0d0879ee41121c62fc1aff87a9b6e0d576b6e534b3caf51b`|

Generator/background seeds remain 2026103080–2026103127; diagnostic shifts use 2026103600 plus global
index. The released Pulse2Pulse checkpoint identity remains the previously audited
`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`; this study did not reopen weights.
See the [executed probe](ssl-synthetic-localization-pretrained-generator-standalone-v1-results.md)
and [learning report](ssl-synthetic-localization-randomized-learning-v1-results.md) for source,
checkpoint, training and intervention identities. The generator is unconditional; physical calibration, clinical normality, disease control and real
disease localization remain unresolved.

Main terminal cost is 8.358138127 CPU s and 12.409262839 wall s, GPU zero, below 180/240 s caps.
Peak process RSS is 770,457,600 bytes. Main files occupy 1,029,906 bytes including the terminal ledger,
below 32 MiB. CUDA was hidden and numerical/PyTorch intraop/interop threads were one. Timers and
byte guards retained the failure reserve. Matplotlib used a temporary local cache after its default
cache directory was unwritable; the run completed without a scientific retry or output replacement.
No model/generator forward, optimizer update, real-waveform read, download or external inference
call occurred. The old learning tree stayed unchanged.

The previous independently verified cumulative costs were 32.489216225 CPU s and
38.433311437256634 GPU-lock s. Adding this main stage once gives 40.847354352 CPU s, with GPU cost
unchanged. The terminal endpoint includes receipt writing/hashing before ledger serialization;
that bookkeeping tail is excluded by the prospectively stated convention. The independent audit added 4.152410182 CPU s and 4.506168588 wall s, GPU zero; peak RSS
703,381,504 bytes. Its 45,939 bytes and the main's 1,029,906 bytes each pass their separate 32 MiB
allowances (1,075,845 bytes combined). All ten measured new-machine stages now total
44.999764534 CPU s and 38.433311437256634 GPU-lock s. Historical absent-artifact costs remain
unverified and are not reconstructed.

The independent checker reconstructed 19,735 diagnostic scalars, 22,656 CSV fields and 2,701
predecessor scalars, verified 52 file bindings before/after calculation, and found zero mismatches.
It independently used lexicographic ranking and derivative arithmetic, with shared grouping/resampling
helpers. Figures and zero-operation boundaries are source-bound, not an independent physiological
review or model-execution trace. Audit summary SHA-256:
`19311a2f6675071945ee058d93f4b222119e474bedff4c1ea3d434ebcc57bf3a`;
latest terminal ledger:
`a2e2c31b6eda8a5aaed2498e69f6ee2bed97bd90d6b8b3d133e6c15a0f617f77`.

The next ranked experimental question is independent fit-randomness replication with an explicit
new evaluation policy. A paired-background suppression objective can be a later controlled
comparison, with all localization modes and controls preserved, after replication; this diagnosis
does not authorize tuning on these spent 48 seeds. Public wave annotations can separately qualify
timing anchors, without turning their boundaries into disease masks. No fit, closed test, clinical
label change or automatic PR merge follows this result.

## Delivery validation

The full integrated CPU suite passed: 1,813 tests, six skipped and two warnings, in 81.44 s. Seven
new main tests and 11 independent-audit tests passed separately. Repository-wide Ruff including
`conftest.py` passes. Source and wheel builds pass; archive inspection confirms both new numerical
modules are present and raw/output/weight/prediction artifacts are absent. All 76 main/failure source
bindings and six independent audit-core bindings match. Validation logs remain local at
`outputs/ssl_synthetic_localization/continuation_control_diagnosis_cpu_tests_v1.log` and
`continuation_control_diagnosis_build_v1.log`. These setup checks are separate from scientific costs.
