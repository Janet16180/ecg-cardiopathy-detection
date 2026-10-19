# Independent training-seed replication on fresh generated ECGs

19 October 2026. Correct support supervision improved introduced-change localization across all
three prospectively fixed new training pairs on 64 fresh Pulse2Pulse backgrounds. All 18 scientific
engineering gates and 11 integrity gates pass. The independent saved-output audit passed with zero
mismatches. This is bounded repeatability for one generator/editor family; three fitted pairs still
provide limited evidence about training variability, and absolute localization remains modest.

Mean equal-mode full-grid AP over the three new fits was **0.114598**. The two primary gains were:

| Fresh-panel primary contrast | Mean gain | Exploratory crossed 95% interval |
|---|---:|---|
| Supervised minus amplitude |0.090876|[0.054197, 0.134239]|
| Supervised minus permuted targets |0.096733|[0.058458, 0.140852]|

All three individual pooled gains are positive against both references. Gains against amplitude
are 0.110741, 0.093712 and 0.068176; against permuted targets they are 0.117279, 0.101920 and
0.071001. These were fixed runs, with no best-fit selection, early stopping or coefficient search.

## Exact predecessor recovery and new evidence

The [protocol](ssl-synthetic-localization-fit-replication-v1.md), configuration and source were
committed before new generation, profiles, fits or scores. Scientific and independent numerical
review approved the exact final source hashes. Seventeen artificial controller/math tests passed;
12 independent-audit tests passed. An artificial figure-interface check is labeled separately and
is not an experimental result.

Old learning case/control/AP/contrast numbers and the preceding diagnosis's three primary intervals
reproduced exactly before new science. Four CPU generator forwards reproduced all original raw,
author-export and canonical arrays. The actual generation profile consumed three original GPU
forwards plus an exact first-input replay, with unchanged generator state. Its outputs were kept
separate from the scientific bank.

The actual training profile saved model and Adam after step one, restored them, and exactly
reproduced steps two and three in both arms, including losses and complete optimizer states. This
consumed ten disposable optimizer updates and two TRAIN-only inference batches. Scientific fits
started from fresh declared initialization and empty Adam states.

Both original arms were then replayed for all 400 updates. Every loss, final model tensor and full
Adam state matched the old run exactly. Those replay states supply fit 0's fresh-panel reference;
fit 0 is excluded from the new-fit primary. This checks more than reproduction of a saved score.

All 64 new raw signals and standardized canonical shams were distinct. Fresh shams were disjoint
from all 128 old shams, without exclusions or replacements. All final float32 intervention/mask,
lead-algebra, intended-support and 99% retained-energy checks passed. The first three scientific
bank outputs exactly matched their profile counterparts. Exact nonduplication is not a privacy,
near-duplicate or patient-independence guarantee.

The unchanged 80 TRAIN backgrounds supplied the same 320 observed variants, capped positive weight
20, true targets and same-mode donor derangement. Three independent initialization/schedule pairs
used the same architecture, Adam settings and 400 updates per arm. Every case appeared ten times;
each scientific arm saw 3,200 presentations. All fits completed before fresh-panel inference.

## Localization and persistent background responses

| Fit | Supervised focal AP | Repeated AP | Pooled AP | Permuted pooled AP | Sham max>0.5 | Nuisance max>0.5 |
|---|---:|---:|---:|---:|---:|---:|
| 0 (reference) |0.058059|0.084157|0.071108|0.015358|24/64|24/64|
| 1 |0.124628|0.144298|0.134463|0.017184|39/64|38/64|
| 2 |0.111072|0.123796|0.117434|0.015514|36/64|36/64|
| 3 |0.067330|0.116466|0.091898|0.020897|58/64|58/64|

Fresh amplitude AP was 0.006236 focal / 0.041209 repeated; the TRAIN-prior baseline was
0.008572 / 0.040656. They were not fitted on fresh backgrounds. The old 48-background panel was
scored only as secondary spent development evidence for the six new states. Its results are saved
separately and never pooled with fresh64. Fit0's old maps were reused without another inference.

The 0.5 cutoff is uncalibrated. The table is an engineering description of added-change controls,
not clinical false-positive rates or labels of healthy ECGs. Permuted control maxima exceeded 0.5
in zero records for every fit. Supervised control mean probability was lower than the matched
permuted mean in each new fit, while its highest outputs were larger; high peaks do not establish
uniform probability inflation.

| New supervised fit | Sham/nuisance hotspot persistence | Focal support precision | Repeated support precision | Focal support recall | Repeated support recall |
|---|---:|---:|---:|---:|---:|
|1|0.872995|0.077057|0.186771|0.130290|0.057114|
|2|0.912292|0.076953|0.153177|0.142049|0.032822|
|3|0.918255|0.050182|0.151120|0.101610|0.029025|

These secondary means use exactly 600 ranked cells per map, with stable flat-index ties. They have
no extra intervals or decision thresholds. Persistent background ranking and low support precision/
recall remain despite successful replication. The fixed fit1 first-eight galleries retain a strong
first case alongside missed edits and diffuse/background responses; examples were not replaced.
The maps do not establish clean or clinically useful localization.

The crossed bootstrap uses 2,000 draws, seed 2026104501: first resample the three fit rows, then one
shared 64-background vector for every selected row and paired contrast. Conditional per-fit fresh
intervals use seed 2026104502; spent48 intervals use 2026103503. No draws are skipped. The 192 fit/
background combinations are not independent patients or fits. Initialization and schedule vary
together. The intervals are exploratory with only three rows, not dependable coverage statements
about training or clinical populations. The gate's 0.05 point margin remains distinct from its
lower-bound>0 requirement; all individual modes and tradeoffs remain in the saved reports.

## Resources, identities and retained artifacts

Main terminal CPU was 110.563450169 s, whole-stage wall 241.488557272 s and GPU-lock wall
140.128609713 s. The combined profile forecast was 150.315616 CPU / 167.773766 whole-wall /
147.268278 GPU-lock seconds. The whole-wall forecast underestimated execution, although all fixed
1,800/1,800/1,200 s caps passed. This discrepancy is retained. The successor will explicitly time
matching per-update loss/component persistence and flushing; that does not revise this forecast.
Peak allocated GPU memory was 149,360,128 bytes (reserved 167,772,160), below 2 GiB. Peak process RSS
was 1,641,037,824 bytes. Main final storage is 891,702,434 bytes, below 1 GiB.

Actual operation totals were four CPU and 68 GPU generator forwards; 10 profile + 800 integrity
replay + 2,400 new-fit optimizer updates; two profile and 400 scientific inference batches.
Scientific inference is 256 batches for all eight states on fresh64 and 144 for six new states on
spent48. Attempt/completion counts match. CUDA0 ran under the shared project lock, deterministic
single-thread settings and disabled TF32. No other GPU process was altered.

The audit added 35.118165415 CPU / 38.689270478 wall seconds and zero GPU, within its separate
300/300 s and 32 MiB caps. Main plus audit CPU is 145.681615584 s. Starting from the pinned prior
44.999764534 CPU / 38.433311437 GPU-lock s, the twelve current-machine scientific stages now total
190.681380118 CPU / 178.561921150 GPU-lock s. Each stage and retained failure is counted once;
historical absent-artifact costs remain unverified. Measurement ends after receipt write/hash,
before terminal-ledger serialization, under the explicit reserved-tail convention.

All evidence remains local under `outputs/ssl_synthetic_localization/generated_fit_replication_v1/`:
`fresh_noise.npy`, `fresh_raw.npy`, `fresh_observed.npy`, `fresh_support.npy` and
`fresh_strict_support.npy`, complete fresh case rows, source snapshots,
profile trajectories, `fit_0`–`fit_3` model/Adam/loss/schedule files, all fourteen saved panel-map
banks and metrics, three fixed figures, closure evidence and the independent audit. No data,
weights, checkpoints, predictions or figures were staged in Git.

| Identity | Value |
|---|---|
| Protocol/config commit |`30f03c8`|
| Main implementation commit |`057cdee` (identical core/runtime/tests to `48fffba`)|
| Core source SHA-256 |`dc640099a7c35336d8ad3bc5aef4b9f30d4bb89a500a1bec8e4dd8b5bb75e06d`|
| Runtime SHA-256 |`f2d43166c0d8f128568abab1fe3eca1ac50d21a9fcd81f2e4be6859c8bdaa4d7`|
| Main summary |`04bab13e3967a3724a63f41d9f164a4315a661c94dda86e0049c72e27b7fdde8`|
| Main terminal ledger |`41743e6cc82a76e7bc2f529033401429f89a0858f921b19e975029b9043f32f9`|
| Independent audit |`a9f9daca25273f23f10dc46faca019bd680eefe24c1c6cc951f33e3fcb46282e`|
| Latest terminal ledger |`f17f0c112a3eeed8b5f46be389c2a06ff8dcb8c2dd6169b1833a398f20a0d264`|

Fresh generator seeds were 2026104000–2026104063; editor indices 1000–1063; new initialization seeds
2026103801–3803 and schedule seeds 2026103901–3903. The public checkpoint remains
`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`, architecture source revision
`f28e4b792e79c503b8b7e0912be9d304a3a7177b`, Hub revision `262e0e67c0101cd4989d7c2f66f16d6d54756e70`.
The independent audit checks bytes/arithmetic/saved states; its operation-path claims are source-bound,
not an independent execution of all neural-network operations. See the [independent results report](ssl-synthetic-localization-fit-replication-independent-audit-v1-results.md).

The four-arm paired/background-suppression successor design was committed as `4500999` before the
coordinator first opened these fresh localization scores. It has a different future64 panel, fixed
coefficients and separate admission/profile/review gates. This study's fresh64 panel is now spent
engineering evidence. The generator is unconditional; physical calibration, physiological fidelity,
disease conditioning, real disease masks and any SSL advantage remain unproven. Clinical labels,
closed tests and the screening pipeline are unchanged. PR #67 remains draft.

## Repository validation

The integrated CPU suite passed1,844 tests, with six skips and two warnings, in104.51seconds.
Repository-wide Ruff including `conftest.py`, source build and wheel build pass. Archive inspection
confirms the new replication/audit modules and excludes raw data, outputs, arrays and weights.
All94 recorded current main/failure scientific-source bindings still match. Local logs are
`outputs/ssl_synthetic_localization/fit_replication_full_cpu_tests_v1.log` and
`fit_replication_package_build_v1.log`. Setup validation is separate from scientific runtime costs.
