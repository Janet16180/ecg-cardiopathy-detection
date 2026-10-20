# Temporal ranking and lead selection in saved synthetic-localization maps

19 October 2026. The combined paired/suppression objective PK improved two distinct descriptive
ranking endpoints against unchanged BCE B: temporal ranking within the known edited leads and
ranking of support-bearing leads. The independent map-derived audit passed with zero mismatches.
This adds evidence of within-lead ranking improvement alongside lead selection on this fixed edit
family. It is not a causal decomposition of full-grid AP or a new scientific success verdict.

| Prespecified PK minus B contrast | Mean | Exploratory crossed 95% interval |
|---|---:|---|
|`PK_minus_B__active_lead_temporal_AP__focal`|0.103682|[0.058979, 0.155425]|
|`PK_minus_B__active_lead_temporal_AP__repeated`|0.103563|[0.055897, 0.153979]|
|`PK_minus_B__active_lead_temporal_AP__pooled`|0.103622|[0.069175, 0.140407]|
|`PK_minus_B__lead_presence_AP__focal`|0.086178|[0.038052, 0.138990]|
|`PK_minus_B__lead_presence_AP__repeated`|0.088217|[0.041863, 0.139527]|
|`PK_minus_B__lead_presence_AP__pooled`|0.087197|[0.052253, 0.128440]|

The primary descriptive contrast is the pooled temporal AP gain, 0.103622 [0.069175, 0.140407].
Its three individual fit gains are 0.079939, 0.116667 and 0.114262. Pooled lead-presence gain is
0.087197 [0.052253, 0.128440], with fit gains 0.067417, 0.085574 and 0.108601.
Every mode/fit PK-minus-B point gain is positive. These are three fits and 64 spent backgrounds,
not 192 independent fitted models or patients. There is no scientific PASS rule or arm selection.

## Definitions, weighting and limits

Temporal AP computes a separate 5,000-sample AP on each support-bearing canonical lead, then
averages active leads equally per case, followed by equal mode/background/fit averaging. The true
edited leads are supplied only to scoring. This is an oracle-conditioned evaluation, not an input
to inference, deployment performance, or a test of finding the leads. It does not measure accurate
onset/offset delineation. Derived limb leads have fixed equal canonical weights but are not
independent physical channels or observations.

Lead-presence AP uses the maximum probability across time for each of twelve canonical leads,
against any final numerical support on that lead. It assesses lead ranking rather than timing;
maxima remain uncalibrated and algebraically coupled leads can influence it.

Both endpoints use the same tie-grouped AP convention and final saved support as the preceding
full-grid endpoint. Controls with undefined all-zero AP targets are not scored. All four arms and
both edit modes remain visible in the fixed 2×2 figure, with common 0–1 axes; no examples were
selected. Below are every fit/arm point mean, without conditional intervals or additional contrasts.

| Fit | Arm | Temporal focal | Temporal repeated | Temporal pooled | Lead focal | Lead repeated | Lead pooled |
|---|---|---:|---:|---:|---:|---:|---:|
|1|B|0.170262|0.280890|0.225576|0.326002|0.316894|0.321448|
|1|P|0.379594|0.506581|0.443087|0.449483|0.510027|0.479755|
|1|K|0.115136|0.237639|0.176388|0.333953|0.294149|0.314051|
|1|PK|0.252909|0.358120|0.305515|0.385228|0.392503|0.388865|
|2|B|0.152097|0.295418|0.223757|0.355498|0.332684|0.344091|
|2|P|0.472350|0.487615|0.479983|0.453533|0.427394|0.440464|
|2|K|0.194492|0.299903|0.247197|0.442647|0.341935|0.392291|
|2|PK|0.270854|0.409994|0.340424|0.451489|0.407840|0.429665|
|3|B|0.155131|0.284773|0.219952|0.323371|0.289375|0.306373|
|3|P|0.390300|0.492482|0.441391|0.350470|0.352058|0.351264|
|3|K|0.132596|0.224208|0.178402|0.387218|0.322357|0.354788|
|3|PK|0.264772|0.403656|0.334214|0.426688|0.403260|0.414974|

PK pooled oracle temporal AP ranges from 0.305515 to 0.340424 across the three fits. P has the
highest temporal means in every fit, alongside the previously documented larger control peaks.
K has variable temporal means despite changes in lead ranking. These observations do not justify
choosing an arm or retuning coefficients.

Fit 1 illustrates why endpoints must remain separate: its repeated-edit full-grid PK-minus-B AP
was −0.008928, while oracle temporal AP improves by 0.077230 and lead-presence AP by 0.075609.
Different ranking domains and weights can disagree. The preceding [four-arm study](ssl-synthetic-localization-paired-suppression-v1-results.md)
still FAILS its frozen 15-gate conjunction (14 passed); this diagnosis neither rescues that verdict nor
identifies a causal explanation for its full-grid gain.

The six intervals use exactly 2,000 crossed draws from `default_rng(2026105601)`: first three fit
rows, then one shared 64-background vector, common to every contrast. Linear 2.5/97.5 percentiles,
no skipped draws and no conditional intervals. With only three fits these intervals are exploratory.
The audit protocol's shorthand “5601” refers to the seed suffix, never a count of 5,601 draws.

## Preserved failure, prospective recovery and exact closure

The [original diagnostic protocol](ssl-synthetic-localization-temporal-diagnosis-v1.md) was committed
as `ec86255` before reading the parent study's scores. It saved all 1,536 case rows, but its strict
range check stopped complete aggregation because one case temporal AP was 1+1 float64 ULP.
Five individual-lead values exceeded one by at most two ULPs. All were finite. The failed attempt,
its original source, charged costs and [independent cause audit](ssl-synthetic-localization-temporal-failure-audit-v1-results.md)
remain intact. Transient B means existed before its P guard; no completed paired contrast or
bootstrap was produced. See the [retained failure report](ssl-synthetic-localization-temporal-diagnosis-v1-results.md).

The separate [recovery protocol](ssl-synthetic-localization-temporal-diagnosis-v2.md), config and
clarification were committed as `611171d`/`1ab7bc2`, then identity-only admission as `ebe4124`,
before recovery aggregation. Reviewed source `2fcd8b7` is byte-identical to `853a6fc`; thirteen
artificial tests passed. An eight-float64-epsilon admission envelope retains every original scalar
unchanged. There is no clipping, dropped case, recomputed endpoint or changed bootstrap/comparison.
The complete `cases.jsonl` was copied byte-identically and checked before aggregation and after use.

The recovery main only used saved rows and metadata, hashing the 29 B files and six failed-stage
files for identity. It did not numerically reopen the B maps/support. It exactly reproduced the
complete B aggregate against both B summary and v1 inherited closure, including all 15 gates and
42 secondaries. The inherited v1 1,536-full-grid-AP closure is explicitly source-bound evidence.

The independent recovery checker freshly reconstructed all 1,536 full-grid APs and all 1,536 new
endpoint rows directly from the admitted B maps/support using separate endpoint arithmetic. It
passed 129 bindings and 16,758 metric checks with zero mismatches, including every
active-lead value, means, six intervals, ranges, raw-byte identity, fixed figure and additive costs.
Twelve artificial independent checks passed before its one copied execution. The original full-v1
checker remains committed and unexecuted; it was not substituted for either actual audit.

## Resources, local artifacts and identities

Main terminal CPU was 7.534621212 s, wall 20.681709279 s and GPU zero; peak RSS 547,774,464 bytes.
Final main storage was 969,761 bytes, below 16 MiB. Caps were 90 CPU /120 wall seconds.
Audit terminal CPU was 10.989279882 s, wall 12.289121443 s and GPU zero;
audit CPU, wall and new-artifact storage met the 180 CPU /240 wall second and 16 MiB caps.
Peak audit RSS was 938,332,160 bytes, reported separately from artifact storage.
Main plus audit CPU was 18.523901094 s. Including the retained failed attempt and its separate
audit once, all eighteen current-machine stages total 429.49696094999996 CPU seconds and
447.98139213304967 GPU-lock seconds. Unknown historical costs remain unknown. Receipt/prior
hashing and byte enumeration precede terminal timestamps; ledger assembly/serialization is the
disclosed reserved bookkeeping tail. Development, setup checks and package tests are separate.

All artifacts remain local beneath `outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v2/`:
unaltered `cases.jsonl`, complete points/contrasts/ranges, roundoff admission, predecessor closure,
source snapshots, fixed `lead_and_time_diagnostics.png`, main/audit summaries and terminal ledgers.
No new waveform, noise, model/optimizer tensor, clinical record or other panel was read; no neural
operation, GPU use, download or external API call occurred. Audit reads only admitted prediction
and support arrays for numerical reconstruction. No signals, maps, weights or figures enter Git.

| Identity | SHA-256 |
|---|---|
|cases.jsonl|`a48bf0672cc4b8c77034d2ea6373483a7e907ab4bc1e371ab38a026099917cdf`|
|summary.json|`e6897dacd5f4a784ea5a5110af8ce71d2de248f9de1bdf715b36963194fbc3b7`|
|prospective_ledger.json|`cc59342a29c56fd339f836ef7682bbb29abc11c8b8b486f94cf5da1dd6b0a9e8`|
|independent_audit_v1/summary.json|`d256bb1290a7edca693492d4f24620ed3513c3b210d03d95588664a9412a11d9`|
|independent_audit_v1/prospective_ledger.json|`a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf`|
|ecg_experiment/ssl_synthetic_lead_time_recovery.py|`1810d42d5d0d2bba677b5624fbe9921b97c80591f5f99578cd10c7d5e1bb2892`|
|ecg_experiment/ssl_synthetic_lead_time_recovery_run.py|`4a4281cd99c16fb162cf5a53ec755236c55b14f77a0eb07e9bd191d7958785e4`|

The underlying seeds remain 2026105000–2026105063 and indices 2000–2063; fit initialization
2026103801–3803 and schedules 2026103901–3903. No new seeds or trained states were selected.
The parent's locally loaded public Pulse2Pulse checkpoint remains
`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`, Hub revision
`262e0e67c0101cd4989d7c2f66f16d6d54756e70`, architecture revision
`f28e4b792e79c503b8b7e0912be9d304a3a7177b`. This diagnosis does not rerun the generator.

The [independent recovery report](ssl-synthetic-localization-temporal-recovery-independent-audit-v2-results.md)
describes its actual verification. Generated ECG shape, engineering checks, disease control, exact
introduced numerical support and real disease localization remain distinct. Physiological validity,
clinical masks, accurate disease delineation and benefit from real-ECG self-supervision are not
established here. Closed tests and clinical labels remain unchanged. Public waveform-annotation
qualification is ranked next; no specialist recruitment is required for existing released wave
annotations. Any new study still needs its own prospective protocol. PR #67 stays draft.

## Final repository validation

The integrated CPU suite passed 1,919 tests, with six skips and two existing all-NaN-fixture warnings,
in 136.03 seconds. Repository-wide Ruff and offline source/wheel builds pass. Archive inspection
found all new research modules and zero raw data, outputs, weights or prediction artifacts.
All 223 checked current-machine source obligations (95 distinct paths) match their frozen receipts.
Local setup evidence is `outputs/ssl_synthetic_localization/autonomous_repository_validation_v1.json`;
its logs and package archives remain local and are separate from scientific accounting.
