# Frozen synthetic-trained maps do not uniformly concentrate at real beat landmarks

21 October 2026. The complete GPU-main diagnostic ran on all160 fixed LUDB development records
and all twelve B/P/K/PK checkpoints. Its original synthetic maps replay exactly, every predecessor
closure and main resource gate passes, and all maps/metrics are saved locally. Independent
recovery34 reconstructs the complete result with zero mismatches after preserving two earlier
audit I/O failures. All final resource gates pass; there is no scientific or clinical PASS rule.

The fixed B primary is negative: native-minus-shifted N-landmark mean rank
−0.04813152337635353, conditional whole-record95%interval
[−0.08043986863626744,−0.01796850935554083]. This does not support positive attraction of this
baseline's output to the released N timestamps under the fixed phase controls. It is not disease
localization, clinical specificity, an alarm-rate measurement or evidence that all beat-related
structure is ignored. This diagnostic has no scientific/clinical PASS threshold.

|Frozen arm|Native N mean midrank|Six-roll mean|Gain|Conditional95%interval|
|---|---:|---:|---:|---|
|B|0.449222|0.497353|-0.048132|[-0.080440, -0.017969]|
|P|0.511124|0.500861|0.010264|[-0.007767, 0.028276]|
|K|0.291054|0.496466|-0.205412|[-0.233924, -0.179498]|
|PK|0.381735|0.499431|-0.117696|[-0.145696, -0.091604]|

B is unchanged BCE; P adds the paired response/outside-support consistency penalties, K adds
control-peak suppression, and PK adds both. These are all fixed checkpoints, with no best-arm
selection. The prespecified PK-minus-B secondary is−0.06956443116899283,
interval[−0.09942025090541526,−0.042412938219429944]. A lower landmark gain is not automatically
better or worse clinical localization. N is a released timing marker, not a healthy label.

## Activity association and uncalibrated output magnitude

|Arm|Energy Spearman|Mean shifted Spearman|q90 Jaccard|Mean shifted Jaccard|Mean probability|Mean lead q95|Maximum probability|
|---|---:|---:|---:|---:|---:|---:|---:|
|B|-0.002185|-0.004564|0.077476|0.056229|0.102920|0.238298|0.980110|
|P|0.004677|-0.000902|0.076798|0.056046|0.122199|0.373274|1.000000|
|K|-0.216717|-0.008695|0.032358|0.055317|0.041498|0.085826|0.766496|
|PK|-0.090509|-0.004702|0.038287|0.057167|0.052342|0.151677|0.930526|

All3840 native lead/model-record spans per arm have defined correlations; there are no constant
maps or empty native map/energy unions. Correlations, ranks and q90 masks ignore output magnitude,
so raw probability summaries are retained. Probabilities remain uncalibrated: the maximum is one
extreme across the complete arm, not an event frequency, referral rate or disease probability.

The baseline's near-zero whole-span rank correlation coexists with modest upper-decile overlap
above its phase controls. The suppressing arms have negative whole-span energy associations and
native top-decile overlaps below their respective shifted controls. These distinct summaries cannot be collapsed into a
claim that every map follows or avoids an entire QRS complex. In particular, a released N point
is neither a complete wave interval nor an annotated suspicious region. Lead and fit differences
remain in the full saved15360-row decomposition and the all-twelve-model figure.

The earlier generated-background hotspot association cannot simply be assumed to have the same
interpretation on this real cohort and representation. Cohort/device/morphology differences,
fixed training objectives and unknown generator-training membership remain possible contributors.
This descriptive study identifies behavior to investigate; it does not causally explain previous
localization failures or establish real disease masks. The previous scientific15-gate rule stays
FAIL14/15 regardless of these measurements.

## Prospective method and integrity

The protocol/config were committed asb829ad1, the fixed/repeated resource-profile amendment as
71d740c, main source ascaf0890 (same bytes as integrationd4cd911), and complete executable admission
v2 as8fb0871 before any new inference. The admission binds160 source files,58 input artifacts and
400 official signal/header identities. Both independent and main metadata preflights passed with
zero new signal reads/scores. Source/test work is setup, not a new numerical cost stage.

All160 IDs not divisible by five are scored; all40 parser IDs remain nonscored. Already verified
saved annotations retain every original N timestamp/multiplicity and first:last timed-event span.
No original annotation stream, clinical diagnosis comment or auxiliary report was read. Physical
signals pass through structural-only headers. Native I,II,V1..V6 are castfloat32, robustly centered/
scaled with the unchanged training representation, then canonical12 are derived. Only the eight
native outputs are scored; all twelve outputs are retained. This dimensionless input path does
not establish physical voltage calibration or deployment preprocessing.

The primary averages events within lead, eight leads and three fixed B fits within record, then
160 records. The six fixed within-span rolls preserve score distributions and may retain periodic
alignment; they are not universal nulls. All2000 shared whole-record draws use RNG2026106501,
linear2.5/97.5 percentiles and zero skipped draws. Fits remain fixed, so uncertainty conditions on
these checkpoints, cohort and annotation provenance. Saved per-fit/per-lead results and all arms
remain visible; no coefficients, thresholds or models were selected from these outcomes.

All1536 old full-grid synthetic AP rows, twelve old complete panel summaries and full fifteen-gate
aggregate reproduce exactly from saved maps/support. Timing26/28's original twelve-lead physical-
energy primary/controls/all2000draws and coherence29/30's328rows/all2000draws also close exactly.
They are separate predecessor estimands, not a claim that the new eight-native learned-map ranks
must equal the old energy qualification. Existing failed experiments and missing historical
payloads/costs are preserved rather than silently recovered from another checkout.

## Actual GPU operation, profile and resources

The shared nonblocking project lock enclosed an actual CUDA tensor operation, twelve checkpoint
loads, all twelve old background2000/four-variant replays, profile and full inference. All old
CUDA map replays have maximum absolute error0, passing the frozen1e-6 tolerance. Every profiled
batch-eight output is byte-identical to its full-inference first batch; all twelve named model
states stay unchanged. No optimizer update or generator forward occurred today.

The actual-path profile uses eight fixed records and the same memmap write/flush primitive as full
execution, including input assembly, persistence, hashes and its fixed figure. Its forecast is
233.689402033CPU/303.007854424whole-wall/102.560143624GPU-lock seconds and732471929outputbytes;
all forecasts pass before new scores. Fixed cold/replay/plot work is counted once, recurring
batch work twenty times, each with the frozen safety multiplier and explicit reserves.

Main31 costs:143.514671634CPU seconds,174.659184951wall seconds and25.068121068GPU-lock seconds.
RTX3090 CUDA0, torch2.6.0+cu124, free memory5618728960bytes at admission; peak allocated/reserved
63441408/75497472bytes. Peak processRSS1320816640bytes. Retained main bytes before terminal ledger
638739250. All900CPU/wall,240GPU-lock,2GiBRSS/CUDA,768MiBstorage caps pass. Numerical threads
and inter-op threads are1, deterministic operations enabled, TF32 disabled; other GPU services
were left untouched. Restricted-shell device invisibility was not treated as GPU unavailability.

There are12 synthetic replay batch-four calls,12 real profile batch-eight calls and240 full real
batch-eight calls; full maps cover1920model-records. There are168 physical-record reads including
profile rereads, zero parser scores, zero new clinical labels and zero closed-test accesses.

Actual main31 appends stage30 once, reaching604.445725531CPU/473.049513201GPU-lock seconds.
The independent attempt32 stopped on the older timing table's parser-first record order before
any new real-map scoring or CPU functional replay. Its37.394049274CPU seconds and immutable failure
remain, reaching641.839774805CPU with unchangedGPU. The separately committed first recovery sorts only
record objects for the same old aggregate; it does not reorder annotation events or alter outputs.
Actual33 then stopped on an object-only JSON reader incorrectly used for array-valued metadata,
after predecessor closure but before new localizer reconstruction. Its36.38738272CPU seconds
remain charged once. The narrow separately committed final recovery replaces only three array
reads, preserves every metric formula/tolerance and exercises actual metadata/array headers in
preflight. The unused originalv1 audit has no fabricated receipt or cost stage.

Successful independent34 reconstructs160physical inputs and their native energy, all1920model-
record/15360native-lead rows,1,978,070numerical scalar comparisons and all2000shared bootstrap
draws with zero mismatches. It also checks36,884predecessor scalars and2072identity bindings.
Twenty-four independent functional CPU forwards reconstruct fixed real IDs1and199 across all
twelve unchanged states; maximum CPU/CUDA difference8.344650268554688e-7 passes the frozen1e-5
absolute tolerance. Independent timing/coherence draws and old15-gate FAIL close exactly.

Independent34 uses81.705986381CPU/87.788420480wall seconds, zeroGPU and1040211968peakRSS bytes,
with2718111retained bytes before its terminal ledger. All900CPU/1200wall/2GiB/128MiB audit gates
pass. Including both failed attempts exactly once, all34current-machine numerical stages total
759.933143906CPU and473.0495132012293GPU-lock seconds. Development, setup, unit/build checks
and unavailable historical costs remain separate. No missing old receipt or cost was invented.
Read the independent [results](ssl-synthetic-localization-ludb-localizer-independent-results-v4.md),
[preserved ordering failure](ssl-synthetic-localization-ludb-localizer-independent-failure-v2.md),
[preserved container failure](ssl-synthetic-localization-ludb-localizer-independent-failure-v3.md) and
[scientific interpretation](ssl-synthetic-localization-ludb-localizer-output-science-review-v1.md).

## Local deliverables and scope

Under outputs/ssl_synthetic_localization/ludb_localizer_v1/: inputs.npy,energy.npy,record_ids.json,
input_metadata.json,all12probabilities/*.npy banks,model_rows.jsonl,bootstrap.npy,model_states.json,
all predecessor closures, profile/replay evidence,source/admission snapshots,summary/terminal
ledger, all_model_landmark_and_activity.png and fixed_input_map_N_overlays.png. The coordinator
visually inspected the aggregate figure. Public ECG records, actual waveform overlays, weights,
individual maps and per-record tables stay local and are not served by the advisor website.

The complete CPU suite at this main-source checkpoint passes2101tests/6skips,2existing all-NaN
fixture warnings,148.49seconds. Ruff and offline source/wheel builds pass. Archive inspection
finds all10new Python files/four package modules and no raw data, outputs, weights or prediction
arrays. The final recovery-source suite then passes2109tests/6skips,2existingfixturewarnings in101.27seconds;
repository-wide Ruff and offline source/wheel builds pass again after the new runtime files.

Main summary SHA256:e25e7be0b12774c485b56b5bff1664b5718d7efc25b18135eb67f45fd9ffd4fc.
Main ledger SHA256:50e558091eb1930e2a25b4b966f51879dcd385bef11ff0671436e856006292a8.
Independent34summary SHA256:e229edb9f0b410531e91ea2ad9170d38b10bde40f32f5bd3eb542edb53a5f942.
Independent34ledger SHA256:ccac6c1b21691705592cd6425107a2647aec7fcbcd7e082a4fac692eabaa864b.
Released generator ancestry remains Pulse2Pulse checkpoint SHA
e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827,
Hub revision262e0e67c0101cd4989d7c2f66f16d6d54756e70 and MIT architecture revision
f28e4b792e79c503b8b7e0912be9d304a3a7177b. No generator was executed in this diagnostic.

The useful next localization question is a separately prospective matched real/generated-background
intervention test with known numerical support and held-out editor families. It can test whether
learned response transfers beyond generated backgrounds while retaining control behavior. Such
edits remain numerical interventions, not verified pathology. Public disease-region annotations
are still required to validate suspicious ECG regions; no cardiologist recruitment, label change,
closed final test or automatic PR merge is authorized by this result.
