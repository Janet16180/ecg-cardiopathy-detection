# Frozen localizer attraction to released ECG landmarks

Prospective protocol, 21 October 2026. Continue draft PR 67. The user's intended
localization is suspicious time segments and leads in an ECG, not an anatomical
lesion map. This bounded diagnostic asks whether existing synthetic-trained maps
concentrate at released beat landmarks on real development ECGs. It introduces
no clinical labels, disease masks, training, generator forwards or closed tests.
Commit this protocol/configuration before new outcomes; commit complete main,
independent-audit source and executable asset admission before actual execution.

Use every locally present paired-suppression checkpoint: fits 1,2,3 and arms
B,P,K,PK, without selecting a winner. They are the unchanged observed-only
Conv1d/ReLU/Conv1d/ReLU/Conv1d network from
`ssl_synthetic_randomized_localization.build_model`, with float32 sigmoid maps.
File-byte hashes and tensor-state digests are separate identities. A missing or
mismatched checkpoint is a technical stop, never a substitution from another
checkout. Preserve the earlier 14-of-15 scientific FAIL and all historical costs.

Use all 160 LUDB 1.0.1 measurement IDs not divisible by five, sorted numerically.
The 40 parser IDs remain nonscored. This is previously inspected development
material, not a new untouched clinical test. Verify all signal/header bytes
against the admitted official release table. Read physical signals only through
the existing structural-header reader and local read-only-use alias. Use the
already verified landmark rows; do not reread diagnosis comments, auxiliary text
or original annotations. Retain the first-through-last timed-event span and all
explicit numeric-code-1 N timestamps with their original multiplicity. Do not
infer missing wave boundaries, call N healthy, or label other samples normal.

For inference select physical I,II,V1..V6, cast to float32, apply the unchanged
per-record eight-lead median/q95-minus-q05 standardization, then derive the
canonical twelve leads with the existing helper. No fitted preprocessing or
annotation-dependent transformation. Reject nonfinite signals or inadmissible
scale rather than exclude records. Save transformation parameters and all model
inputs. Score only I,II,V1..V6 outputs against their own released annotations;
derived limb outputs remain saved but have no landmark metric. Their algebra
after independent I/II standardization differs from native released limb leads.

For each native output lead crop its probability map to the fixed unlabeled
annotation span of length L. Empirical midrank is
(count strictly lower + half count equal)/L, including the landmark itself.
Average ranks at all retained N events. A constant map gives exactly 0.5.
Controls circularly roll the cropped rank vector by floor(L*f), f in
[.17,.31,.43,.59,.71,.83], retaining landmarks and score distribution. Require
six distinct nonzero offsets. Report native mean, all six controls and gain
(native minus six-control mean), without thresholds or clipping. These controls
can retain periodic alignment and are not a universal null or causal control.

The one primary descriptive estimand is B's landmark gain: equal N weighting
within each lead, equal eight-lead weighting within a record, equal three fixed
B-fit weighting within a record, then equal weighting of 160 records. Bootstrap
whole records with shared patient_groups/patient_resample, 2000 default_rng draws,
seed 2026106501, linear 2.5/97.5 percentiles and zero skipped draws. Use the same
record draws for every prespecified arm and paired PK-minus-B secondary contrast.
Fits are fixed, not resampled: uncertainty conditions on these three checkpoints,
this cohort and its annotation provenance, not a training/patient-population claim.

Prespecified secondaries show every fit/arm and native lead, each arm's gain,
PK-minus-B gain, and map association with the unchanged 20-sample squared-
derivative activity proxy computed from the exact full standardized native model
input, before cropping to each annotation span. Report Spearman correlation and
its mean six-roll control, using average ranks for ties. If either vector is
constant, retain the row, emit null correlation and explicit reason/count; defined-
only correlation summaries state their denominators, never convert null to zero.
Also report strict within-span linear-q90 activity/map-mask Jaccard and its six
rolls. Empty union contributes zero with an explicit flag. These are descriptive
activity associations, not disease explanations. Also retain raw probability
mean, linear q95, maximum and constant-map counts within every native span;
rank/mask association is scale insensitive and cannot quantify alarm severity.
These probabilities are uncalibrated and have no clinical threshold.
No additional endpoint selection,
threshold tuning or multiplicity-adjusted confirmatory claim.

Before new scores, exactly reproduce saved all-twelve synthetic full-grid AP
rows and the parent's full aggregate/fifteen-gate verdict from its immutable
maps/support and unchanged evaluation helpers; close timing26/28 and coherence29/30
statistics and bootstrap draws from saved outputs. Replay fixed synthetic
background 2000, all four variants, through each admitted checkpoint on CUDA;
require finite float32 maps and absolute difference <=1e-6 (rtol=0) from the
saved corresponding maps, with frozen deterministic/TF32 policy. Record every
maximum difference and tensor-state digest. No optimizer or generator call.

Use CUDA device0, one CPU/inter-op thread, eval/inference_mode and the shared
nonblocking GPU lock. Before full inference, perform a small actual CUDA tensor
operation and an actual-path batch-eight profile on fixed IDs1,2,3,4,6,7,8,9
with all twelve checkpoint loads/forwards, input assembly, output persistence,
hashing and a fixed figure. No new landmark/activity outcomes before admission.
Profile maps are retained and compared exactly to the full first batch; the
eight records remain in the 160-record cohort. Main inference has twenty batches
of eight per model. The conservative full-path forecast is twice twenty profile
batch-path times plus 180 seconds of CPU/wall overhead; include remaining replay,
closure, final persistence, figures and hashes, plus observed setup elapsed time.
GPU forecast uses measured locked paths with the same factor and overhead.
Forecast output size includes all 460,800,000 map payload bytes plus inputs,
proxies, profile outputs, rows, figures and a 64-MiB reserve. All forecasts and
actual hard gates must pass; preserve a failed profile without new full scores.

Main hard caps: 900 process CPU seconds, 900 wall seconds, 240 GPU-lock seconds,
2 GiB peak RSS, 768 MiB retained new files, 2 GiB maximum allocated CUDA memory
and at least 2 GiB free device memory at admission. Independent audit caps:
900 CPU,1200 wall seconds,2 GiB RSS,128 MiB new outputs,zero GPU. Guard resources
between bounded batches/stages; report any overrun as failure. Reserve a disclosed
two-second/16-KiB terminal serialization tail after receipt hash and byte count.
No wait for a busy GPU: record the concrete reservation and resume later under
the unchanged admission, without claiming a new completed experiment.

Append actual stage30 ledger SHA
`a3acf4aae3f27b00d51b18e6e50921c9935267e0d8646433bf486ec3f274e11f`,
CPU460.93105389699997 and GPU-lock447.98139213304967, exactly once. The next
actual main attempt is stage31; its audit or any failure successor appends the
latest actual ledger, never resets cost. Include profile/replay/failed numerical
work in measured stage costs. Setup/source editing/tests are separate and do
not certify unknown historical costs. Keep original failed27/nulls and its
explicit setup reconciliation unchanged.

Save complete per-record and per-lead twelve-model rows, all probability maps,
model inputs, energy scores, bootstrap samples, source/checkpoint/data identities,
actual resource measurements, predecessor closures and fixed figures locally.
Figures: all-fit/all-arm landmark gains and paired controls, fixed activity
associations, and model-input/heatmap/landmark overlays for IDs1,2,3,4, leadsII,V2,
fit1 with all four arms and common probability axes. No outcome-selected examples.
Independent arithmetic must reconstruct metrics from saved maps and physically
decode inputs independently, check all twelve state identities and all draws.
Independently replay real measurement IDs1 and199, all twelve states, using
functional CPU convolution/ReLU/sigmoid on the six admitted tensors. Predeclare
absolute tolerance1e-5,rtol0 against saved CUDA maps, accounting for cross-backend
floating-point arithmetic; report maxima without relaxing tolerance after scores.
Write output-derived main and independent reports and update both queues/backlog.

There is no scientific-success or clinical PASS threshold for this diagnostic.
Technical completion requires source/asset identities, all records/models, exact
predecessor closure, replay tolerance, finite or explicitly undefined metrics,
resource gates and independent numerical agreement. A positive gain demonstrates
landmark attraction; it does not show that maps mainly represent generic activity,
identify pathology or fail to respond to edits. A low gain does not establish
clinical specificity. Any subsequent intervention or held-out-editor learning
study needs its own prospective protocol and actual-path gates.
