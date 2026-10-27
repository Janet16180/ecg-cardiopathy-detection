# Frozen localization across ECG backgrounds and numerical editors

Prospective protocol, 21 October 2026. Continue PR67's public-generator/localization
research locally. The user explicitly requests GPU acceleration and no pushes to
avoid GitHub runner use. Commit this protocol and configuration before new scores;
commit reviewed executable sources and identity admission before execution. Keep
PR67 draft. This is a bounded development diagnostic, with no training, generator
forwards, clinical label changes or closed-test access.

## Question and fixed design

Can an existing synthetic-trained, signal-only localizer find introduced changes
on real ECG backgrounds when their waveform shape was absent from its training?
The source-by-editor comparison separates two descriptive transfer dimensions. It
cannot identify a causal source effect or validate regional disease.

Use every existing paired-suppression checkpoint: fits1,2,3 and B,P,K,PK. The primary
arm is B, chosen prospectively rather than from the new scores. All other arms are
reported without winner selection. Preserve the earlier14/15 scientific FAIL.
The architecture, weights, float32 sigmoid outputs, 12-lead representation and
all inference settings remain frozen.

Use all64 paired-study generated backgrounds, global indices2000--2063, released
generator seeds2026105000--2026105063. Their fitting schedules use only original
backgrounds0--79. The64 backgrounds have already been evaluated, so they are spent
development material, not a new untouched test. Use the first64 ascending LUDB
measurement IDs not divisible by5 (IDs1--79 with multiples of5 excluded), from the
already independently verified160-record input bank. The remaining96 measurement
and40 parser records are not scored. Selection uses IDs only, never model outcomes.

Pair sources by64 numerical slots, with the same nuisance and original editor
random draws at global index2000+slot. These are geometry pairs, not matched people.
Real ECGs may already have findings; neither source is labelled healthy. Use saved
inputs only: no raw files, reports or diagnosis comments need to be reread.

Generated native float32 I,II,V1--V6 signals receive the unchanged per-record median
and q95-minus-q05 scaling. Real inputs already received that transformation: select
their native canonical positions[0,1,6,7,8,9,10,11] without scaling again. Derive
the remaining leads with the existing helper, require exact reconstruction of the
real canonical input and the first four original generated variants. Perform no
standardization, filtering or clipping after the edits.

## Six observed variants and exact numerical truth

The order is sham, nuisance, Hann focal, Hann repeated, biphasic focal, biphasic
repeated. Sham is the standardized unedited canonical signal. Nuisance is the
existing0.1-amplitude0.33Hz sinusoid added to every independent lead with the exact
original RNG phase. Both controls have zero intervention target; that means no
added target editor, not a medically normal signal or a mathematically unchanged
waveform. Each positive is added to the matched nuisance baseline.

Reuse `ssl_synthetic_randomized_localization.edit_parameters(2000+slot)` unchanged:
independent lead, sign, amplitude[0.4,0.8), width61/81/101 and focal/repeated centers.
The original family is `sign*amplitude*h`, where h is NumPy's Hann window of widthW.
The held-out family is constructed in float64 from j=0,...,W-1:

    q = h * sin(2*pi*j/(W-1))
    g = q * sqrt(sum(h*h) / sum(q*q))
    bump = (sign * amplitude * g).astype(float32)

Add the bump using the same float32 arithmetic and locations as the original family,
then derive all12 canonical leads. The new kernel is biphasic, nominally zero-DC and
matched in pre-addition L2 energy to the original kernel. It does not match peak
amplitude or thresholded support size. Save measured kernel means, peaks, L2 norms,
post-addition change energy, strict and thresholded support size. Do not substitute
another shape or tune amplitudes from outcomes.

For each positive, strict support is final float64 observed-minus-nuisance !=0;
the scoring target is abs(that difference)>0.01. Save both masks, canonical equations,
affected-lead/time hull checks, retained change-energy fraction and residual maxima.
No intended window, N timestamp or diagnosis becomes a pathology mask. Model inputs
contain only observed ECGs, never nuisance differences, masks or editor parameters.

## Endpoints and decision

Score all60000 canonical lead-time cells. For every record, fit, arm, family and
mode, save direct edited average precision(AP), nuisance-site AP using the same
positive mask, their response difference, absolute observed-input amplitude AP,
support prevalence, probability-change means inside/outside support and selectivity.
Also score each edited map against the actual final-difference target of donor slot
(slot+1)%64 in the same source/family/mode. Use the fixed nonself permutation without
overlap filtering or retries. It preserves the cohort's support-count distribution
while breaking case correspondence; it is an alignment control, not a universal null.
No empty mask is accepted; do not silently skip a case. Save sham/nuisance mean,
q95, maximum and mean top600 probabilities, with fixed stable sorting when needed.
Raw probabilities are uncalibrated engineering outputs, not disease risk.

Primary population is REAL background / HELD-OUT biphasic family / fixed B arm.
Average the two modes equally within each record, the three fixed fits equally,
then all64 records equally. The narrow numerical-transfer conjunction requires:

1. Edited-minus-nuisance AP response mean>0.
2. Its conditional95% lower interval bound>0.
3. Direct edited AP minus absolute observed-input amplitude AP mean>0.
4. Its conditional95% lower interval bound>0.
5. Direct edited AP minus donor-mask AP mean>0.
6. Its conditional95% lower interval bound>0.

All six must pass, along with technical integrity/resource checks and independent
verification. There is no added minimum effect margin or clinical threshold.
Failure remains failure; success establishes only this introduced-change comparison.
Report each focal/repeated and fixed-fit value, including unfavorable ones.

Use2000 whole-slot draws from default_rng(2026106601), using the shared
`intervals.patient_groups` and `patient_resample` helpers. Each draw samples64 slots
with replacement and retains both sources, both families, both modes and all three
fixed fits at every selected slot. Fits are not resampled. Percentiles are linear
2.5/97.5 with no skipped draws. The interval conditions on these three states,
development records, one released generator and the declared geometry pairing.
It is not a population disease interval or an interval for new training runs.

Exploratory reporting covers all16 source/family/arm cells, primary components,
direct AP, amplitude AP, response, selectivity, original-minus-held-out and
real-minus-generated contrasts within every arm, plus the source-by-family
interaction. Controls remain separate. No secondary result rescues a failed primary
or selects an arm. Differences can reflect background morphology, support geometry,
editor shape/peak and representation; no single causal mechanism is established.

## Identity, closure, GPU profile and limits

Before scoring, bind complete new sources/config/protocol, frozen imported helpers,
all12 checkpoints, generated raw/observed/support/cases and real input/ID/metadata
files, predecessor summaries/ledgers and required audit evidence by path and SHA256.
Use repository-relative stored paths and no dependency changes. Preserve original
source snapshots and all costs. Reproduce the preceding paired-study full aggregate
and15-gate verdict exactly from saved rows, and the real landmark diagnostic's
primary values from saved rows with its frozen aggregate helper. Verify latest
successful independent stage34 and cumulative accounting; don't fabricate old costs.

Actual GPU access was confirmed independently by a small locked CUDA matrix
operation: two RTX3090s, PyTorch2.6.0+cu124. This is setup, not an ECG experiment
result. Main execution uses CUDA0, deterministic operations, TF32 disabled,
CUBLAS_WORKSPACE_CONFIG=:4096:8, and one numerical/intraop/interop CPU thread.
Use the shared nonblocking `ecg_experiment.gpu.gpu_lock('cuda')`. A busy lock stops
the attempt safely; do not overlap another GPU job or use CPU fallback.

Profile the first8 slots from each source with all6 variants/all12 states using the
same batch8 inference, CPU transfer, full float32 memmap persistence/flush and scoring
paths as the full experiment. Retain these profile arrays and timings. Include
actual model loading and fixed overhead, not just warmed tensor timing. Require exact
profile/full first-slot replay and unchanged state identities. Before new full-panel
inference, use a conservative factor2 times8 recurring profile phases plus spent
time and fixed reserves180CPU/240wall/15GPU-lock seconds; no timing samples discarded.
Bound profile scores are allowed only after this committed protocol. Reproduce a
fixed parent map batch for every state at tolerance1e-6; CPU independent replay has
separate1e-5 tolerance. No optimizer or generator execution is authorized.

Main hard caps: CPU1800s, whole-wall2400s, cumulative GPU-lock300s, process RSS2GiB,
peak CUDA allocation2GiB, new outputs3GiB. Require2GiB free device memory and4GiB free
output disk. Forecast all full/profile map/input/mask bytes plus64MiB reserve before
bank allocation, and apply live guards throughout. Full12-model maps total
2211840000bytes; profile maps add276480000bytes. Existing outputs stay preserved;
the new storage budget does not retroactively alter any old study's limit. Release
the GPU lock before long CPU scoring. Limit live arrays/mappings so peak RSS stays
within the actual guard. Never delete partial outputs or retry inside a failed run.

New exclusive output directory is
`outputs/ssl_synthetic_localization/background_editor_transfer_v1/`. Save full
observations, masks, geometry, record IDs, all12 probability banks, complete metric
rows, all2000 bootstrap draws, fixed examples(first slots, no selection), aggregate
figures, profile/replay evidence, file/tensor identities, operation counts, terminal
summary and cumulative ledger. Ledger starts at actual34 stages,759.933143906CPU
and473.0495132012293GPU-lock seconds. Add main and independent audit costs once,
including failed attempts; setup/development/test costs remain separately described.

Independent audit is separately prospectively committed and source-admitted before
execution, with CPU600s/wall900s/RSS2GiB/new outputs64MiB, zero GPU, no new clinical
reads and exactly96 fixed model-record CPU replays: both sources, slots0 and63,
variants Hann focal and biphasic repeated, and all12 states. Reconstruct inputs, kernels,
all strict/threshold masks, all per-record endpoints, bootstrap draws, every
aggregate/gate and profile/full identities from saved files independently. Require
zero mismatches, while allowing the declared CPU/CUDA tolerance for replay only.

The coordinator writes results from actual outputs and updates both queues and the
backlog. New sources remain immutable after execution. Failed forecasts, failed
scientific gates and older missing artifacts remain visible. No fine-tuning or next
experiment is implied by technical completion. Keep all ECGs, weights and individual
predictions local. Commit locally as Janet16180 <clarajanet16180@outlook.com>; do not
push, trigger GitHub Actions, merge PR67 or publish new individual-record material.
