# NTT-04: reference discrimination among same-wave record findings

Prospective CPU protocol, 18 October 2026. NTT-03 selected supplied wave windows but tied a generic
QRS-phase comparator. Here every target uses an automatically detected QRS region. The endpoint is
record-finding classification among CRBBB, CLBBB and existing strict-clean NORM source annotations,
not etiologic diagnosis, clinical localization or precise pathological highlights. No QA answer, provided
window, target lead answer, clustering assignment or SHAP result enters selection or prediction.

## Reservation and cohort before scoring

Verify the exact shared PR67 manifest and policy bytes from CLUSTER-01's retained copies, metadata SHA256,
and the independent train-role rule: SHA256(`ssl_synthetic_partition_v1:20261018:{patient_id}`), integer
digest modulo 10,000 below 8,000, excluding every patient touching folds 9/10. Require exact agreement
for all 13,938 training ECGs / 12,078 patients before diagnosis parsing. Read only identity/path columns
outside that permitted set. No broad NTT-01 label replay, Challenge/EchoNext/ECGAD/MIMIC/SPH access,
new data download, age/sex analysis or clinical label redefinition.

Choose each permitted patient's lowest ECG ID before parsing diagnoses. Among those records, CRBBB or
CLBBB requires the corresponding existing SCP code confidence >=50. Exclude records listing both complete
BBB codes, regardless of confidence. NORM requires the existing `project_label == 0` helper; other records
are outside this three-finding task. Retain existing labels and report diagnostic-superclass co-occurrence
descriptively. These are source human-read annotations, not newly adjudicated specialist truth.

Split patient roles by SHA256(`ntt04_pool_v1:20261018:{patient_id}`) modulo 10,000: below 5,000 support
pool, otherwise target pool, all within PR67 train. For each finding choose 20 target patients ordered
by SHA256(`ntt04_targets:20261018:{patient_id}`). For each of seeds 20261018/19/20 choose three support
patients/finding ordered by SHA256(`ntt04_support:{seed}:{finding}:{patient_id}`). Draws may reuse support
patients, but no target can be a support in any draw. Choose before waveform extraction/model scores.
Require enough patients prospectively; fail clearly otherwise. No score-driven class selection.

## Same automatic representation for every class

Read native 500 Hz physical mV signals, canonical lead order. Reuse the existing all-lead `qrs_peaks`
detector. Require at least three complete beats for its fixed [-0.30,+0.65) second median-beat window.
Band-pass 0.5–40 Hz and call the shared `median_beat`; subtract each lead's median-beat mean over
[-0.28,-0.20) seconds. Use the shared `qrs_bounds` estimate for duration. These are heuristic estimates,
not validated clinical delineations. If any selected record fails quality/extraction, preserve the failure
and stop before classification; do not replace it or search a different detector/window.

Render the entire fixed median beat on a 518x518 white canvas: V1 in the top half and V6 in the bottom,
two-pixel black lines, common time scale [-0.30,+0.65), fixed +/-3 mV per panel, no text or disease cue.
All classes receive exactly this lead pair and geometry. Pool trace ink only inside the programmed QRS
ROI [-0.15,+0.15] seconds using NTT-03's frozen soft-occupancy patch pooling, then unit-normalize. Use
cached frozen DINOv2-S/14 local public weights/source only; no SAM or new weight/representation training.
Average three independent support vectors per finding into unit prototypes; classify each target by
maximum cosine. Ties follow finding order NORM, CRBBB, CLBBB.

## Fixed controls with identical support draws

Compare DINO QRS matching with raw physical morphology: concatenate V1/V6 samples in the identical
[-0.15,+0.15] ROI, unit-normalize, average supports into unit finding prototypes, cosine classification.
Prespecified scalar comparators use class-wise median support descriptors and nearest Euclidean distance:
log QRS duration, log peak-to-peak QRS amplitude in V1/V6, log QRS derivative RMS energy in V1/V6, and
their five-dimensional concatenation. All are log ratios in fixed units with floor 1e-8; no fitted scaler
or target normalization. Duration uses all-lead automatic bounds, a deliberately strong contextual
descriptor, while raw morphology and visual features share V1/V6 inputs.

Late-wave reference control: same rendered image and exact 300 ms mask width, but support vectors pooled
inside [+0.25,+0.55] seconds, while targets retain QRS pooling. It is a phase-mismatch control; late waves
can carry correlated finding information and are not known clinical negatives. Shuffled-label control:
cyclically rotate the QRS support-bank labels (NORM -> CRBBB -> CLBBB -> NORM), target labels unchanged.
No alternative masking/draw/permutation is tried after scores.
The frozen DINO tokens are globally contextual, so late-region pooling also does not guarantee removal
of QRS information from its representation. Neither programmed ROI constitutes an expert evidence mask.

## Primary comparison and stopping rule

Primary endpoint is three-class balanced accuracy, averaging each target patient's correctness over
the three frozen support draws. Compare DINO with the **best point-performing prespecified baseline**
among raw morphology, duration, amplitude, energy and concatenated scalars. Report every method, per-class
recall/confusions, all draws, support reuse, quality/clip descriptors and both controls.

Use shared whole-patient grouping/resampling for 2,000 paired bootstrap draws, seed 20261018. Resample
all target patients together; skip/count draws missing any class. In each resample recompute balanced
accuracy for every method and compare DINO against the maximum baseline accuracy in that same resample,
so the interval preserves baseline selection. It conditions on the fixed support banks and detector.
Advance only if observed gain >=0.05 AND the bootstrap 95% lower bound exceeds zero. A pass would admit
a new independent reference study only, not clinical validity. No tuning after the initial result.

Reconstruct NTT-03's six-target saved-score counts and three proposal oracles from its frozen outputs
before new scores, never rerunning models or interpreting unselected beats as false pathology.
Commit this protocol before scores and the implementation before execution. Use existing venv, two
CPU threads, no GPU/dependency change/push/PR; target <=10 minutes wall time, abort at the next image
boundary if exhausted. Store input/source/checkpoint hashes, patient/reference manifests, features,
all predictions, aggregate JSON and fixed first-example/aggregate plots under `outputs/ntt04_same_wave/`.
Identifiers, raw traces and exemplars stay local; only aggregate plots/reports are committed. Preserve
failed runs. Root owns both queue documents and ranked follow-up backlog; no follow-up is run here.
