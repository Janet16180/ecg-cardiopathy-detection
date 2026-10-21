# NTT-03: compact visual reference-transfer pilot

Prospective protocol, 18 October 2026, before any real-ECG visual-model score. Test a compact implementation
of the paper's component idea: frozen visual patch features pooled inside marked reference regions,
instance-to-class averaging, candidate similarity, and automatic segmentation proposals. Use pretrained
DINOv2-S/14 and SAM2-Tiny, not the paper's large/large recipe. This is a CPU engineering variant, not a
paper benchmark reproduction or a validated clinical localizer.

## Already inspected

NTT-02 has completed its independent native candidate-choice experiment. Its selected cohort and labels
are development data. Schema inspection found single-lead links for several criteria; a lead answer in
one step does not automatically apply to all that step's wave questions (an LVH question may explicitly
refer to V1 while another answer selects V5). No real ECG has been processed by a visual model.

Official model sources are pinned: DINOv2 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, SAM2
`2b90b9f5ceec907a1c18123530e92e794ad901a4`; public checkpoint bytes are hashed before the run. A synthetic
line-image smoke measured DINO load+forward 1.024 s and SAM2 8x8 prompts 7.717 s on two CPU threads.
No real label or signal entered that check. The optional runtime overlay contains only iopath 0.1.10 and
portalocker 2.10.1; the project's pinned dependencies/environment are unchanged.

## Cohort and references

Reproduce NTT-02's 1,507-question inventory and stored output hash before selection. All input rows are
already PR67-training-role patients, inside PTB folds 1–8, with independent inner folds. Verify this again
against the saved reservation manifest. Do not read reserved patients, ECGAD holdout, Challenge/EchoNext
tests or MIMIC. Infer a lead only from exactly one explicit `lead NAME` in the wave question, or from a
question saying `selected lead(s)` with exactly one single-lead grounding answer in that step. Otherwise
exclude and count; never attach V5 to a question explicitly asking about V1.

Group by NTT-02 class key. Require six independent patients, three outside inner fold zero and two in
fold zero. Choose the first lexical eligible class per diagnosis, then the first four diagnoses. Sort
patients by SHA256(`ntt03:` + patient ID): three nonzero-fold reference patients and two zero-fold targets
per class, one deterministic question per patient. Keep source QA, criterion, lead-link provenance and
patient roles local. No choice depends on a visual score. These small counts support engineering
description only, without a population-level clinical significance claim.

## Frozen visual methods

Read physical 100 Hz mV traces using source lead names. Subtract each lead's full-record median and render
the selected lead as a 518x518 black, two-pixel trace on white, with fixed +/-3 mV scale. The full view spans
ten seconds. The crop view spans one second centered on each supplied candidate (support: selected region;
target: each candidate separately), shifted inside record boundaries. Target answers never select crops.
Retain an exact inverse x-to-time transform. Store clipping fraction rather than hiding it.

Use DINO normalized patch tokens with ImageNet input normalization. Downsample a mask of the drawn trace
inside a candidate interval by area interpolation to the 37x37 token grid; pool with soft occupancy weights,
then unit-normalize. This explicitly preserves thin ink rather than thresholding it away. Average reference
instances into the class prototype and rank target candidates by cosine. Compare full view, crop view,
wrong-window support (first unselected option), and duration matching. Report NTT-02's QRS-phase comparator
on the identical targets when its predictions are available. Known finding/lead/candidate inputs make this
a conditional engineering task; it does not discover diagnoses or select leads autonomously.

Separately, run SAM2 automatic masks on the full ten-second view of the first target per selected class:
32x32 prompt grid, batch 8, predicted-IoU threshold 0.4, stability threshold 0.95, box NMS 0.5, no crops or
small-region cleanup. The predictor is CPU float32; optional CUDA hole/sprinkle cleanup is disabled. Keep
at most 100 masks ordered by predicted quality. Report mask count, best temporal bounding-box IoU against
the nominated silver interval (proposal oracle), and the IoU of the mask with highest reference cosine.
Oracle matching uses the answer only for evaluation. Another equally diagnostic beat is not a verified
negative. No full-page pathology claim follows from this selected-interval endpoint.

## Resources, decision and reporting

At most two numerical threads, no training, no GPU, no environment/lockfile mutation, no pushes or PR
changes. One real-ECG run, at most 15 minutes CPU wall time, with source/checkpoint/input hashes and stage
timings. Stop between images if the budget is exhausted and report every unprocessed image. Preserve
failed receipts. No post-score tuning of rendering, masks, supports or model size.

Report selected-option agreement, interval IoU, per-target predictions, support and rendering sensitivity,
and proposal-oracle versus semantic-selection failures. This pilot selects no clinical winner. Advance
only to a new prospectively powered study if visual matching beats both duration and phase in point
agreement and at least half of SAM targets have oracle IoU >=0.5. Otherwise record the engineering failure
or inconclusive result. Even a pass would use generated, selected silver answers rather than independent
specialist localization truth. Save fixed first-target overlays and reference comparisons locally.

Write a results report from outputs and rank follow-ups. Source commits and all future work use the
personal repository identity. The original large/large model remains a separate unexecuted recipe.
