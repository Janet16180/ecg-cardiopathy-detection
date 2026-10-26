# NTT-07: blinded review packet and semantic evidence validation

## Question and prospective decision

Can a local, model-blinded first-round specialist review packet be prepared without creating clinical
annotations, leaking source labels, or consuming held-out patients? This is technical readiness work,
not a clinical accuracy or localization evaluation. Commit this protocol before selecting or rendering
patients, then commit the source and synthetic tests before execution. The primary readiness gate is
exactly 30 calibrated, byte-bound, role-verified cases and 30 structurally valid pending review forms,
zero fabricated clinical annotations, and passing identity, coordinate, privacy and provenance checks.
Failure preserves a receipt and cannot be repaired by selecting replacement cases after inspection.
A clinician must approve the rubric before actual review or a later evaluation.

## Inputs and integrity

Bind CLUSTER-03 result SHA256
`64949f3472efdc309792d4ea8ca0d6bdd0c017383796bc60246b9536882e67f9` and cache result SHA256
`6758aee6b8d90b12e0772e28a1f877bd275e17779c062c1386cbbd0f93033fb3`.
Verify their source, input and artifact bindings (excluding documented mutable status snapshots),
then use the frozen `native_cache.verified_cohort()` role audit before reading individual feature,
waveform or diagnostic payloads. Verify PR67 training ownership, whole-patient source-fold 9/10
exclusion and exclusion of all 27 NTT support patients. Reproduce the exact 1,993 identities and
1,582 fitting /411 validation-anchor split, frozen CLUSTER-02 identity hash, saved split assignments,
and cache identity/offset/coordinate shape, finiteness and half-open ten-second bounds. Native model
scores are not recomputed because no model is used in this study. No diagnostic labels are parsed.

## Frozen selection and blinding

From the exact 411 CLUSTER-03 validation anchors, sort by hexadecimal SHA256 of UTF-8
`ntt07_selection:20261019:<integer patient_id>` and take the first 30. Break a hash tie by integer
patient ID. This selection depends only on frozen identity, never source finding, model score, cluster,
quality or appearance. Independently permute these 30 using NumPy PCG64 seed 2026101907 and assign
`case-001` through `case-030` in display order. Do not replace cases on quality or visual inspection.
The private key retains identities, file bindings, signal hashes and rendering transforms separately
from the reviewer directory. Reviewer filenames, image/PDF metadata and forms must not contain source
IDs, raw filenames, source codes, cluster labels, model scores or predictions.

Every case receives the same blank finding checklist: sinus rhythm, complete right bundle branch
block, complete left bundle branch block, and other finding (free text). These are review prompts,
not existing patient labels, an exhaustive diagnosis taxonomy or a revised clinical label definition.
All states, annotations, reviewer and timestamps remain null/pending; never encode pending as clinical
uncertainty. A separate pending-form schema prevents blank forms from satisfying the annotation
contract. Round one contains no source labels, detected-beat marks, native/model overlays or metrics.
No clinician is contacted and no packet is uploaded or delivered.

## Signal, rendering and semantic contract

Use only frozen `native_cache_v2.read_waveform` after verifying each selected header/data digest from
the cache receipt. Require canonical 12 leads, physical mV, 500 Hz and 5,000 samples. Retain the
unaltered canonical float64 signal locally, bind its explicitly defined little-endian C-order byte
hash, original file hashes and output hashes. Render all 12 full ten-second traces with a physical
25 mm/s and 10 mm/mV scale, 1 mm/5 mm grid and a calibration pulse; show lead names and signal seconds
without source identifiers. Use a sufficiently large fixed PDF page to avoid overlapping leads or
signal clipping; abort if the fixed amplitude range is exceeded. Store exact data-to-page transforms
and PNG dimensions in the private key, and test coordinate round trips on synthetic data. PDF print
must use actual size; screen PNG scale depends on display settings. No automated delineation is truth.

Implement a new reusable validator over the unchanged draft evidence schema with date-time format
checking. Check all numbers finite, exact externally supplied signal identity/hash/rate/length/leads,
unique set/region IDs, lead membership, same-set measurement references, lead relationships and
explicit measurement units/method. Seconds/samples obey half-open record bounds; ordinal spans require
an external frozen, hash-bound beat list with integer bounds before any mapping. Preserve the
original coordinates, alternative sufficient sets, joint regions, unknown unmarked samples and
record-level absence. Do not silently flatten alternatives or turn absence into negative intervals.
Require present findings to have evidence or a reason. Reviewed/adjudicated records require an actual
reviewer and valid timestamp. Reject contradictory independence/blinding claims; independent expert
status requires an external reviewer/signal/policy binding, not a flag. Under a declared strict
blinded-expert evaluation policy, reject context-only sets and missing expert independence; this
technical policy does not judge clinical sufficiency. Validation never certifies clinical truth.

## Execution and reporting

Use two CPU threads, no backbone inference, GPU, training, data downloads or dependency changes.
Use the existing installed JSON Schema validator. Limit the real packet stage to 900 measured seconds,
checking between cases. All signals, private key, PDF/PNG and forms remain under
`outputs/ntt07_blinded_review/`; only safe aggregate receipts, code and synthetic fixtures enter Git.
Source hashes, protocol/source commits, runtime, selected-identity hash, all artifact hashes, audit
counts, failures and limitations go into the local receipt. A small public aggregate omits identities
and signal hashes. Write the results report from actual executed outputs even if readiness fails.
Human annotations, clinical accuracy, independent evaluation and clinical rubric approval remain
pending. Follow-ups go to the shared backlog through the coordinating agent; no push or merge.
