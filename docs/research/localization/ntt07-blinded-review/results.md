# NTT-07 results: local review packet prepared, specialist review pending

The repaired technical stage completed in 213.50 seconds and passed its readiness gate: thirty
anonymous twelve-lead cases, thirty genuinely pending review forms, and zero clinical annotations.
Each case has a six-page calibrated PDF and six PNG viewing copies, for 180 PDF pages and 180 PNGs.
No finding accuracy, clinical evidence agreement, expert localization or screening performance was
evaluated. The review rubric still awaits clinician approval, and the packet was not delivered.

The original display stage failed its prespecified gate. After six completed cases, the seventh
signal exceeded its fixed +/-4 mV display range. The runner stopped after 27.90 seconds, preserving
six PDFs, six PNGs, six pending forms and seven canonical signals. No patient was replaced. An
explicitly authorized, prospectively committed technical repair widened the physical display while
retaining exactly the same thirty selected identities and display order. The first stage remains
failed; successful repair does not rewrite that outcome.

| Executed stage | Complete cases | Pending forms | Parsed waveform records | Clinical annotations | Technical gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Original fixed +/-4 mV layout | 6 | 6 | 7 | 0 | Failed |
| Repaired fixed +/-15 mV layout | 30 | 30 | 30 | 0 | Passed |

There were 37 waveform-parser calls across the attempts and thirty unique selected patients. The
six initial blank forms remain with the failed attempt; the active repaired packet has exactly thirty
blank forms. Pending was never encoded as a clinical finding of uncertain or absent.

## Selection, role and predecessor integrity

The frozen CLUSTER-03 receipts and their source/input/artifact bindings were checked before consuming
individual coordinates or waveforms. The existing role audit verified PR67 training ownership,
whole-patient exclusion of source folds 9/10, one record per patient and exclusion of all 27 NTT support
patients. The exact 1,993 retained identities, 1,582 fitting patients, 411 validation anchors and all
nine patient-sample hashes reproduced. All 22,156 cached original-time QRS regions passed the frozen
identity, offset, finiteness and half-open ten-second coordinate checks. This replay concerns technical
identity and coordinates; it does not repeat encoder inference or create another model result.

The thirty cases were selected only by the committed patient-identity hash among the 411 validation
anchors, then displayed using a separate fixed random permutation. Source findings, model predictions,
cluster labels, age and sex did not enter selection. All seven signals saved before the first failure
matched the repaired stage byte for byte. Every file of the failed attempt remained unchanged.
These are previously explored development anchors, not an independent hospital or final clinical test.

The repaired stage verified the bytes of all 3,986 cache-bound source waveform files. Hashing those
files is distinct from parsing and rendering the thirty selected waveform records. No additional raw
record entered the packet. No encoder forward, training run, GPU operation, download or source-label
interpretation was performed.

## Physical rendering, privacy and forms

Every case retains canonical twelve-lead, 500 Hz, 5,000-sample physical-mV signals. The repaired layout
uses six 570 mm by 380 mm PDF pages, with two full ten-second leads per page. Each panel is exactly
250 mm by 300 mm: 25 mm/s, 10 mm/mV, and a fixed +/-15 mV range. Fine and major grid squares are
1 mm and 5 mm. A separate 1 mV /200 ms calibration pulse appears on every page. No signal was clipped,
filtered, normalized, marked with detected beats or overlaid with model output.

Local PDF text and metadata audits checked all twelve lead names, six-page inventory, every page's
physical geometry and the restricted anonymous text template. Every PNG passed its dimensions,
resolution and metadata checks. PNGs are viewing copies; apparent screen scale depends on software.
The private key retains exact page/pixel transforms, canonical signal byte hashes, original source
file bindings and artifact hashes. Pixel coordinates use the actual integer raster height and an
upper-left origin before cursor rounding; synthetic tests compare that convention with the renderer's
actual artist transforms.

Reviewer filenames, forms and image/PDF metadata contain anonymous case references without source
identifiers, raw filenames, source codes, model scores or cluster assignments. The same four draft
prompts appear on every pending form. The sinus-rhythm prompt is distinct from source NORM and from
clinical normality. All clinical states, evidence entries, reviewer references and review timestamps
remain null. The separate pending-form schema prevents these blanks from satisfying the evidence
annotation contract or being presented as independent expert provenance.

## Semantic validator and verification

The new [validator guide](annotation-validation.md) documents exact external signal binding,
original-coordinate bounds, frozen beat-list requirements, unique set/region IDs, same-set measurement
references, explicit units and methods, and preservation of joint regions and alternative sets.
The unchanged draft evidence schema is checked with an explicit date-time checker; the installed
JSON Schema runtime lacked an optional default date-time backend, which the synthetic tests detected.
Malformed calendar dates and offset components are rejected without a dependency change.

General source dependence flags remain distinct from clinical expertise. Independent physical rules
and synthetic examples can retain their actual dependence flags while remaining nonexpert provenance.
A reviewed flag alone cannot admit them under the strict expert policy. Expert claims require a
separate reviewer/signal/policy binding. Context-only sets warn under general validation and fail the
declared strict admission rule; this is not a medical judgment of evidence sufficiency. Structural
and semantic validity never certify clinical truth or authenticate a clinician. The coordinator's
separate forward strict import adapter requires finite standard JSON values and excludes adjudication
kinds or adjudicated status until a lineage-aware policy exists, while reusing the unchanged pinned
validator. Actual expert review and any adjudication lineage must be verified externally before
future clinical evaluation.

The focused synthetic suite passed 99 tests in 10.80 seconds and Ruff passed for every new source and
test file. Tests cover valid and invalid signal bindings, half-open coordinate boundaries, bound beat
ordinals, finite JSON numbers, alternative/joint sets, cross-set references, provenance contradictions,
reviewer attestations, strict admission, actual PDF/PNG metadata, physical transforms and blank forms.
No completed human annotation was imported or evaluated.

An [independent artifact audit](../../../../reports/localization/ntt07-blinded-review/verification.json)
passed all sixteen source, 4,091 input and 272 artifact byte bindings, frozen Git source bytes,
selection/roles, all thirty canonical signal bindings, all 180 PDF pages and PNGs, thirty pending
forms, and separation of the private key. It independently reproduced all nine patient splits and
22,156 coordinate checks and verified that all seven original signals and failed-stage files were
unchanged. No waveform or image was forwarded to an external API for either rendering or audit.

## Receipts and local handoff

The initial [protocol](protocol.md) was committed as `2b517af` and the first implementation as
`dc627d9` before its patient selection/rendering. The separate [repair amendment](rendering-repair.md)
was committed as `5b7cdb5` and its source/tests as `35b28a3` before further waveform access.
The repaired receipt binds sixteen source files and 272 private output files.

The active private reviewer packet is `outputs/ntt07_blinded_review/layout_v2/reviewer/`.
Its private identity/signal key is separately stored in
`outputs/ntt07_blinded_review/layout_v2/private_key/`. The failed attempt stays in the parent output
directory. Only the [safe aggregate receipt](../../../../reports/localization/ntt07-blinded-review/aggregate.json),
code, schemas, synthetic tests and documentation enter Git; no patient or signal hash is published.

The completed local result is `outputs/ntt07_blinded_review/layout_v2/result.json`, SHA256
`e6b98f43a915363cec1591a0f1565687aa9800fd6d10d046b5dbeea506550c01`.
The preserved failure receipt is `outputs/ntt07_blinded_review/failure.json`, SHA256
`c2fdfcbc808d081cd54035bfa7d9c5af89a0f6aaa8c5e406df8e5cff7056a803`.
No push, merge, upload, clinician contact or delivery occurred. The coordinating agent will update
both queue documents and rank the remaining specialist-rubric, review and later evaluation work.
