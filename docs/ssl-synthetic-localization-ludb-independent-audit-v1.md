# Independent public LUDB timing qualification audit

Prospective 20 October 2026 independent audit of the separately frozen LUDB qualification.
Commit this protocol, the checker, artificial tests and identity admission before the one
real audit. No waveform or annotation asset has been read by this auditor at protocol writing.
The main protocol remains authoritative; this audit never changes a scientific rule.

Read only the identity-admitted LUDB 1.0.1 release assets, main config/protocol/source snapshots,
main per-record rows, summary, figure and cost receipts, and predecessor temporal-v2 summary/
audit/ledger evidence. No diagnosis metadata, CSV, header comments, other clinical datasets,
neural model, generator, private report or closed test is admitted. Hash complete admitted
headers for identity, but decode only the first thirteen signal metadata lines. Raw assets and
all new audit outputs remain local beneath baseline/outputs/. No external API is used for data.

Independently decode MIT annotation binary words using the public WFDB specification, with
bracketed onset/peak/offset triples and p/N/t wave types; independently decode interleaved
format-16 samples and header gain/baseline/lead mapping. Do not import main endpoint/parser
functions. Verify all 200 numeric records and their twelve per-lead annotators i/ii/iii/avr/
avl/avf/v1--v6; record IDs divisible by five are the forty parser records and the other 160
are measurement records. No record/lead exclusions, silent repairs or data-driven tuning.

Reconstruct every P/QRS/T triple, inclusive QRS mask, per-lead coverage from first complete
wave onset through last complete wave offset inclusive, target prevalence and signal metadata.
This envelope is an operational annotation envelope, not proof every intervening sample was
individually judged negative. Per-lead boundaries remain independent; derived limb leads are
not independent observations. Check exact finite 12-by-5000 geometry at 500 Hz.

Reconstruct the frozen float64 activity score: prepend-first difference, square, twenty-sample
uniform convolution in numpy same mode over the full record, divide by maximum if positive,
then crop to the annotation envelope. Compute AP independently by descending equal-score
tie groups. Shift the cropped score circularly by floor(length*f) at fixed fractions .17,.31,
.43,.59,.71,.83 and average the six APs. Compare every observed/control AP, mean and gain,
wave count, offset and role. Use absolute 1e-12 numeric tolerance, exact identity/count/order
checks, finite values and an eight-float64-epsilon AP admission envelope without clipping.

Average twelve leads equally within record and 160 measurement records equally. Reconstruct
the paired 2,000 whole-record bootstrap at seed 2026106101 using shared patient_groups and
patient_resample; linear 2.5/97.5 percentiles, no skipped draws. The main decision requires
mean AP gain at least .05 and lower bound strictly greater than zero, with every technical
gate satisfied. Report all leads/roles and failed gates. A technical audit PASS means agreement,
including agreement with a negative scientific decision; it is not disease localization.

Verify source and asset hashes before and after use, fixed figure geometry, complete main
output bindings, exact predecessor closure and additive accounting. The inherited terminal
ledger SHA is a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf,
with cumulative CPU 429.49696094999996 and GPU-lock 447.98139213304967 seconds before
this LUDB study. Preserve all failed attempts exactly once and historical unknown costs.
Raw download/setup and development costs must be separately disclosed if not in numerical
accounting; they cannot silently vanish or be presented as scientific operation time.

One CPU numerical thread, hidden CUDA, no GPU or network during audit. Independent caps:
900 CPU seconds, 1200 wall seconds, 256 MiB new audit artifacts; source/read-only raw assets
and existing main outputs are excluded from new-artifact storage. Report peak RSS separately.
Reserve five CPU/wall seconds and 2 MiB for terminal success/failure receipts and ledger.
Stop with an immutable failure receipt on unexpected exception or cap breach; no unrecorded
retry or score-dependent checker amendment. Root must explicitly admit main summary and
checker/core hashes before execution. Audit report is written from the actual saved results.
