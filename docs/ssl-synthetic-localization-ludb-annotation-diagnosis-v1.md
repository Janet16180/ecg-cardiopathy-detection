# Bounded annotation-only diagnosis of LUDB parser record 95

20 October 2026. Commit this protocol/config, new source/tests and identity admission before opening
record95 annotation contents. Preserve all strict qualification v1/v2 failures and frozen sources.
This is one small format diagnosis, not a scored qualification, repair or new clinical objective.

The actual v2 stage24 attempt stopped in parser qualification at record95 with the frozen guard
'Annotations must be complete ordered triplets'. Eighteen parser rows (IDs5..90) completed; the
record95 source path reached annotation parsing after its signal/header read. No AP was computed
and no160measurement record was read. Preserve ownCPU3.3457967980000003 /wall26.20539320912212,
all four passing resource gates and cumulativeCPU435.4966687269999/GPU-lock447.98139213304967.
A failed strict grammar alone does not establish a corrupted ECG or faulty public annotation.

## Format question and precise permitted inputs

Which condition of the strict guard fails, in which of record95's12streams, and do WFDB and an
independent MIT-format event decoder agree on the original event sequence? Public source review
is allowed before local events. The [MIT annotation format](https://physionet.org/physiotools/wag/annot-5.htm)
defines timing words, SKIP/NUM/SUB/CHN/AUX fields and EOF; [annotation attributes](https://physionet.org/physiotools/wpg/wpg_30.htm)
and [ordering conventions](https://physionet.org/physiotools/wpg/wpg_48.htm) do not promise our
consecutive wave-triple grammar. The [LUDB release](https://physionet.org/content/ludb/1.0.1/)
describes per-lead wave annotations, not a guarantee of this parser's stricter representation.
Do not infer the local cause before this prospective diagnostic executes.

Input allowlist is only public95.hea and95.i,95.ii,95.iii,95.avr,95.avl,95.avf,95.v1..95.v6 via the
read-only raw alias and actual verified v4 output target named in config. Verify their existing
release hashes; a whole-header byte hash is allowed for identity, but interpret only first13structural
lines. Never open/read/hash95.dat, any other record's raw files or the160measurement records.
Failed v2receipt/ledger, identity admission and source/config/package metadata may be read for
closure/accounting. Prior saved rows may be hashed/counted as bytes/lines only, without interpreting
other records' wave annotations. No waveform, AP, proxy, bootstrap, training, GPU or network work.

## Header-I/O correction and preserved possible deviation

Static review of installed WFDB4.3.1 found that rdann may fall back to rdheader on its record path
when annotation sampling frequency is absent. Thus prior qualification annotation calls may have
internally parsed original header comments despite our caller ignoring them. Preserve this as a
possible annotation-I/O protocol deviation; do not claim those comments were proved unread.
No clinical comment values were used, logged, scored or made into labels, and no clinical analysis
is authorized by this finding. Existing clinical labels and all closed-data restrictions remain.

The diagnostic MUST create a scratch header containing exactly the original first13structural
lines and read-only symlinks to the12admitted annotation streams beside it. Call WFDB rdann only
on this sanitized scratch record path, never the original record path. No DAT symlink is needed or
permitted. The independent byte decoder reads annotation streams directly and needs no header.
Pin WFDB version/installed annotation and record source identities in admission. The same sanitized
annotation-I/O design must be included in any future parser successor before measurement reads.

## Complete events, fixed checks and independent verification

Retain all12streams in canonical order, including successful streams and decoder errors. Preserve
original byte/event order; never sort, drop, merge, infer or repair events. Save complete local event
tables of sample, symbol, numeric label code, subtype, channel and num. Record raw control-word
counts, offsets and auxiliary byte lengths without auxiliary text or diagnostic interpretation.
Independent decoding follows MIT event semantics, including signed PDP-11 SKIP intervals, NUM and
CHN persistence, SUB's current-event scope, AUX padding and EOF. Standard label codes are syntax;
N is a QRS marker here and is not a clinical healthy label. Unknown codes and decoder errors are
evidence, never silently discarded. Report any intentionally excluded format-definition records
explicitly under the documented WFDB convention, rather than changing the comparison silently.

For each stream separately report: event count/empty flag, count modulo3, symbol/code counts,
all negative adjacent-time indices, all positions outside0..4999, duplicate sample/code counts,
and every violation of frozen onset/peak/offset symbols/within-triple timing/cross-wave ordering.
Invoke or exactly reproduce the frozen guard on decoded samples/symbols solely to identify its
failure branch; do not create QRS masks or evaluate signals. Distinguish empty, incomplete count
and decreasing-time conditions that share the current exception message.

Save every offending index and complete event table. Display fixed±3event neighborhoods, deduplicated
in original order, capped at20displayed neighborhoods per stream while retaining the full index
list. For count/empty failures also retain the first6and last6events if present. No favorable lead
or neighborhood selection, and no local waveform plot. An explanatory future parser proposal must
state precisely how all original boundaries/unmatched events/overlaps would be represented; this
diagnostic cannot authorize that change or resume qualification automatically.

Use a separately reviewed independent raw-event implementation in the SAME one charged run, not
another unbudgeted data peek. Its complete comparable event tuples must agree with WFDB for all12
streams for technical PASS, alongside scope/hash/resource checks. If a decoder fails or disagrees,
retain complete errors/indices and an inconclusive/technical FAIL; do not force agreement. Parser
incompatibility flags are the subject of diagnosis, not criteria for declaring this diagnostic a
scientific success. No scientific PASS threshold. Independent source review precedes execution;
post-run source/output/metadata review is setup verification, not a fabricated extra cost stage.

## Limits, accounting and next decision

Single thread, CUDA hidden. One run:60CPU/120wall seconds,512MiB peak RSS and16MiB new outputs,
including source/admission/header snapshots, event/diagnostic tables and receipts. Guard full work,
file hashes and exact retained bytes, with disclosed2-second/16KiB terminal-ledger tail; preserve
failure receipts even if a live timer interrupts final enumeration. Refuse overwrite and preserve
all failed attempts. Start before imports/input verification and include both decoders/summary work.

Immediate prior is actual qualification-v2 stage24 ledger pinned in config/admission, with its
verified cumulative costs unchanged. Diagnostic stage25 appends only its own measured CPU and zero
GPU once. No new narrow-audit stage is required by this single dual-decoder protocol. Unknown old
costs remain unknown; output root is `outputs/ssl_synthetic_localization/ludb_annotation_diagnosis_v1/`.
Keep raw events/data/receipts local; Git gets source/protocol and aggregate findings only.

Write an output-derived diagnosis report separating actual format facts from hypotheses. Preserve
the possible old rdann header-comment deviation. Any annotation parser or coverage interpretation
change requires a separately committed source/protocol amendment and independent review before
any160measurement read or score. No record/lead dropping, threshold tuning, silent rescue of the
strict failed attempts, disease-mask claim or clinician-recruitment prerequisite is authorized.
The coherence bridge remains gated on a future complete valid qualification/audit conjunction.
