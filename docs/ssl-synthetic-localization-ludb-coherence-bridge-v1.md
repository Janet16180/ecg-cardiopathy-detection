# Conditional descriptive bridge between generated and public real ECG activity

Prospective protocol,20October2026. Commit this protocol/config, source/tests and identity-only
admission before computing this comparison. Run only after the complete public LUDB annotation/
QRS-proxy qualification passes its technical and scientific conjunction and independent audit,
including actual resource limits. This is a small interpretation bridge back to PR67's released
ECG generation/localization research. It does not replace the subsequent held-out-editor learning
question. No training, GPU use, downloads, new generator outputs or clinical classification.

## Question, estimand and interpretation

How does the exact activity-coherence descriptor already measured on128released-generator ECGs
compare descriptively with the same descriptor on200public LUDB recordings? The single primary
descriptive contrast is real minus generated mean native-minus-shifted activity Jaccard. Its sign,
size or interval has no scientific PASS, equivalence, realism or physiological-quality threshold.
A higher value can reflect tighter synchrony or synchronized artifacts; it is not superior ECG
quality. Distribution overlap would not establish P/T morphology, valid rhythm, disease conditioning,
physical amplitude calibration, privacy, diversity or real disease localization.

The separately qualified QRS-ranking proxy and this descriptor share a derivative-energy ingredient,
but their outcomes differ: QRS AP evaluates alignment to annotations within admitted coverage;
coherence measures cross-lead agreement over the entire5000samples. Passing the former does not
validate the latter as an ECG-quality score. Generated samples have no true public wave or disease
mask. Keep all five claims separate: ECG-shaped output, physiological plausibility, disease control,
known introduced numerical support and real-disease localization.

## Cohorts, allowed reads and execution gate

Use the complete existing `outputs/ssl_synthetic_localization/generated_quality_v2/` bank:128native
float32 arrays in `raw.npy`, seeds2026103000..2026103127, index0..127. Allowed old artifacts are
`raw.npy`, `cases.jsonl`, main `summary.json`, and `independent_audit_v1/summary.json`, plus the frozen
scientific sources/configs/protocols identified in their receipts. No models, noise, variant/support
banks or other generated-learning panels are needed. The released generator checkpoint remains
`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`; no new forward or weights
read is authorized. Training-cohort overlap with real records remains unresolved.

LUDB uses all200numeric record IDs1..200 after successful qualification. Their40parser and
160measurement roles remain documented for that predecessor; here all200are spent public
measurement/development evidence, not a new held-out cohort. No patient split, tuning, model
selection, diagnosis matching, demographic analysis or disease labels. Require source400files
(200`.dat`+200`.hea`) from the verified1.0.1release through its declared read-only-use raw-data alias.
Interpret only each structural header's first13lines using the frozen qualified signal reader;
never parse/store comments or `ludb.csv`. No new numeric annotation-stream reads are needed.
Qualification rows may be read solely for exact predecessor closure. Dataset/generator provenance
and their uncertainty remain those of the prior studies; bootstrap units do not establish privacy
or independence from model training data.

An identity-only admission must bind the actual successful LUDB main/audit receipts and terminal
ledgers, original role/source identities,400used raw-file hashes, logical source alias and actual
local target. Before opening real samples, require independent audit PASS/zero mismatches and
all main/audit resource gates, then reproduce the160-record predecessor primary point, all2000
paired bootstrap gains/interval and scientific verdict exactly from its saved measurement rows.
Preserve its full-QRS qualification interpretation; the bridge does not modify its scientific rule.
If its conjunction fails or any required artifact is missing, do not run this bridge or substitute
another dataset. All signals, masks, per-record outputs and figures remain local under baseline
outputs; only source/protocol and small aggregate reports enter Git.

## Identical geometry, representation and fixed descriptor

Both cohorts use500Hz,5000samples and exactly I,II,V1,V2,V3,V4,V5,V6 in that order. These are the
eight native generator channels, excluding algebraically derived limb leads; they remain correlated
measurements, not independent subjects. Map LUDB by canonical signal name, not channel position.
Use its physical signal values from the source reader and cast the selected8x5000array tofloat32.
Then use the exact frozen `ssl_synthetic_generated_quality.standardize` on both cohorts: median
and linear q95-minus-q05 in float64 per lead, require each range>1e-6, subtract/divide in float64,
cast standardized values tofloat32. Any nonfinite/invalid geometry/range fails the entire attempt;
never discard a lead/record, substitute zeros, change units or tune a range floor after inspection.
This operational floor reproduces the original computation; it is not a clinical amplitude rule.
The transform changes between-lead amplitude ratios and cannot establish generator mV calibration.

Recompute original128standardization parameters, all coherence fields and its primary comparison
exactly before any new real-cohort metric. Original old primary is0.8094248365473182 with interval
[0.8038931804338517,0.8148564292963815],2000whole-seed draws fromdefault_rng(2026103301),
zero skipped. Compare raw recomputed values against the saved outputs at exact tolerance0; rounded
report numbers are insufficient. Do not rerun old interventions, model replays or unrelated quality
metrics to obtain this closure.

For each standardized lead, float64 first difference with the first sample prepended gives initial
zero difference. Square and NumPyconvolve with20weights1/20,mode='same', using its exact even-kernel
alignment/zero extension. Activity is strict greater than that lead's linear90th-percentile energy
across the full5000samples. For each of28unordered lead pairs, compute activity-mask intersection/
union Jaccard, with an empty union contributing0; average28equally. Controls independently roll
the completed8masks within the full5000sample length. Do not crop or resample LUDB using its wave
coverage; matching geometry is essential. Full-record descriptor interpretation explicitly extends
outside the subset in which QRS AP was qualified and cannot inherit annotation accuracy there.

Keep generated shifts exactly `default_rng(2026103200+index).integers(1,5000,size=8)` for index0..127.
For LUDB numeric IDr, fixed indexr-1, use `default_rng(2026106300+(r-1)).integers(1,5000,size=8)`.
Preserve every8shift vector, native/control pair Jaccards, mean native/control/excess values and8
activity occupancies. Empty activity masks remain included under zero-union convention; report
counts. No shift, threshold, lead, record or example is selected after outcomes. Require finite
Jaccards within[-8eps,1+8eps] and excess within[-1-8eps,1+8eps], epsfloat64, preserving raw scalars
without clipping. The independently shifted control is a phase-disruption reference, not a clinical
null distribution or evidence that all pathological structure was removed.

## Descriptive aggregate, uncertainty and fixed figures

Mean each cohort equally by whole record/generator seed. The primary descriptive point is
`mean(LUDB_native - LUDB_shifted) - mean(generator_native - generator_shifted)`.
For2000draws fromdefault_rng(2026106401), first draw200whole LUDB records, then128whole generator
seeds, independently within their respective sorted ID/index arrays, using shared `patient_groups`
and `patient_resample` helpers. Reuse those cohort indices for native/control/excess. Save each draw's
two means and contrast; use linear2.5/97.5percentiles of contrasts, no skipped draws. This is an
independent two-cohort bootstrap, not a paired real-to-generated record comparison. It conditions
on one checkpoint and these source cohorts and omits model-training/site/population uncertainty.
The original128seed interval retains its original seed as an exact-closure result; do not silently
replace it with a new interval or infer clinical significance from the new contrast interval.

Report all328records, cohort means, standard deviations(ddof=1), minima, quartiles, medians, maxima
and empirical distributions for native Jaccard, shifted Jaccard and excess. Also report each cohort's
8lead occupancies, all shift vectors and empty-mask counts; no new hypothesis tests or subgroups.
Fixed figures: paired native-versus-shifted scatter for all records, common0–1axes; cohort ECDFs for
native/shifted/excess, with native/shifted0–1 and excess-1–1axes; fixed-bin distributions(edges every
0.05over0..1 for Jaccards/every0.1over-1..1 for excess); activity-mask rasters for generated indices
0,1,2,3and LUDB IDs1,2,3,4, all8leads and full5000samples. No waveform, diagnosis label, selected
favorable example or outcome-dependent plotting scale. Save figures locally with explicit captions
that synchrony is not clinical quality.

## Technical gates, resources, outputs and independent audit

Technical PASS requires the successful qualified/audited predecessor gate, exact old128closure,
all328records/8leads/5000samples retained, exact identities/source hashes, canonical mapping,
finite/range-valid representations and descriptors, exact shifts/bootstrap/aggregation, zero
omissions, resources and separately reconstructed independent audit. There is no scientific PASS
flag. A technical PASS only means this descriptive comparison was executed reproducibly.

Main limits180CPU/300wall seconds,2GiB peak RSS and128MiB new retained outputs. Independent audit
separate180CPU/300wall,2GiBRSS,64MiB outputs; both hidden CUDA, numerical threads1, zero GPU,
training, generator forwards or network operations. Guard whole stages including I/O, hashes, saved
masks/figures/receipts, with exact byte enumeration and disclosed2second/16KiB final ledger tail.
Use conservative retained-byte accounting rather than repeated growing-tree scans. Refuse overwrite,
preserve failed attempts/partial outputs and append actual own cost once. Immediate accounting base
is the latest completed LUDB qualification independent-audit terminal ledger pinned prospectively
in identity admission, not the old generator ledger or historical429CPU baseline. All transport
failures and intervening audit/main costs remain in that chain; unknown old costs remain unknown.

Destination `outputs/ssl_synthetic_localization/ludb_coherence_bridge_v1/`: full328coherence rows,
400raw-file and old-bank identities, qualified-predecessor and old128closures, all13,120,000boolean
activity-mask cells, shifts/pair Jaccards,2000bootstrap contrasts/cohort means, complete aggregates,
fixed figures, source/admission snapshots, technical gates and measured summary/terminal ledgers.
Do not copy raw waveforms/weights/prediction maps unnecessarily. Independent code must be committed
before execution, reconstruct the descriptor from original generator raw and LUDB signal bytes
with separate arithmetic, reproduce every row/bootstrap/control/figure-identity and additive costs,
and report every mismatch. Before qualification, acquisition receives independent source/receipt/
asset-identity review; its complete numerical acquisition audit may be combined with the full
qualification audit after main. The bridge requires that completed combined audit to pass first,
without inventing an extra acquisition-audit stage.
Coordinator writes the output-derived results; update both queues/backlog and keep PR67draft. Any
new localization fit or diverse-editor transfer still requires its own prospective protocol and
actual-path GPU resource profile including map persistence/whole-stage overhead.
