# KEY-03: the ECG references behind the condition averages

Prospective descriptive audit, 6 November 2026. User requested thorough analysis of the ECGs used
for averaging. Freeze before new descriptive computations. No classifier, label, threshold or
reference pool is changed. No new evaluation scores, clustering or clinical adjudication.

## Population and provenance

Exactly KEY-01/02's 25,000 admitted training ECGs. Use original cohort metadata, HEADS-01 labels,
pooled vectors and training tokens after receipt hash checks. Verify saved KEY-01/02 private
artifacts against their results. Repeat their saved metric dictionaries as arithmetic integrity only;
no evaluation waveform is opened. All individual waveforms examined must first resolve to an
admitted training row and pass `pr74_screening_features.read_training_signal` raw and canonical
checksum checks. No Challenge/EchoNext closed data, MIMIC or CODE labels, age analysis or new mapping.

## Descriptive questions (no promotion gate)

1. Count source/family, known patients, repeated known-patient ECGs, unknown patient identities,
   canonical/full-signal hash duplicates, label evidence and review flags. Per class and normals,
   show counts, source shares and four-label co-occurrence. Distinguish source provenance from
   individual human-validation fields; missing validation is not proof of software-only annotation.
2. Audit whole, short selected, fixed-one-second, 2.5-, 5- and 7.5-second keys, both positive and
   class-specific normal keys. Every ECG has equal averaging weight after feature normalization.
   Reconstruct each key directly from its saved reference selections and the original cached vectors,
   require unit-key agreement within 1e-12 (summation order tolerance), and preserve exact inputs.
3. Report resultant length (norm of mean unit vector), reference-to-key cosine quantiles,
   source counts among the 50 nearest references, source-removed key rotation, and rotation from
   whole-record to selected-region keys. Compute equal-patient summaries only as geometry diagnostics,
   never rescore or select a replacement model. Unknown-patient records remain individually weighted
   and cannot establish unique-patient counts. Show exclusive-group and co-occurring-group key
   directions when both exist. No "typical disease" or causal interpretation of nearest examples.
4. For positive and normal references separately, show saved region lengths and starts, source
   differences, and distance from the corresponding whole-record direction. The regions are contextual
   token spans, not physiological boundaries or localized disease truth. Reference selection can
   emphasize different shapes or acquisition properties; no mechanism is assumed before analysis.

## Fixed waveform review selection

For each of the four annotation groups and the shared normal pool, choose four reference examples:
nearest to the whole-record key within each of its two largest source families; farthest from the
whole-record key; and largest whole-to-short-selected vector angle (normal uses the MI-specific
short normal selection, explicitly labeled). Resolve ties by cohort order. Keep every resulting
role, deduplicating waveform loading only. Expect 20 panels with possibly repeated ECGs.

Read only these training ECGs through the frozen checksum-validating loader. Plot all 12 leads,
physical mV and time, with nominal selected short, five- and 7.5-second spans indicated. Use
anonymous panel IDs, source family and the existing annotation memberships; keep original IDs
and mappings in ignored outputs. Summarize executed amplitude, finite/flat-lead checks and
available metadata review flags. These are descriptive quality observations, not diagnostic
interpretation. Do not average unaligned raw beats or call an embedding mean an average waveform.

## Outputs and verification

Two CPU threads, cached features first, no GPU. Private reference metrics and waveform panels,
PDF atlas and an executed waveform notebook stay in `outputs/reference_ecg_audit_20261106/`.
Git receives only aggregate reports, source, protocol and an executed aggregate notebook
`notebooks/ReferenceECGAudit.ipynb`; no individual waveform images or per-record coordinates.
New helpers import shared path/normalization/readout utilities. Freeze sources before execution,
refuse output overwrite, retain failures and write source/input hashes and timings.

Audit aggregates against cohort denominators and saved selections, unit-key reconstruction,
co-occurrence symmetry and source-share sums. Report missing raw waveforms or failed integrity
checks transparently without substitution. Finish with a results report from outputs, both queue
updates and ranked follow-ups. The existing KEY-01/02 verdicts and notebooks remain intact.
