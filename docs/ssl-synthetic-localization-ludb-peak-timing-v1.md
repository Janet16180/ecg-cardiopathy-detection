# LUDB released-N landmark timing qualification v1

Prospective protocol, 2026-10-20. Freeze this document, its JSON configuration,
new main/audit source, tests and complete executable admission before any of
LUDB's 160 measurement records are read or scored. This is a new estimand;
it does not repair the failed dense-QRS-span experiment.

The saved record-95 diagnosis found 571 events: 72 complete QRS triples,
35 N/offset pairs lacking observed onsets, and 95 complete T triples.
Both decoders agreed in all 12 streams. These observations justify retaining
released N timestamps without guessing boundaries. They do not independently
validate the physiological location or provenance of each N timestamp.
Preserve the strict v1 source-guard and v2 parser failures, all original receipts
and costs, and the possible prior WFDB header-comment parsing deviation.

Use all 200 public LUDB 1.0.1 records, 500 Hz, 5000 samples, 12 canonical leads,
with the frozen extension-to-lead mapping. IDs divisible by five are the 40
parser records; the other 160 are measurement records. Process all parser
records first; no parser metrics or controls. No exclusions, substitutions,
training, downloads, GPU work, diagnoses, age analyses or closed-test access.
Read-only-use aliases point to the verified local v4 acquisition. Never change
its files or stage aliases. Read only the 13 structural header lines and pass
WFDB annotation reads through scratch annotation links beside a sanitized
structural header; never pass the original record path to rdann. Do not output
or interpret header comments or auxiliary text; retain auxiliary lengths.

Retain every decoded timed annotation event in original source order, including
all numeric codes and duplicate multiplicities. Verify WFDB/raw-decoder numeric
event agreement. Samples must be integers in 0..4999 and nondecreasing.
Unknown symbol lookup alone is not corruption: retain its numeric code/status.
Only explicit numeric code 1 (N) defines landmarks; require at least one per
lead. Preserve duplicates and report counts. Do not sort, drop, deduplicate,
repair or impute annotations, and do not require exact wave triples or only
five symbols. Boundary-completeness summaries are descriptive, with missing
onsets left missing. Annotation channel attributes are metadata, not physical
lead identity. Every record and all 2400 lead streams must remain represented.

Reproduce the exact frozen C2 predecessor six contrasts and intervals using
its original 2000-draw seed 2026105601 before new measurement. Reconstruct and
compare all 12 saved record-95 event tables and their fixed diagnostic counts
before measurement. Record-95 annotations are parser material; its waveform is
never used to select this endpoint or tune the proxy.

The unchanged activity proxy is float64 diff(x, prepend=x[:1]), squared,
convolved over the full 5000 samples with twenty weights of 1/20 using NumPy
same mode and zero extension; divide by the positive full-record maximum,
otherwise return zeros. Reject nonfinite inputs or outputs. Annotations never
affect this calculation. For each lead, the unlabeled reference span is the
first through last retained timed event inclusive, implemented [first,last+1).
This span does not label any sample healthy, non-QRS or negative.

For every released N event, its rank is
(count of reference scores strictly lower + 0.5 times count equal) / L.
Thus a constant score gives 0.5. Average all N events equally within a lead,
all 12 leads equally within a record, and all 160 records equally. Controls
roll only the cropped score vector by floor(L*f), for fixed fractions
0.17, 0.31, 0.43, 0.59, 0.71, 0.83. Require six distinct offsets in 1..L-1.
Keep the original N positions and score distribution; do not reconvolve.
These fixed phase controls can retain periodic alignment and are not a
universal null. The one primary endpoint is mean original rank minus mean of
the six shifted ranks. It measures concentration at released N landmarks.

Bootstrap whole records with shared patient_groups/patient_resample, 2000
numpy.default_rng draws, seed 2026106101, no skipped draws, and linear 2.5/97.5
percentiles. Scientific PASS requires primary gain >= 0.05 and lower bound > 0.
These are engineering thresholds in percentile-rank units, not clinical
accuracy, AP or AUROC. Overall PASS additionally requires all 200/2400
structural/identity/decoder checks, exact closures, finite metrics, resource
gates and an independently reconstructed audit. No alternative endpoints.

Save all event/coverage metadata locally, 1920 measurement lead rows with all
six controls, 160 record rows, bootstrap draws, identities, seeds, actual costs
and failure receipts. Report per-lead distributions and reference-span extent.
After measurement numerics are complete, create aggregate 12-lead rank/control
figures and exactly eight proxy overlays: parser IDs 5,10,15,20 by leads II,V2.
Overlays show released landmarks without parser ranks or dense-mask claims.

Main caps: 600 CPU seconds, 1200 wall seconds, 2 GiB RSS, 256 MiB new outputs.
Independent audit: 600 CPU seconds, 1200 wall seconds, 2 GiB RSS, 128 MiB outputs.
Use one CPU thread and hidden CUDA. Reserve the disclosed two-second/16-KiB
terminal bookkeeping tail; preserve measured failures and resource violations.
Append latest actual diagnosis stage 25 (ledger SHA
cb86ee511b4740ab7b84a589fc81448556df3d5561907b4bd07e2f4748a020ba;
CPU 435.78864791, GPU-lock 447.98139213304967) once: main is 26, audit 27.
Admissions must verify the full earlier chain without resetting failed costs.

A PASS qualifies the same fixed activity proxy used in synthetic-localizer
hotspot diagnosis at released N timestamps. It does not test learned-model
transfer, onset/offset accuracy, dense QRS masks, physiological plausibility,
pathology localization or SSL benefit. Original AP-gated coherence bridge v1
remains unexecuted; using this new landmark gate for a descriptive real-versus-
generated coherence comparison requires a new prospective bridge amendment
before any such comparison score. Reports and backlog must preserve that scope.
