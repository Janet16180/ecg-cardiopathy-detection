# Experiment 057: prospective annotation-format amendment

Frozen before any waveform score. The metadata-only audit of all90 reference annotation files
found embedded NUL-terminated strings with trailing padding bytes in four records, and12 unfinished
episode keys in nine records. All unmatched starts are terminal episodes; there are no unmatched
ends or extrema after splitting auxiliary strings at their first NUL. Record these source defects.

Preserve unfinished episodes with `end=null` and `censored=true`, without inventing a positive label
until the end of the recording. Four terminal ST keys occur in three records: e0405, e0409 (two
channels) and e0704. For the ST endpoint and positional-ST stress control, restrict both methods'
common evaluation domain to window centers strictly before the earliest unfinished ST or st onset.
Preserve all computed waveform scores and an explicit evaluation mask, so the gold-label restriction
is visible. An unfinished T episode does not erase independently complete ST annotation coverage;
its unknown end remains recorded and supplies no T endpoint. This is an annotation-completeness
restriction for a conditional benchmark, not an inference filter available in deployment.

Completed ST episodes crossing the cutoff retain their available evaluable centers; only episodes
with at least one evaluable center enter the top-5% coverage denominator. Record windows censored
per record, all incomplete keys, and any record with no evaluable window; exclude such a record
explicitly without replacement. The85-record and30-subject quality gates remain unchanged.

The parser yields364 complete ST episodes plus four unfinished ST episodes, while the official page
lists367 ST episodes. Report that discrepancy instead of changing the parser to match the published
aggregate. Candidate features, first30s reference, comparator, targets for complete episodes,
patient grouping, seed, resource budget and promotion thresholds in the original protocol are
unchanged. No new score or label-directed waveform inspection preceded this amendment.
