# Permitted waveform input audit policy

18 October 2026. Freeze this policy before opening any real waveform or header in this audit.
This audit qualifies physical inputs for the SSL/synthetic engineering stream. Waveform quality
is not clinical normality. No diagnosis, report, age or sex field is interpreted.

## Scope and binding

The immutable partition remains `data/processed/ssl_synthetic_localization_v1/partition.json`.
Replay it with `ssl_synthetic_checksum_audit.verify_partition`, verify all original partition-source
hashes, and bind the official `SHA256SUMS.txt` to the earlier declared-checksum receipt before opening
payloads. Reproduce the earlier aggregate duplicate audit exactly. Only `train` and `validation`
records may open `.hea` or `.dat`; engineering-test and excluded rows remain present and unread.
Check the role before even constructing payload access. No Challenge, EchoNext, SPH or other source
is an input. No optimization, scoring, GPU, waveform cache, upload or clinical annotation inspection.

Before decoding each permitted record, freshly SHA-256 both local header and signal bytes and require
exact agreement with the official table. Validate header metadata: 500 Hz, 12 unique canonical leads,
5,000 samples, explicit `mV` on every lead, and the expected single same-stem `.dat` signal file only.
Reject metadata that could redirect decoding to an unverified payload. Reorder into canonical `LEADS`
using shared `read_record`; do not filter, normalize, resample or otherwise repair signals.

## Outcomes fixed in advance

Hard integrity/decode failures include absent files/checksums, byte mismatches, invalid path/header,
wrong dimensions/rate/leads/units, inaccessible or undecodable payloads, and nonfinite decoded values.
Retain their typed reasons locally; any such failure rejects the aggregate input integrity gate.
Continue auditing other permitted records so failure accounting stays complete. Unexpected software
errors raise and leave the directory incomplete; an incomplete run never admits inputs.

On successfully decoded signals apply the existing `ecg_experiment.ecg_quality.assess` unchanged.
Its training exclusion reasons are nonfinite, constant lead, constant segment at least one second,
ADC rail (absolute peak at least 32.6 mV), extreme amplitude (above 20 below 32.6 mV), near-flat
record (median lead SD below 0.02 mV), and noise-dominated (median 40–150 Hz power fraction above
0.5). Nonfinite values also fail the hard decode gate. Review flags are peak above 10 mV,
limb-identity residual above 0.05 mV, and median 0–0.7 Hz power fraction above 0.8. Review flags
alone never exclude training records. These are technical recording rules, not disease labels.

The derived eligibility manifest retains every original record and role with audit status, hashes,
quality reasons and flags. Train optimization eligibility requires successful integrity/decode and
no quality exclusions. Validation is never automatically removed for quality reasons; record all its
quality flags and preserve its entire role. An integrity failure blocks admission of the whole input
set rather than making validation easier. Protected roles have status `protected_unread`, null
eligibility, and no payload hashes. The base partition is never changed. Patient grouping survives
any record-level training exclusion and no record moves between roles.

## Execution and receipts

Commit implementation and focused tests before the real run. Use one numerical CPU thread, hidden
CUDA and streamed single-record decoding. The destination is the new directory
`outputs/ssl_synthetic_localization/input_audit_v1`; refuse any existing directory or symlink.
Write record-level `records.jsonl` and `eligibility.json` only locally. Aggregate `summary.json`
contains counts by role/status/reason/flag, train eligibility and retained validation counts, fresh
verified byte totals, source/input/output hashes, dependency versions, UTC timestamps and wall/CPU
seconds. Store paths repository-relative with `paths.to_stored`. No record IDs enter console output,
versioned reports or aggregate receipts. Preserve failure detail locally. A partial directory cannot
be overwritten; a new policy/version is required to rerun.

Tests cover protected-role rejection before file access, byte corruption, unit/lead/header failures,
quality exclusions versus review-only flags, retained validation, and refusal to overwrite outputs.
The decision is input integrity plus recorded train-quality eligibility, explicitly **not full
training admission**. Generator qualification, executable protocol, real GPU profile and other
scientific gates still apply. Report executed totals and limitations from the live receipt, including
all failed or excluded cases; do not relabel this audit as a model or clinical result.
