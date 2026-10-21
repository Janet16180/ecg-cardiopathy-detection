# Selected real training cache policy

18 October 2026. The pilot subset was committed in `530cbb9` before real waveform auditing.
This document freezes its cache implementation before any cache preparation. No model trains here.

Bind the immutable partition and the completed permitted-input audit, whose aggregate summary SHA-256
is `160bc03cd0d476f1679508635f99c2c2affe31bb0e565f1592adbd9dd504e2fa`.
Require the audit input-integrity gate, validate its complete input/source/output hash bindings, exact
original-row retention, and unchanged base partition. Read the pilot policy as its clarified `4f6f268` bytes and configuration as its unchanged `530cbb9`
bytes, included in cache source hashes. The subset rule remains the original `530cbb9` rule; the later
clarification changes no real-data selection. Never interpret clinical metadata.

From the immutable partition's TRAIN patients, sort canonical patient IDs by the lexicographic SHA-256
hex digest of `ssl_pilot_v1:real:<patient ID>`; break ties by numeric patient ID. Select the first 2,048
patients before examining quality eligibility; choose the lowest numeric ECG ID per selected patient.
Preserve this ordered original selection and its canonical JSON identity hash. Join quality only after
selection. Skip a selected record failing the frozen training-quality policy without replacement,
retaining its identity and reasons locally. Do not substitute another ECG from the same patient.
A missing/inconsistent audit row is a hard failure. Fewer than 1,900 usable selected patients rejects
cache admission. A hard integrity/decode failure rejects admission rather than becoming a quality skip.

Only selected eligible TRAIN payloads may be opened; validation, engineering-test, excluded and
unselected records must never reach payload access. Recheck the role before constructing a path.
Freshly verify each selected local `.hea` and `.dat` against both official checksum metadata and its
recorded audit hashes. Apply the committed explicit-unit/header checks before canonical `read_record`.
Stream one physical mV record at a time through shared `resample_full(signal, 1, 2)` on the complete
record: float32, 12 leads, 2,500 samples at 250 Hz. Preserve mV; do not divide by the fixed model-entry
2 mV scale here, normalize, clip, fit a transform or change any original partition/eligibility row.

Use one numerical CPU thread and hidden CUDA, with no GPU and no downloads/uploads. Exclusively create
`outputs/ssl_synthetic_localization/real_cache_v1`, refusing any existing directory or symlink.
`signals.npy` is a NumPy-format float32 memmap, ordered by original subset selection after skips.
`selection.json` preserves every selected original row with selection digest, audit status/reasons,
fresh file hashes, preparation failure details and `cache_index` (null for skipped/failed rows).
The `selection_sha256` identity is `sha256_json` of the ordered original selected rows, before audit
information; the separate file hash covers eligibility, indices and observations. Record successful
partial cache rows if a known integrity/decode failure occurs; the aggregate gate is false and the
cache must not train. Unanticipated software errors raise, leaving an incomplete directory without
an admitting summary. No directory may be overwritten or silently resumed.

`summary.json` records complete selected/usable/skipped/failure counts, original subset identity,
shape/rate/units/lead order, all source/config/policy/dependency and audit input/output bindings,
cache/selection hashes and bytes, UTC times and measured wall/process CPU seconds. Store paths with
`paths.to_stored`. Record total local stream-output bytes before and after creation, including this
cache; reject admission above the pilot's 4 GiB ceiling. Retain no record identifiers in console or
versioned aggregate results. `cache_gate_passed` concerns these physical inputs and subset sufficiency;
`training_admitted` remains false. Generator qualification, executable pilot and real GPU profile
remain separate requirements.

Commit code and focused tests before preparation. Tests use artificial inputs for selection before
eligibility, patient/record ordering, no replacement, protected role rejection before file access,
fresh hash validation, units, whole-record native shape/mV preservation and overwrite refusal.
Write the executed results report from live receipts after verifying its output/source bindings.
