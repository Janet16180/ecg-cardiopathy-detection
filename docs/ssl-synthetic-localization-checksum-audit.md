# Checksum audit for the SSL/synthetic patient partition

18 October 2026. Prospective metadata audit, separate from waveform decoding and localization scoring.
The [patient partition](ssl-synthetic-localization-partition.md) is immutable once receipted. This audit
may reject it for training; it must never silently reassign or delete records to make the gate pass.

Read the local partition, its aggregate receipt, and PTB-XL 1.0.3 `SHA256SUMS.txt`. Require the partition
hash to match its receipt, its policy to match the committed v1 allocation, every original source hash
to remain unchanged, and the receipt's metadata digest to match the official checksum entry. Recompute
patient assignments from the manifest's identity fields and require an exact match. Require a unique,
valid SHA-256 entry for every referenced high-resolution `.dat` file, including excluded records.
No waveform, waveform header, diagnosis value or closed external cohort is opened.

Group all records by the official `.dat` digest. Report aggregate numbers of duplicate groups/records,
cross-patient groups, cross-role groups and affected role pairs. Repeated files within one patient are
allowed only within its single role. Any cross-patient duplicate group rejects the duplicate gate,
even if both patients currently have the same role: the future background bootstrap must not pretend
such copies are independent. Fail on missing entries, malformed checksums, inconsistent assignments,
source/metadata mismatch or a changed manifest. Retain every finding.

This checks exact encoded signal-file duplication as declared by the official checksum table. It does
not detect near duplicates or identical physical signals encoded differently, and it does not freshly
verify local waveform bytes against that table. The earlier source-setup checksum verification remains
separate evidence; a training loader still needs fresh integrity/quality checks on its permitted inputs.
Passing this audit records `training_admitted: false` until those checks, generator qualification and
resource admission exist. It does not authorize reading engineering-test waveforms or reserved tests.

Write a new aggregate receipt only to
`outputs/ssl_synthetic_localization/partition_checksum_audit.json`, binding partition, metadata,
checksum table, policy and implementation hashes. Refuse overwrite. Tests use artificial checksums to
exercise cross-role/patient copies, within-patient repeats, missing entries and tampering, and assert
that no waveform payload can be opened. A rejected duplicate gate returns exit status 2; malformed or
inconsistent inputs raise before writing an audit receipt. Publish only aggregate findings in a results
report. This is a data-preparation check with no phenotype score or GPU work.
