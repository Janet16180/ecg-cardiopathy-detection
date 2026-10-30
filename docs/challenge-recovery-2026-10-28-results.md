# Original Challenge split recovered

28 October 2026. The original Challenge split `rows.csv` was recovered byte for byte from the public raw
headers and waveforms. Its SHA-256 is
`5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8`, exactly the value recorded in
[the original split report](challenge-splits-v1.md). This was mechanical reconstruction under the user's
explicit authorization; no model was fitted, no prediction or performance score was computed, and no test
evaluation was performed.

## Executed outputs

All outputs remain local under `outputs/cohort25_recovery_20261028/challenge/`:

- `full/rows.csv`: the exact original 65,824-record split.
- `full/metadata.json`: the new reconstruction receipt, SHA-256
  `256ff40fa0f7513bb5f0574e22bbc9d799813fca2b0d2af7f282d88e30a8c94f`.
- `full/*_verified.parquet` and `full/*_receipt.json`: per-source raw-checksum-verified headers and hashes,
  with checkpoint digests. The local tables retain age, sex, original diagnosis codes and release-relative
  paths for subsequent authorized reconstruction; none of those rows enter Git.
- `validation.json`: independently checked historical counts, duplicate families, split separation,
  checkpoint hashes, original receipt hashes and producer hashes.
- `orphan_header_verification.json`: checksum verification of the one present Ningbo header whose waveform
  is absent. Together with the paired-record sweep, all 131,649 present raw header and MAT files match the
  local official checksum manifests. The only absent files are the three historical upstream Ningbo
  problems specified by the protocol.
- `profile/metadata.json`, `full_run.log` and `resumed_run.log`: resource profile and execution logs.

Both official SNOMED mapping CSVs passed the frozen exact SHA-256 checks. The local release checksum
manifests have SHA-256 `f29c2609343c38734f3a3d62be4a9052ca56a9e560e7ad2efc2118563fca2473` (2021)
and `90cfba453b46deac18340eb359332aabbc7300d03f4ea421c225d16787121864` (2020).

## Integrity checks

| Source | Train | Calibration | Test | All records |
| --- | ---: | ---: | ---: | ---: |
| Ningbo | 20,941 | 6,979 | 6,983 | 34,903 |
| Chapman/Shaoxing | 6,147 | 2,050 | 2,050 | 10,247 |
| Georgia | 6,207 | 2,066 | 2,071 | 10,344 |
| CPSC 2018 | 4,121 | 1,375 | 1,381 | 6,877 |
| CPSC-Extra | 2,074 | 692 | 687 | 3,453 |

All 45 historical count cells checked in `validation.json` agree: total records and evaluable primary
positive/negative records for each source and split. There are 64,984 duplicate groups; 816 groups contain
1,656 records. The 406 cross-source groups divide exactly into 71 Chapman–Ningbo groups and 335
CPSC–CPSC-Extra groups. No duplicate group crosses splits.

The real CPU profile processed 20 records from each source in 28.08 seconds. All raw sweeps used four
workers and numerical-library threads fixed at one. The largest recorded full-run worker RSS was
254,556 KiB. The completed Ningbo stage took 1,004.09 seconds; the resumed run took 563.37 seconds,
including the remaining sources and checkpoint loading. These stage timings exclude the discarded partial
Chapman work before the controlled interruption.

## Provenance and limits

Protocol and initial producer were committed as `44a7dd7` before raw processing. After Ningbo completed,
the original process was interrupted and exited with status 130. A behavior-preserving lint cleanup and
explicit checkpoint producer binding were committed as `bd30803`; the resumed process exited with status
0. The original Ningbo checkpoint and its receipt were retained byte for byte. Its producer snapshot is
`producer_44a7dd7.py`, SHA-256
`42127f2d4d50658366e70abdd09d56c8c163e823e8c62ecccc9963ef708ab58d`.
The remaining sources and final assembly use producer SHA-256
`9e213da5916b0ecf9b42dd3e6f3ddfb159569f1e0eb92b522cf6d9291768f70b`.
The aggregate metadata binds each source to its producer and unchanged receipt hash.

Validation passed 21 targeted tests with one skipped, including rejection of modified checkpoints,
incorrect record counts and missing legacy producer provenance. Targeted Ruff checks passed.

The recovered split rows are exact. The reconstruction metadata is new; it does not reproduce the original
metadata JSON or its lost cache-input hashes. This stage also does not recreate the full historical Ningbo
clean manifest, near-duplicate audit, or unrelated external-overlap receipts. Subsequent training quality
checks and cohort selection are separate stages gated on the recovered split hash.
