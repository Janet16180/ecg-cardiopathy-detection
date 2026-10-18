# SSL and synthetic localization: patient partition and runtime setup

18 October 2026. Data preparation completed; no representation training or localization scoring ran.
The local manifest contains 12,078 training patients, 1,476 validation patients and 1,469 engineering-test
patients, with zero patient overlap. Source folds 9–10 remain excluded. The official checksum table
declares no exact signal-file duplicates among all 21,799 PTB-XL records. The pinned Python environment
has been restored and its focused CPU checks pass. Waveform quality and generator qualification remain
unassessed, so training is not admitted.

## Executed partition

The [partition policy](ssl-synthetic-localization-partition.md) and fixed allocation configuration were
committed as `ba663bc` before the real manifest. The executed implementation is `837a2b2`.
Allocation uses a fixed SHA-256 bucket of canonical patient ID, namespace and seed; it never uses
clinical labels or recording counts. Repeat records and future edited versions inherit the background
patient role. No seed was retried.

| Role | Patients | ECG records | Permitted use |
| --- | ---: | ---: | --- |
| Train | 12,078 | 13,938 | SSL and supervised/generator optimization after remaining gates |
| Validation | 1,476 | 1,719 | Settings/threshold selection after input qualification |
| Engineering test | 1,469 | 1,761 | Future frozen engineering evaluation; no training or selection |
| Excluded folds 9–10 | 3,846 | 4,381 | Excluded from this stream |
| Total | 18,869 | 21,799 | Metadata accounting only |

The receipt records zero patients in multiple roles, zero excluded-fold records in eligible roles,
and zero eligible-fold records requiring mixed-patient quarantine. All 17,418 eligible header/data
pairs exist. Existence does not establish waveform validity. Only identity, source-fold and filename
columns were interpreted; no diagnosis, report or demographic field was interpreted. No waveform or
header payload was opened.

The record-level manifest stays local at
`data/processed/ssl_synthetic_localization_v1/partition.json`. Its SHA-256 is
`8c47123df83da631765e120cfec8e944a8a2b18a34b14bd0a39ab07061e48e2f`.
The aggregate receipt is `outputs/ssl_synthetic_localization/partition_summary.json`, SHA-256
`a64474f5f53f7c095eefda5c50f49336727639353485e010efb6215d48c1498b`.
A fresh in-memory replay from the original metadata exactly matched every rebuilt field, and the
four recorded policy/source hashes matched their current bytes. Reversed artificial rows preserve
assignments; original metadata byte hashes appropriately change with row order.

## Declared exact-duplicate audit

The [checksum-audit policy](ssl-synthetic-localization-checksum-audit.md) was committed as `8499c7b`
before execution from `a484713`. The audit verified the partition's original hash, replayed assignments,
checked original source hashes and matched the original metadata hash to PTB-XL's checksum table.
All 21,799 records have official high-resolution signal-file digests, and all 21,799 digests are unique.
Duplicate groups, cross-patient groups and cross-role groups are all zero. The duplicate gate passed.

The local aggregate receipt is `outputs/ssl_synthetic_localization/partition_checksum_audit.json`,
SHA-256 `8aecf4d84481b2e7de1894ea71fad133c55d23f8f38d6a318bf4971d57db5070`.
The official checksum table's SHA-256 is
`b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d`;
the metadata SHA-256 is `7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b`.

This is an audit of declared encoded-file equality. It does not freshly verify waveform bytes, detect
near duplicates, identify copies with different encodings, validate biological identity or establish
clinical truth. The earlier source-setup checksum receipt remains separate evidence. Engineering-test
waveforms and reserved external tests remain unread by this work.

## Restored runtime and verification

After a fresh locked dry run, 43 GiB of available disk and no competing install were observed.
`uv sync --locked --inexact` completed in the shared main-checkout `.venv`; observed free space afterward
was 37 GiB. Neither dependency file changed. Python 3.11.13 imported NumPy 2.4.6, SciPy 1.17.1,
pandas 3.0.6, WFDB 4.3.1 and PyTorch 2.6.0+cu124. Small CPU array/tensor arithmetic passed with CUDA
hidden. This establishes a working CPU runtime, not CUDA compatibility or GPU resource admission.

`outputs/ssl_synthetic_localization/runtime_restore.json` records the command, installed versions,
dependency hashes and installation-log hash. Its SHA-256 is
`7a9be3a9fc8a92490a0065382ec112e8c0fcc8d1d40cf0983ed3def0d04dda61`.
The restore log stays local at `outputs/ssl_synthetic_localization/runtime_restore.log`.
All 40 focused tests passed in the restored environment, covering readiness, stable repeat grouping,
mixed-fold exclusion, missing paths, bad metadata, checksum copies, source replay/tampering and refusal
to overwrite. Repository-wide Ruff lint passed. Full-suite CI is recorded by the draft PR's checks.
Per-stage preparation/build time was not separately metered; backlog time estimates remain estimates.

The latest path-only receipt is `outputs/ssl_synthetic_localization/readiness_after_setup.json`,
SHA-256 `8a2012b26369d5f9e7c1b7c113db2b749bfb81df21494d36bee5bfdbccf585f7`.
It now lists three absent required paths: qualified generator, executable pilot protocol and GPU profile.
The [initial readiness report](ssl-synthetic-localization-readiness.md) remains a historical snapshot.
The path-only checker deliberately does not certify any scientific gate, including this partition.

## Next work and limits

1. Freeze a permitted-input audit policy, then verify/decode training and validation waveforms, units,
   lead order and signal quality. Use checksum metadata for cross-role identity comparisons; preserve
   the engineering-test waveform restriction. Hash and retain all failures. Do not silently change this
   manifest or call uncertain real backgrounds clinically healthy.
2. Implement and qualify paired controlled ST edits with tracked postprocessing support, shams,
   nuisance controls and an independently held-out rendering mechanism. Recover/reproduce any reused
   predecessor before a numerical comparison; original artifacts are still absent.
3. Freeze the matched random/SSL/frozen-encoder pilot, its input gates and real GPU profile before
   training. The historical V100 forecasts do not apply to the local GTX 1060 3 GB.

This completes metadata partitioning and the declared duplicate check, not all of Stage 0. ECGAD's
technical annotation audit still lacks its local arrays/match table. The new engineering-test patients
come from previously used PTB-XL and cannot establish independent clinical confirmation. No new disease
label, clinical relabeling or final-test authorization is introduced.
