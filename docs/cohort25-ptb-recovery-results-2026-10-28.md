# PTB-XL training quality recovery

The executed recovery assessed all 17,418 records of the original PTB-XL training manifest. Exactly 17,405
passed the frozen quality policy and 13 were excluded, matching the PTB-XL count documented for
`clean_25k_v3` and `clean_25k_v4`. This restores the PTB quality decisions; it does not establish identity
of the complete 25k cohort or produce model performance results.

The local evidence is
[`quality/receipt.json`](../outputs/cohort25_recovery_20261028/ptb/quality/receipt.json) and
[`quality/record_quality.csv`](../outputs/cohort25_recovery_20261028/ptb/quality/record_quality.csv).
The CSV SHA-256 is `8485ec71be1a2dc06a6cc68cc9134f4a07292aab3e9cad7f610d37118e7e645e`.

## Integrity and access

The training manifest's MD5 is `82d391124fdf28d5ee2a55d29e5dcf2c`, exactly the existing DVC pointer.
All 17,418 selected ECG IDs are unique and their official folds are between 1 and 8. Every selected row's
patient and waveform path agrees with the raw release metadata. The metadata and input hashes are bound
in the receipt and were checked again after processing.

For every selected recording, both the `.hea` and `.dat` files matched the official release SHA-256 before
and after decoding. The runner called the unchanged `waveforms.read_record`, `ecg_quality.assess` and
`public_sources.signal_sha256` functions. The worktree copies of these helpers were also compared byte for
byte with the main checkout before processing. The local per-record CSV retains canonical waveform hashes,
official raw-file hashes and each original quality reason.

The decoder opened training waveforms only. No Challenge or EchoNext waveform or label was opened, and no
PTB validation/test waveform was opened. PTB release metadata was read to enforce the training-fold guard.
There was no GPU use, model fitting, age-subgroup analysis or label-definition change.

## Counts

| Check | Executed count |
| --- | ---: |
| Training recordings assessed | 17,418 |
| Quality passed | 17,405 |
| Quality excluded | 13 |
| ADC rail reason | 1 |
| Extreme amplitude reason | 10 |
| Constant lead reason | 1 |
| Flat segment reason | 2 |
| Amplitude over 10 mV review flag | 40 |
| Strong baseline wander review flag | 1 |
| Limb identity violated review flag | 2 |

Reasons overlap: `ptbxl:12722` has both `constant_lead` and `flat_segment`. The other excluded records are
`10565`, `11813`, `12251`, `12369`, `12523`, `13317`, `16440`, `16448`, `19299`, `20014`, `20045` and `20926`.
The full-label manifest retains 15,349 of its original 15,360 training records; the limited-label manifest
retains 1,517 of 1,518. These retained counts were computed by joining the executed exclusions to the
recovered seed-42 label manifests.

## Resources and validation

The real 64-record resource profile completed before the full audit: 64 passed, 2.58 seconds elapsed,
parent peak memory 260.7 MiB and largest worker peak 169.6 MiB. Its receipt is
[`profile64/receipt.json`](../outputs/cohort25_recovery_20261028/ptb/profile64/receipt.json).

The full audit used four CPU workers with OMP, OpenBLAS, MKL and NumExpr limited to one thread. It completed
in 342.41 seconds; parent peak memory was 261.6 MiB and largest worker peak was 173.7 MiB. Child CPU usage
was 98.91 user seconds and 8.41 system seconds. These memory figures are per-process peaks, not a measured
simultaneous total.

The two access-boundary tests passed: a held-out fold and an unsafe waveform path are rejected before any
waveform decoder call or output-directory creation. Ruff passed for the new helper, CLI and tests.

The implementation was committed before processing as `633fb52`. Run against a new output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=/tmp/ecg-cohort25-rebuild \
/home/janetrivera/ecg-cardiopathy-detection/.venv/bin/python \
  -m scripts.data.recover_ptb_training_quality \
  --input-root /home/janetrivera/ecg-cardiopathy-detection \
  --output outputs/cohort25_recovery_20261028/ptb/quality_second_run --workers 4
```

The existing quality output is preserved. The runner refuses an existing destination and records paths
relative to the input repository. Frozen files were not edited. A restored whole-cohort manifest and its
original storage pointers remain outside the scope of this PTB-only result.
