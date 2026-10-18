# SSL and synthetic localization: permitted waveform input audit

18 October 2026. The input integrity gate passed on all 15,657 permitted train/validation ECGs.
Fresh local header and signal bytes matched the official checksums, headers explicitly declared mV,
and every permitted record decoded to canonical 12-lead, 500 Hz, 5,000-sample physical signals.
No integrity or decode failure occurred. The unchanged recording-quality policy makes 13,927 training
records from 12,072 patients eligible for optimization. All 1,719 validation records remain retained,
including two with technical quality exclusions under the training policy. This is input qualification;
training is still not admitted, and no model or localization performance was scored.

The [prospective policy](ssl-synthetic-localization-input-audit.md) was committed as `7cde696` before
any real header or waveform was opened. Implementation and synthetic tests were committed as `ab2724e`
before execution. The run started at 21:05:09.957 UTC and finished at 21:07:11.144 UTC on 18 October.
The command used `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/tmp/ecg-wt-ssl-inputaudit .venv/bin/python -m
scripts.validation.audit_ssl_synthetic_inputs`. It streamed records with one numerical CPU thread,
used no GPU, and created no waveform cache. Wall time was 121.187 seconds; process CPU time was
106.906 seconds. It freshly verified 1,888,356,060 header/signal bytes.

| Original role | Records | Patients | Integrity/decode outcome | Training optimization eligibility |
| --- | ---: | ---: | --- | --- |
| Train | 13,938 | 12,078 | All decoded; no hard failures | 13,927 records, 12,072 patients |
| Validation | 1,719 | 1,476 | All decoded; no hard failures | Not a training role; all retained |
| Engineering test | 1,761 | 1,469 | Protected and unread | Not assessed |
| Excluded folds | 4,381 | 3,846 | Protected and unread | Not assessed |
| Total | 21,799 | 18,869 | Complete original accounting | No role changes |

The 11 training records with exclusion reasons comprise one ADC-rail record, nine extreme-amplitude
records and one record with a flat segment of at least one second. Thirty-seven training records
have review flags: 36 for amplitude above 10 mV, one for strong baseline wander, and one for violated
limb identity. Flag counts overlap; review flags alone do not remove optimization eligibility.
Validation contains one record flagged for both a constant lead and flat segment, and another for
extreme amplitude. One validation record has the amplitude-above-10-mV review flag. None was removed.
The policy assessed technical waveform quality without interpreting diagnosis, report, age or sex.
It does not establish clinical normality or label correctness.

Before payload access, exact partition replay and every recorded original-source hash passed.
The earlier declared-checksum duplicate audit reproduced exactly: 21,799 unique declared signal-file
digests, zero duplicate groups, zero cross-patient groups and zero cross-role groups. The immutable
partition hash remains `8c47123df83da631765e120cfec8e944a8a2b18a34b14bd0a39ab07061e48e2f`.
The official checksum table hash remains
`b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d`.
No engineering-test/excluded header or waveform, Challenge, EchoNext or SPH input was opened.

The local aggregate receipt is
`outputs/ssl_synthetic_localization/input_audit_v1/summary.json`, SHA-256
`160bc03cd0d476f1679508635f99c2c2affe31bb0e565f1592adbd9dd504e2fa`.
The complete record-level audit is `records.jsonl` in that directory, SHA-256
`0c8030a3d429aeaab5078d146462dffb20464f2c2688c4ab0745e1862a120555`.
The derived `eligibility.json` hash is
`0f9628f39c272fe907d777bb7bb8f3fa49f949155a34d049cdacfc392fa95814`.
Both record-level outputs stay local. They preserve every original record, patient and role with
permitted-file hashes, quality reasons, review flags and eligibility; protected rows have no payload
hashes. The base partition is unchanged. The output directory refuses overwriting, and the aggregate
receipt was written only after the complete audit. Total output size is 23,312,623 bytes.

After execution, a metadata-only verification confirmed exact original-row preservation, equality of
the JSONL and eligibility records, all 6,142 protected rows without payload hashes, retention of every
validation row, and all recorded input/source/output hashes against current bytes. The runtime was
NumPy 2.4.6, SciPy 1.17.1 and WFDB 4.3.1. Fifty-two focused tests passed before the real run, including
20 new artificial-input tests plus existing partition/checksum integrity checks. Focused Ruff passed.
The artificial all-zero recording produces the shared quality helper's two expected all-NaN spectral
warnings; the real audit emitted no such warning. The repository-wide suite was not run by this task.

Input integrity and recorded training-quality eligibility are established for these permitted local
bytes. Generator qualification, the committed executable pilot, predecessor integrity for any reused
comparison, and a real GPU profile remain separate gates. Protected waveform quality, near duplicates,
biological identity and clinical truth remain unassessed. Engineering-test patients come from previously
used PTB-XL and do not establish independent clinical confirmation. No clinical relabeling, role change,
training admission or final-test authorization follows from this audit.
