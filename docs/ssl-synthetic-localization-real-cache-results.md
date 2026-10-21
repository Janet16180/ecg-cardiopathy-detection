# SSL and synthetic localization: selected real TRAIN cache

18 October 2026. The fixed real-data pretraining cache passed its input, subset-sufficiency and size
gates. All 2,048 prospectively selected patients supplied their originally selected ECG. There were
zero quality skips, zero replacements and zero integrity/decode failures. The local NumPy memmap has
shape `(2048, 12, 2500)`, float32 physical mV, at 250 Hz in canonical lead order. No model trained and
no performance was scored; cache admission does not admit full training.

The patient-selection rule was committed in `530cbb9` before the real waveform audit. The pilot policy's
later clarification at `4f6f268` did not change that rule. The [cache policy](ssl-synthetic-localization-real-cache.md)
was committed as `d217335`, and implementation/tests as `754b02b`, before cache preparation.
Selection used only immutable TRAIN patient IDs: the first 2,048 in SHA-256 order under namespace
`ssl_pilot_v1:real`, with numeric patient tie-breaking, then the lowest numeric ECG ID per patient.
Only afterward was training-quality eligibility joined. The no-replacement rule was not needed on
these selected records. The minimum usable threshold was 1,900 patients.

The original ordered subset's canonical-JSON identity SHA-256 is
`4a7355814061d5fead5ef3a598bbcee89a648c1f1643f96a4989f9885caa5caf`.
The base partition remains unchanged at
`8c47123df83da631765e120cfec8e944a8a2b18a34b14bd0a39ab07061e48e2f`.
The completed input-audit summary remains
`160bc03cd0d476f1679508635f99c2c2affe31bb0e565f1592adbd9dd504e2fa`.
Preparation verified all original audit inputs, sources and outputs, replayed the original partition,
and required exact complete original-row preservation before selecting any payload.

Every selected eligible TRAIN header and signal was freshly hashed against both official checksums
and the input audit's recorded payload hashes. All headers passed the shared explicit-mV, lead, rate,
sample-count and same-stem payload checks before canonical decoding. The command freshly verified
247,005,196 bytes. Whole-record `resample_full(signal, 1, 2)` produced the 250 Hz cache without fitted
normalization, clipping, cropping or per-record scaling. The fixed model-entry division by 2 mV has
not been applied. No validation, engineering-test, excluded or unselected raw waveform was opened by
this cache preparation. Clinical metadata and clinical normality were not interpreted.

The executed command was `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=/tmp/ecg-wt-ssl-inputaudit .venv/bin/python -m
scripts.data.prepare_ssl_synthetic_real_cache`. Preparation ran from 21:12:59.980 to 21:13:11.150 UTC
on 18 October, using one numerical CPU thread and no GPU. Measured wall time was 11.170 seconds and
process CPU time was 11.082 seconds. Runtime versions were NumPy 2.4.6, SciPy 1.17.1 and WFDB 4.3.1;
the receipt also pins the unchanged dependency files and resampler/decoder sources.

All artifacts stay local under `outputs/ssl_synthetic_localization/real_cache_v1`:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `signals.npy` | 245,760,128 | `058721f9dc72428020b219dd457252620da6fae789e0db6628c5fb682b98eda7` |
| `selection.json` | 1,919,426 | `578aaa4f89bcc057ca5e15c903c9d44ec7f5f506cc161a4c18864261e79e2f2d` |
| `summary.json` | 6,117 | `d06ae164fdf6444fe3d82a734fc874172e4e3d5cfdf0020a241c58aa6810cf70` |

Total new cache-directory size is 247,685,671 bytes. The receipt observed 646,036,380 bytes of research
stream outputs before preparation and 893,715,934 bytes before writing its own 6,117-byte summary.
Its conservative size check, including a 100,000-byte summary reserve, passed the 4 GiB ceiling.
Other work can create outputs afterward; a later training profile must check its own complete budget.

The record-level selection manifest retains every selected original row, audit status and eligibility,
quality reasons/review flags, selection digest, fresh payload hashes, and the zero-based cache index.
Post-run verification independently reconstructed the original subset, checked all input/audit-source/
cache-source/output hashes against current bytes, verified TRAIN-only membership and cache indices
0 through 2,047, and streamed the memmap to verify its native shape, dtype and finite values. No raw
waveform was reopened by this post-run check and no record identifier enters this report.

Sixty-seven focused tests passed before the real preparation: 15 new cache tests plus 52 input-audit,
partition and checksum checks. They cover quality-independent selection, lowest numeric record identity,
no replacement or alternate ECG, protected/nontraining access guards, fresh corruption and audit-hash
rejection, explicit units, native geometry/mV preservation, insufficient usable patients, frozen audit
summary binding, and refusal to overwrite. Focused Ruff passed. The two known spectral warnings came
from the earlier artificial all-zero quality fixture; real preparation emitted no warning. This task
did not run the repository-wide suite.

These results establish a qualified cache for this selected subset of previously used PTB-XL TRAIN
patients. They do not establish all-cohort SSL, clinical correctness, unused-domain confirmation,
generator validity or GPU admission. The matched pilot must still satisfy its separate scientific
and real resource-profile gates before training.
