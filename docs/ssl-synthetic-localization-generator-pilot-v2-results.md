# Generator v2 admits the declared pilot domain

18 October 2026. The forward v2 generator passed every frozen gate on all 3,060 planned pilot pairs,
with zero exclusions, retries or failures. Its exact independently seeded pilot-domain admission
supersedes the failed v1 admission for this pilot. These are simulated intervention-truth checks;
no model was trained or scored, no real waveform/header/label was opened, and no clinical claim follows.

The [policy](ssl-synthetic-localization-generator-pilot-policy.md) was committed as `cebb138` before
the v1 admission or its forward contingency. V1 then failed 28 persistent-occupancy cases: nine training,
five validation, nine evaluation A and five evaluation B. Its maximum was 22.96% against the frozen 18%
limit. The [v1 failure report](ssl-synthetic-localization-generator-pilot-v1-results.md) and original
failed receipt remain intact. No v1 source or threshold was changed.

V2 was committed as `fde7bc313c07bc91eca28f44e2ecf36ada2c26f1` before the full admission run.
It preserves the exact v1 background and nuisance arrays and the same signs, amplitude draws,
selected beats and A/B profile formulas. It applies the prospectively declared narrowing to every
role, mode, profile and target state: width maps uniformly to 0.14–0.19 seconds; limb edits retain
one uniformly selected originally proposed chest coordinate. Chest-only edits retain one to three
chest coordinates. At most six derived leads change; at most 17 latent beats and a 0.19-second width
bound the continuous nominal persistent domain by 16.15%. Actual postprocessing support still passes
all original numerical gates; the nominal bound is not substituted for measured support.

## Source-bound admission

The final aggregate receipt is `outputs/ssl_synthetic_localization/generator_pilot_v2/qualification.json`,
SHA-256 `452ec35407527fda26bcfb1fa22532783849e0c4b266825c6684a52cc1e2b2ef`.
Every recorded source hash was freshly compared with current bytes after execution and matched.
The source hashes include the v1 dependencies, new v2 generator, added tests, both admission runners
and prospective policies. The local complete per-case parameter/diagnostic/hash audit is
`outputs/ssl_synthetic_localization/generator_pilot_v2/local_case_audit.json`, SHA-256
`dab5cb3b49c781d6fea860efee4076728abfe756580046137d29879fad046dad` (13,864,604 bytes).
It contains no model prediction. The earlier v1 design examples remain local and are historical
v1 examples; the final pilot bank must be generated with v2, not silently reuse those v1 arrays.

| Cohort / profile | Latent identities | Pairs | Failed gates |
| --- | ---: | ---: | ---: |
| Training / A | 500 | 500 | 0 |
| Validation / A | 128 | 512 | 0 |
| Evaluation / A | 256 | 1,024 | 0 |
| Evaluation / B | Same 256 | 1,024 | 0 |
| Total | 884 distinct identities | 3,060 | 0 |

All 3,060 exact array/metadata replays passed. There are no duplicate latent seed IDs within or across
roles. Related focal, persistent, sham, nuisance and A/B renders retain their background identity and
role. Seed allocation is exactly the first eight SHA-256 bytes, unsigned big endian, of
`ssl_pilot_v1:synthetic:<role>:<index>`, with the declared role/index counts. No independent patients
are implied by multiple related simulated renders.

Training has 188 focal and 187 persistent positives, 62 clean shams and 63 nuisance-only negatives.
Positive nuisance is the independent `bool(seed & 1)` coin allocation, not forced exact balance:
171 nuisance-present and 204 clean training backgrounds, 59 and 69 validation backgrounds, and
120 and 136 evaluation backgrounds. Each validation/evaluation background receives both positive
modes with the same nuisance flag and both zero-target controls. A is the only training/validation
profile; B was used here only for generator-design admission and remains excluded from learning,
threshold selection and model selection.

## Support, tails and artifact gates

Scoring uses lead-major (12,2500) float32 mV, ten seconds at 250 Hz; each example's raw (12,5000)
pair is also checked. Both observations have the same whole-record FIR preprocessing. Support is
still the strict final-observation difference greater than 0.005 mV, retaining every induced limb
change. The following support percentages are median [minimum,maximum], on all 12 lead/time cells.
They are the relevant uniform full-domain chance AP baselines, not model results.

| Cohort / profile | Focal support % | Persistent support % |
| --- | ---: | ---: |
| Training / A | 0.713 [0.110,0.893] | 7.840 [0.860,14.727] |
| Validation / A | 0.713 [0.113,0.893] | 7.537 [0.897,14.933] |
| Evaluation / A | 0.733 [0.107,0.893] | 7.217 [0.940,13.747] |
| Evaluation / B | 0.700 [0.100,0.873] | 6.917 [0.907,13.100] |

The fixed all-beat/all-lead padded candidate domain occupies 22.96–56.32% of the record. Conditional
candidate support remains separately recorded in the receipt: median focal/persistent percentages
are 1.793/22.525 for training A, 1.706/21.780 for validation A, 1.691/22.204 for evaluation A and
1.624/21.141 for evaluation B. Candidate and fixed ST-prior metadata depend on all background beats,
not the realized edited lead/beat. They are privileged audit/prior information, never network input.

The maximum omitted subfloor difference is 0.0049999574 mV; at most 0.335403% of total change energy
is below the support floor, under the frozen 1% limit. Up to 8.9433% of all cells carry nonzero
subfloor differences, including small FIR tails. Maximum change-energy fraction beyond nominal edit
windows is 1.041e-7, and no above-floor support extends beyond nominal windows. The maximum difference
outside the padded candidate domain is exactly zero. The receipt retains every energy quantile in
mV²-samples; division by 250 converts the sum to a time integral in mV²-seconds.

All shams and nuisance-only controls are exactly identical pairs with zero support. Exact support
reconstruction, candidate containment, waveform finiteness and shape gates pass every case. Maximum
limb algebra error is 7.153e-7 mV, maximum observed magnitude 2.063 mV and maximum intervention
0.353202 mV. Maximum amplitude-normalized slope is 44.790 per second and maximum raw endpoint-transition
slope 4.082 per second, below the original 80 and 10 bounds. The 39 combined focused tests pass,
including v2 broad/empty-mask, hard-seam and corrupted-limb controls, independent streams, preserved
backgrounds, candidate independence and shams. New Python files pass Ruff.

## Reproduction and scope

The full v2 admission took 38.161 wall seconds with one OPENBLAS/OMP/MKL/NUMEXPR numerical thread,
CUDA hidden and no GPU. The focused tests took 0.80 seconds. Python 3.11.13, NumPy 2.4.6 and
SciPy 1.17.1 are recorded in the receipt. The full v1 plus v2 admission times are measured separately;
training, compilation and GPU costs are not inferred from them.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
.venv/bin/python -m scripts.validation.qualify_ssl_synthetic_generator_pilot_v2
```

The command refuses an existing output directory. Final generator API:

```python
from ecg_experiment.ssl_synthetic_generator_v2 import generate_pair
from ecg_experiment.ssl_synthetic_generator_qualification import gate_failures

pair = generate_pair(seed, profile="A", mode="focal", sample_rate=250,
                     target=True, nuisance=bool(seed & 1))
# config is configs/ssl_synthetic_generator_v1.json; thresholds are unchanged.
failures = gate_failures(pair, config)
# Abort on any failure; use only pair.edited as network input and pair.support as truth.
```

The dataclass API is unchanged: `baseline`, `edited`, `support`, `candidate`, `metadata`,
`raw_baseline`, `raw_edited`, `nominal` and actual float64 `difference`. Its metadata identifies
`generator_version="v2"`; original width is retained for the narrowing audit. No clipping, independent
paired normalization or target-dependent nuisance was added. The 3,060 qualification cases are exactly
the planned pilot domain; a newly proposed seed set or preprocessing still requires admission, with
failures retained rather than dropped. Root-level preparation also rechecks every generated case.

A/B share the same background simulator and differ only in the edit profile. B therefore tests a
held-out editing mechanism, not an independent physiological generator or clinical domain. No real
disease validity, real-data localization, complete clinical pathology support or SSL benefit has been
established. Experiment 058 artifacts remain absent and were not reproduced or numerically compared.
The remaining real-input, protocol and GPU gates remain the coordinator's responsibility. No reserved
clinical final test was opened.
