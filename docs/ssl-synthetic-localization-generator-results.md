# Controlled ST intervention generator v1: executed qualification

18 October 2026. All prospective generator-design gates passed for the committed grid: 512 positive
pairs and 256 zero-target controls. This qualifies an engineering intervention on a simulated
background. It supplies no disease diagnosis, clinical annotation, model score or real-data transfer
result. Mechanism B remains unavailable for training, validation and model selection.

The [qualification policy](ssl-synthetic-localization-generator-policy.md) was committed as `9410312`
before implementation or the full run. The executed implementation and 22 focused tests were committed
as `0d2fec16ab3cb1e623ea539ee54e54b6f322eb26` before qualification. This is an independent analytic
compact-waveform implementation; no Experiment 058 source was copied. Shared lead constants,
whole-record resampling, hashing and receipt-path helpers are imported. The pinned 058 report was read
at `8902c1306d090e6fd8d338a47bab5d45965ef52d`; its `outputs/experiment058_synthetic_generator_v1/`
and audit directories are absent locally. Its numbers were not reproduced and are not compared here.

## Receipt and executed grid

The live aggregate receipt is `outputs/ssl_synthetic_localization/generator_v1/qualification.json`,
SHA-256 `869f75ef17dd93995024623f4b747728cf00b62a39ef6f9729f2b0c442684799`.
It records source hashes, the complete prospective configuration, dependency versions and all gate
outcomes. Every recorded source hash was checked against current bytes after the run and matched.

| Executed check | Result |
| --- | ---: |
| Positive pairs: A/B × focal/persistent × 250/500 Hz × 64 seeds | 512/512 pass |
| Sham/no-nuisance and nuisance-only paired controls | 256/256 pass |
| Exact deterministic array and metadata replay | 768/768 pass |
| Independent nuisance/edit substream checks | 512/512 pass |
| Exact final-difference support mismatches | 0 |
| Positive support outside candidate domain | 0 |
| Negative positive-mask cells and paired waveform differences | 0 |
| Deliberately broad mask, empty mask, hard seam, limb corruption, unpaired nuisance | All rejected |
| Zero endpoints and zero endpoint derivative, profiles A/B | Both pass |

All pairs are ten seconds at 500 Hz internally, in mV. The preferred compact-model observations are
250 Hz and have shape (12,2500); the second qualification branch retains (12,5000). Both paired
observations undergo the same deterministic whole-record polyphase FIR. The strict support rule is
`abs(edited.astype(float64)-baseline.astype(float64)) > 0.005` mV. It includes all induced limb-lead
changes, rather than only the directly edited primitive lead. No per-record normalization or clipping
is performed. Baseline and all label/audit metadata must remain unavailable to inference.

The 768 paired examples remain local in `outputs/ssl_synthetic_localization/generator_v1/examples/`.
Each NPZ contains raw and processed baseline/edit, actual final difference, exact support, candidate,
nominal edit domain and all parameters. Total file size is 608,860,846 bytes (580.65 MiB). Their local
index SHA-256 is `4cc2c61b250d532ec5b2f0e2cbf20843ee8962671ee4383a080e406371789819`.
Arrays and per-example parameters were not staged in Git or sent to an external service.

## Occupancy, chance and changes

Each row below describes 64 latent identities. The percentages are median [minimum, maximum].
All 12 lead/time coordinates are the scoring domain; support occupancy is its uniform chance AP
baseline. Conditional occupancy instead conditions on the known all-beat ST candidate domain. This
privileged candidate domain is not a model input and is independent of realized edit lead/beat/profile.

| Profile / mode / Hz | Full-record support % | Conditional candidate support % |
| --- | ---: | ---: |
| A / focal / 250 | 0.905 [0.127, 1.380] | 2.361 [0.322, 5.101] |
| A / focal / 500 | 0.908 [0.125, 1.377] | 2.348 [0.322, 5.114] |
| A / persistent / 250 | 9.927 [1.047, 17.760] | 27.482 [3.785, 41.779] |
| A / persistent / 500 | 9.941 [1.048, 17.763] | 27.469 [3.788, 41.728] |
| B / focal / 250 | 0.865 [0.120, 1.320] | 2.261 [0.309, 4.848] |
| B / focal / 500 | 0.869 [0.122, 1.320] | 2.261 [0.307, 4.823] |
| B / persistent / 250 | 9.492 [1.020, 17.047] | 26.416 [3.650, 40.027] |
| B / persistent / 500 | 9.499 [1.015, 17.050] | 26.349 [3.648, 40.034] |

The 250 Hz candidate domain occupies 23.08–52.92% of all cells (median 36.34%); at 500 Hz it occupies
23.10–52.80% (median 36.30%). These relatively broad candidate windows are explicitly different from
the much narrower actual intervention masks. Persistent support remains under the frozen 18% bound
and focal support under 3% in this grid, rather than covering nearly the whole record.

Change energy is measured as the sum of squared final mV differences; to integrate over seconds,
divide by the scoring sample rate. At 250 Hz, median [minimum,maximum] energies are 3.397
[0.0839,16.315] for A focal, 37.572 [1.091,195.231] for A persistent, 2.008 [0.0496,9.643] for B focal
and 22.208 [0.645,115.395] for B persistent, in mV²-samples. The receipt retains all quantiles and
the corresponding 500 Hz results. These are generated waveform alterations, not clinical effect sizes.

Subfloor changes remain present: at most 0.14119% of total difference energy lies outside the strict
support threshold, below the frozen 1% limit. The largest omitted amplitude is 0.004999161 mV.
At 250 Hz up to 11.3567% of cells have nonzero subfloor differences, including tiny FIR tails; this
count is not a positive mask and should not be interpreted as a physiological extent. Beyond nominal
edit windows, maximum FIR-tail energy fractions are 9.10e-8 for A and 3.30e-8 for B. No tail exceeds
0.005 mV outside the nominal window, and the maximum difference outside the padded candidate domain
is exactly zero. Raw 500 Hz differences have no energy outside the nominal windows.

The maximum limb algebra error after either preprocessing is 5.59e-7 mV (limit 1e-6). Absolute observed
voltage stays below 1.972 mV and the largest edit is 0.337936 mV (limits 10 and 0.50 mV). The largest
amplitude-normalized finite-difference slope across every affected raw/processed lead is 44.624 per
second (limit 80); the largest raw endpoint-transition slope is 3.885 per second (limit 10).
The two edit mechanisms share background and parameter generation; profile B tests a held-out edit
shape on the same background simulator, not an independent physiological simulator or domain.

## Runtime, interface and limits

The full executed grid took 26.494 wall seconds, including compression of local examples and all
replays. OPENBLAS, OMP, MKL and NUMEXPR environment thread limits were one, CUDA was hidden, and no GPU
was used. Python 3.11.13, NumPy 2.4.6 and SciPy 1.17.1 are recorded in the receipt. The focused suite
passed 22 tests in 0.61 seconds and the four new Python files passed Ruff. The runner refuses an
existing output directory; it retains a receipt and exits nonzero when a qualification gate fails.

```python
from ecg_experiment.ssl_synthetic_generator import generate_pair

pair = generate_pair(seed=17, profile="A", mode="focal", sample_rate=250,
                     target=True, nuisance=True)
model_input = pair.edited             # float32 mV, (12, 2500)
training_target = pair.support        # bool, exact postprocessing support
```

Other stable attributes are `baseline`, `candidate`, `metadata`, `raw_baseline`, `raw_edited`,
`nominal` and computed float64 `difference`. Optional `edit_seed` and `nuisance_seed` expose independent
substreams. `target=False` preserves the same background/nuisance baseline and returns an identical
edited copy with zero support. `nuisance=False` is also available for independently balanced positive
and negative training conditions. The generator does not choose data roles; callers must assign
all related seeds/renders to one role and use disjoint latent identities for train/validation/test.
Metadata contains every beat's engineering R/J/ST-end reference, permissible leads and a fixed
ST-prior rule independent of which beat or lead is edited. These are simulation landmarks.

Qualification covers seed IDs 0–63 exactly. It does not prove that every arbitrary new seed passes
occupancy, energy or numerical gates: rate × duration × affected-lead count can produce broader edits
outside this grid. Future generated training/evaluation examples need prospectively specified
admission checks; failures must stop the run and be reported, never silently retried/dropped. Any
needed implementation fix must use a new forward version because this implementation is now pinned.
The design grid is not a model benchmark or an independent patient sample. Physiological realism,
clinical ST boundaries, disease validity, a complete pathology mask, real-data transfer and the
matched SSL benefit remain unestablished. Full training still requires the coordinator's remaining
input/protocol/GPU gates. No reserved final test was opened.
