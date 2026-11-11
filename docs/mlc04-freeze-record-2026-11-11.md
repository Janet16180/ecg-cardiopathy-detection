# MLC-04 freeze record

11 November 2026. Step 2 of the frozen [MLC-04 protocol](mlc04-final-test-protocol-2026-11-11.md) (commit
`fbf26bf`). The code, the frozen files and the dry run are fixed here, before any Challenge test-group record
is opened. Until this file is committed, every loader of the runner refuses the role `challenge_test`
(`mlc04_data.check_role`): the file must be tracked by Git, the tracked tree clean, and the code commit named
in the JSON block below an ancestor of `HEAD`. No header, waveform, label or feature row of a test-group record
has been opened. The runner has read only the identity columns of the split table (`read_split_table`), to
name the 13,172 test records and to close them to the dry run's readers.

The labels are ECG-annotation proxies, not diagnoses. No cardiologist has reviewed anything; clinical
validation is pending.

## Code

Code commit `89e2f3a`:

- `ecg_experiment/mlc04_data.py`: role guard, records, MLC-01 scan and labels, cache, Part A, method B, windows,
  rules C1-C12, marker;
- `ecg_experiment/mlc04_models.py`: extractors, frozen models, thresholds;
- `ecg_experiment/mlc04_evaluation.py`: metric, intervals, comparisons, readings, descriptive tables;
- `ecg_experiment/mlc04_checks.py`: the dry-run reproductions;
- `scripts/experiments/run_mlc04_final_test.py`: the runner;
- `scripts/reports/mlc04_tables.py`: the report tables;
- `tests/test_mlc04_final_test.py`: 17 tests, synthetic data, one skipped without the local split table.

At that commit, `ruff check ecg_experiment scripts tests conftest.py` passes, and the 17 tests pass. Every
frozen module is imported, not copied. Two pieces restate frozen orchestration that could not be called
directly, and the dry run checks both:

- `mlc04_data.assemble` repeats the steps of `mlc01_build.assemble` with the frozen minimum-positive table;
- `mlc04_data.method_a` repeats the derivation of `mlc03_dataset_v5.measurements_a` on given arrays.

## Dry run (validation, the Challenge calibration groups)

The dry run ran at `89e2f3a` with no modified tracked file (receipts in
`outputs/mlc04_final_test_20261111/receipts/`). Its output is
`outputs/mlc04_final_test_20261111/validation/check.json`, printed by
`python -m scripts.reports.mlc04_tables --section check`. Every check passed (`failed: []`):

| check | result |
|---|---|
| MLC-01 scan of the 13,162 calibration records against v4's `waveform_scan.parquet` | equal |
| v4 manifest rows, 13,162 records, 122 columns, 1,605,764 cells, as text | equal, same order |
| v5 manifest rows, 13,162 records, 174 columns, 2,290,188 cells, as text | equal, same order |
| waveform cache, 13,150 eligible records, against the MLC-02 cache | bit-identical |
| Part A measures and flags, 13,150 records | equal |
| method B, 12,955 included records; window summaries, 1,181 records longer than 10 s | equal |
| method A from the re-extracted measurements, against `measurements_a` on the stored ones | equal |
| analysis table, 12,955 records, against Part B's `analysis.parquet` | equal |
| rule flow, 170 rows, against the v5 build's `rule_flow.csv` | equal |
| re-extracted xECG | largest absolute difference 9.54e-06, 13,136 of 13,150 records bit-identical |
| re-extracted ECG-JEPA | largest absolute difference 2.38e-07, 13,148 of 13,150 bit-identical |
| re-extracted ECG-CPC and the 130 measurements | bit-identical, 13,150 of 13,150 |
| stored validation outputs from the stored features: P0-v5, P0-v4, H-All-v5, H-All-v4 | at most 4.44e-15 |
| the same for K6 (15,097 records) | at most 3.46e-14, relative 2.68e-16 |
| stored validation outputs from the re-extracted features (12,916 or 12,955 records) | at most 5.39e-07; K6 relative 1.68e-07 |
| P0-v5 validation metric, stored features, and with the re-extracted Challenge rows | 0.9697697420282113 both, difference 0 |
| thresholds refitted | equal to the frozen file |

The first dry run, with uncommitted code, differed from the stored Part A and method B values by about 1e-10.
Those MLC-03 steps had run with the machine's default OpenBLAS kernel and the dry run with Haswell. The runner
now requires each step's kernel (reading 6 below). That run is kept, unused, in
`outputs/mlc04_final_test_20261111/superseded_first_dry_run/`; the dry run above is a complete rerun at the
code commit.

No calibration record shares its first-10-s hash with a v4 train record (the scan's cross-role count is 0 for
every source).

## The test metric on validation, and the protocol's comparator

On the calibration groups, the evaluation the test read will run gives (`--section models`, `--section
sources`):

| model | five sources, entry 100 (95% interval) | five sources, entry 20 |
|---|---|---|
| P0-v5 | 0.9611 (0.9557, 0.9663) | 0.9701 |
| P0-v4 | 0.9604 (0.9549, 0.9656) | 0.9692 |
| H-All-v5 | 0.9644 (0.9589, 0.9696) | 0.9752 |
| H-All-v4 | 0.9660 (0.9606, 0.9710) | 0.9751 |
| K6 | 0.9588 (0.9532, 0.9640) | 0.9644 |

Under the 100-positive rule, CPSC-Extra enters with any MI alone (P0-v5 0.8914), Chapman and CPSC 2018 with
four conditions each, Georgia with eight and Ningbo with eleven.

The protocol reads the test metric (five Challenge sources, entry 100) against 0.9698, the six-source
metric with entry 20 that includes PTB-XL fold 9. These are not the same quantity: the same five-source,
entry-100 metric is 0.9611 on the calibration groups. The bound of the reading, 0.9598, therefore lies inside
the like-for-like validation interval. An "undecided" or "not confirmed" reading can follow from the change of
definition alone, without any optimism of the validation numbers. The protocol is explicit here: it names the
value (0.9698) and the rule. The test read therefore applies it as written, and the report prints the
like-for-like validation value (0.9611) beside it, with no second reading. The orchestrator was told before
this record was committed, and no amendment arrived.

## Readings made before the freeze

Where the protocol's wording leaves a choice, the reading closest to the MLC-02 and MLC-03 code was taken.

1. **Role names.** The runner's roles are `validation` (the Challenge calibration groups, the dry run) and
   `challenge_test` (the test groups). In the role manifests the test groups carry MLC-01's role name for a
   same-source test, `test_same_source`, with origin `test`; the MLC-01 duplicate rule and the v5 readers take
   that name unchanged. PTB-XL fold 10 is not part of any MLC-04 output.
2. **Minimum-positive rule** (MLC-01 Amendment 1, ruling 6). The frozen v4 table (`minimum_positive_rule.csv`)
   is applied; positives are not recounted with the test records. CPSC-Extra stays unannotated for left axis
   deviation and LAFB, Georgia for any MI.
3. **Duplicates.** The MLC-01 rule is applied inside the role. The split assigns every duplicate group (signal
   or window hash) to one group, so this equals the rule over all records. A test record whose first 10 s
   equals a v4 train or validation record's is counted and reported, not removed: no model is refitted.
4. **Screening labels withheld.** The MLC-01 label code also produces the space-A groups and the `normal`
   column, in memory, because its duplicate rule compares every label space. For the test groups those
   columns are emptied before anything is written. No file then holds the ingredients of the binary screening
   label (any MI, STTC, CD or HYP code against sinus rhythm alone), and no binary label or screening score is
   computed.
5. **Features.** `outputs/features_challenge_v1/` does not exist on this machine, and it read the centred 10 s
   window rather than the first 10 s every MLC-01 model sees. Features are therefore extracted now by the
   MLC-02 code (`mlc02_features.run_encoder` and `run_measurements`), from a cache built by the MLC-02 cache
   code. All three encoders run on GPU 0 under the shared lock (MLC-02 ran ECG-JEPA and ECG-CPC on GPU 1).
6. **OpenBLAS kernel.** Each step uses the kernel of the code it reproduces, and the runner refuses another
   setting:
   - the MLC-03 steps (`measure`, `rules`) use the machine's default kernel, as their receipts record;
   - the MLC-02 extractors and every model score (`features`, `check`, `score`) use
     `OPENBLAS_CORETYPE=Haswell`.
7. **Primary reading.** "Below it" is read as below the validation value minus 0.010, the same bound as the
   confirmation rule; otherwise the three outcomes would overlap. The bound is 0.9697697420282113 - 0.010 =
   0.9597697420282113.
8. **Metric.** The metric is averaged over the Challenge sources where at least one condition has 100 positives
   and 100 negatives in that source's test records. The validation value (0.9698) is the six-source metric
   with the 20-positive rule, as the protocol states (see the section above).
9. **Intervals.** They come from `mlc02_bootstrap.cluster_resamples`, built on `intervals.patient_groups` and
   `patient_resample`:
   - within each source, each record its own group;
   - 2,000 draws, seed 20261110 (MLC-02's);
   - a draw is redrawn if an entering condition loses its positives or negatives;
   - the metric's draw is the mean over sources of the draw's within-source macro AUROC, and every model uses
     the same draws;
   - percentile intervals.
10. **Holm.** Each secondary comparison has a two-sided bootstrap p-value of its difference against its null
    value (`mlc02_bootstrap.bootstrap_p_value` of the draws minus the null). The null is 0 for cleaning (v5-rule
    labels) and for H-All, and -0.010 for K6. The three p-values are Holm-adjusted, and a comparison is read as
    above or below its null when its adjusted p-value is below 0.05, in the direction of the observed
    difference.
    - Cleaning helps only if its v5-label comparison is above zero and its v4-label 95% interval is not below
      zero (upper bound at least zero). The v4-label interval is a condition, not a fourth test.
    - K6 is equivalent when above -0.010, clearly worse when below it, undecided otherwise.
11. **v4 labels.** "On v4 labels (no rule applied)" means the v4-included test records with their v4 labels:
    no exclusion and no label rule of v5.
12. **K6 on Challenge records.** Each record is centred on its train source's mean (every Challenge source has
    one). Scores are mapped to ranks within the evaluated set, which changes no AUROC or average precision.
13. **Thresholds.** They come from P0-v5's stored validation probabilities and the v5 validation labels, all
    six sources pooled, by `mlc02_evaluation.specificity_thresholds` (90% specificity). The file is
    `outputs/mlc04_final_test_20261111/freeze/thresholds.json`, listed with its hash below.
14. **Descriptive intervals.** Per source and condition, they come from record resamples with the 20-positive
    rule; cells below it get point values only. Calibration is reported as:
    - the Brier score per source and condition, with its interval;
    - a 10-bin reliability table, with its mean gap and its largest gap (bins of at least 20 records).
15. **Marker.** The score step writes the marker after reading the labels and before the first model output,
    and refuses a second read. A crash after the marker allows a rerun only with `--rerun-after-crash`, once
    the partial outputs are deleted unread; the marker keeps every attempt.

## The test read

After this record is committed, from the worktree (`/var/tmp/janetrivera/mlc04/run_role.sh challenge_test`
runs the first six steps in this order, each with its kernel):

```
scan, cache, measure                 default kernel, 8 processes
features xecg, jepa, cpc             GPU 0, Haswell, 2 threads
features measurements                Haswell, 6 processes
publish, rules                       default kernel
score --role challenge_test          Haswell, once
```

Nothing is fitted, tuned or chosen during or after the read.

## Frozen files

`code_commit` is the commit the guard checks. The hashes come from `run_mlc04_final_test freeze-draft`
(`outputs/mlc04_final_test_20261111/freeze/freeze.json`, the same content with the validation entering lists
left out). They cover:

- the code: the five MLC-04 files and every frozen module the runner calls;
- every model artefact of P0-v5 and H-All-v5 (MLC-03 Part D) and of P0-v4 and H-All-v4 (MLC-02);
- the KEY-05 keys;
- the thresholds;
- the stored validation outputs;
- the stored MLC-02 features and the dry run's re-extracted features;
- the encoder checkpoints;
- the split table, the v4 manifest and minimum-positive table, and the v5 manifest.

```json
{
  "experiment": "MLC-04",
  "code_commit": "89e2f3af8189e2db7c387248d86b1e3e4075660a",
  "code_sha256": {
    "ecg_experiment/mlc04_data.py": "8854cb33fa9aa68dcd426f021870922679cb28cb18e74babd81ba55addb39d7b",
    "ecg_experiment/mlc04_models.py": "86b5a5cf9e78b4160f7511e3921a1c3fb0599f35ed063ac31136776c67ef4be2",
    "ecg_experiment/mlc04_evaluation.py": "a696674c2ef409e55278a35f0fd34bfdda89d539748dbba3cc100dbe9d0c8a4b",
    "ecg_experiment/mlc04_checks.py": "207768a31e95315282bed7106b71259317f881ee331f5bc2621768c69a6165e0",
    "scripts/experiments/run_mlc04_final_test.py": "c2ffd9ce7d066b66a4154e9f74f717a2d89e714616bd7d4af0cb3ee7e5c8bae4",
    "ecg_experiment/mlc_labels.py": "a4221dc76ef1e04ff8447c93e0116c379085fe240d9c9638e7b9e4788cf68a84",
    "ecg_experiment/mlc01_dataset.py": "c421be0778fcbfcc8d95cf01a9758bdabfbd82b7daef5384396d4b12460f7e36",
    "ecg_experiment/mlc01_build.py": "a36fc4f10ecdff1081265811904cf5cdbdbd9090466bd436b3b40dfdddb4eb54",
    "ecg_experiment/mlc02_waveforms.py": "3ee3e615b5fb1b4b54350edff262000e465138d16ee6b02b15ecfca82552de3a",
    "ecg_experiment/mlc02_features.py": "3c71c7a882a563bb3cd9b292315b0242350196b3a602070ddc26d4eb1084f8de",
    "ecg_experiment/mlc02_cpc.py": "bb946a82c457ace89d3d480d1ac8461d4830ccc4f66dff6a94bfab3ac6fc7e52",
    "ecg_experiment/external_encoders.py": "b21784379805e126fb5f0068c0cef53efb346bffe1ccd9f15f7fced9306a7c48",
    "ecg_experiment/measurement_extractor042.py": "7cabf8fb9670119e577ad6eb017feb7e663a3f00f107ae527c8eac8ca4c2b88b",
    "ecg_experiment/mlc02_feature_models.py": "20a72cf6c854d5e34f3b2b3e61290dd1495f4afc6f732c778e7916661a317bb3",
    "ecg_experiment/mlc02_evaluation.py": "b8219eeb313be523e32a225df41cc62c8296fa18bd61143a05e2b1c389bb1a75",
    "ecg_experiment/mlc02_bootstrap.py": "9932690bf5cf4068d98c765527ab4809407cc9d1d8e5ab1a3a16907f3276acf7",
    "ecg_experiment/intervals.py": "b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af",
    "ecg_experiment/mlc03_signal_quality.py": "922d185783532e4f859677dc9935d99cf3f61cd9531db615268440a9d2b77ee7",
    "ecg_experiment/mlc03_signal_flags.py": "44eeb3b524885980d2b4338ace05995de8e816aef6dcac298bfde0b70501934f",
    "ecg_experiment/mlc03_measurements.py": "30c532d00e00c91d33cbb63a9cf7d1d782c52012abdf410e6c2f7c0386b828ee",
    "ecg_experiment/mlc03_label_quality.py": "6287a990963b995e172769b01b94521a8b14319588f64b5b68d8a6ffc217cc33",
    "ecg_experiment/mlc03_dataset_v5.py": "d250f517e9428d3f054ced1b9f94b65d35095606c64a86d238844a313f3c6fbb",
    "ecg_experiment/key05_keys.py": "d67ca7c42e29eb2b6b7e4038a47c5f013b8ab856fabff10fcbdc81448b1fb1b7",
    "ecg_experiment/key05_analysis.py": "de57c930faa4ed4b6ed994cda27e922716408d548233048136c0cd1fed545fb5"
  },
  "artefacts_sha256": {
    "outputs/mlc03_data_quality_20261111/effect/candidates/P0/candidate.json": "677c20b44e0f345ac5ecfb55b2d75a4e1024233030eb9fe4cbc42826031f9ddb",
    "outputs/mlc03_data_quality_20261111/effect/candidates/P0/logistic.npz": "937a2b119dc6536e2d2ba7d2d30e2a6b641eb454ee86019632081f35ee20c378",
    "outputs/mlc03_data_quality_20261111/effect/candidates/P0/model.json": "0dd10c40bdc7ed12d15bcbfa1d06e3328160b23c44eefa06f601e5a115e436ee",
    "outputs/mlc03_data_quality_20261111/effect/candidates/P0/scaler.npz": "c4927dff6ac6e7c5e9897efb11e585ba84c93d4b952163913c68edaa26af648d",
    "outputs/mlc02_condition_models_20261110/candidates/P0/candidate.json": "3f44f3c76aedbb15d31f221d54c52d46e4aec08cbff7e4cf4f70bc6ef58d3ab0",
    "outputs/mlc02_condition_models_20261110/candidates/P0/grid.npz": "23e53b4cad97784b2d07e4a57f9db0e377273d55036938427c6c0c550ee6718a",
    "outputs/mlc02_condition_models_20261110/candidates/P0/logistic.npz": "8365ea05fbb7b7bcafe7f0ba3f27eb3a9b61195859ab0f64e0b1f0afcb1d22d5",
    "outputs/mlc02_condition_models_20261110/candidates/P0/model.json": "4c3d23a470514d23b35981049c74aeeb9e784c87e7e77c38ad1fe005612ecb28",
    "outputs/mlc02_condition_models_20261110/candidates/P0/receipt.json": "1c89b00f6603e325fad42d9b0f1227c640c3b567efc97a65fd3b6db112d2cc27",
    "outputs/mlc02_condition_models_20261110/candidates/P0/scaler.npz": "077b6cae9868ae7f57fc1ea214c41db026d4633bc5974e6a66cab02520241830",
    "outputs/mlc02_condition_models_20261110/candidates/P0/training_log.csv": "fc13bba1151b78a2f6ed70d1bf98b33f48ff6535b945e45dd9811d8a0ffb04c5",
    "outputs/mlc02_condition_models_20261110/candidates/P0/validation_by_source.csv": "07d1d11ae162ff1a0a41cc12fd96a861b90e06881b08a108a2effd3a5a84149f",
    "outputs/mlc03_data_quality_20261111/effect/candidates/H-All/candidate.json": "a38c37b8a6ef41ad9c28513292a707d1d79b4520cb95a670e6ceb6a90f73e7ab",
    "outputs/mlc03_data_quality_20261111/effect/candidates/H-All/imputation.npz": "89a7232e0896e147bf2595e27ab9b8bb050b14da1885b0d2dbbafdc0c8d882cc",
    "outputs/mlc03_data_quality_20261111/effect/candidates/H-All/model.json": "baf5409827e9ecce0ed72a9d080e843bd07d8fab2c7faee5405bd31e144e8ea8",
    "outputs/mlc03_data_quality_20261111/effect/candidates/H-All/profile.json": "5f47d3787b3999b150be7609892ee2e42801b32e06f3dadf5fd7f5e38338464d",
    "outputs/mlc03_data_quality_20261111/effect/candidates/H-All/scaler.npz": "dd06f6a2a3ae6adef3d6043752d6ddf711a5f47700167042010caecf8fb206e4",
    "outputs/mlc03_data_quality_20261111/effect/candidates/H-All/weights__seed-20261110.npz": "55cbd906a2713c8cc4c743802cdd00538efd13dd0945a37bcade33bfd48576c6",
    "outputs/mlc03_data_quality_20261111/effect/candidates/H-All/weights__seed-20261111.npz": "be91de986f03270e5e7d73565d129c5e8d6c69f06f32d40a953e1f505d279d03",
    "outputs/mlc03_data_quality_20261111/effect/candidates/H-All/weights__seed-20261112.npz": "4f6bf95a6412e92cf7172075f6fa04ac027e7ba43220e5abe3f18900bc1dd75d",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/candidate.json": "f03dbda3486e7b5eb6e183fa04d3a7cb7da0c4eb66e1023ca889954ac6ad0f99",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/imputation.npz": "cd9861f6115b56cb9cb57dae4733e098f8cff36198b487ca52d28935976062c2",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/model.json": "9051379eade444846f84ce58a557352913cdb3c8297479982ac892b0ff55cf12",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/profile.json": "09359a8e12f76f4beade93088b3319763a027c76299c138efa32f8a196d17938",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/receipt.json": "8fd022011aa36df64470bd36e3857689b23287388f338f6240f9ab851ac56905",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/scaler.npz": "c434a0af1e3c49f9f5870104e3c5b9889097e9fc4ecbd6aaae6af30aaa94272d",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/training_log.csv": "d4ce8b9e62d3105eb5d634d98fcb6d4a8014d3aafb06eee3d3e6e795beecf6ea",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/validation_by_source.csv": "9fe4a56284254ba48ab3b9d19840815a4646afd628072ca142d7cd1c31e9fa8b",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/validation_by_source__seed-20261110.csv": "ecb01505bef93e1dbeb3bf32716dd7e153e09d7df747627b89bb5d8b61ebf109",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/validation_by_source__seed-20261111.csv": "7d2010ba9a2e160d2e2c05776c8963567f7f95b1174fe251e9c1082cf87fbb4a",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/validation_by_source__seed-20261112.csv": "276e3c70df80b5874d517bf652dec2add1c4b114f54e907e85a2bfe907d4ee13",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/weights__seed-20261110.npz": "2acd1199e93e0d57dec8054390e639ecf85964e04012b61f374ca5729a1754a8",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/weights__seed-20261111.npz": "6e32b432ffe690218bcf1440fb25d9281ee446e462f8d95d8423d3c6882e3aad",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/weights__seed-20261112.npz": "3babbeb700b80f0b53a7f62c3101361145b1ca787f3dfb2f5852d47e7e50d4ef",
    "outputs/key05_condition_keys_20261111/keys.npz": "beb0ba4dcf04010b6de733a7c5d78dbdb73e643bb7ab616f2370e5d236e5e77d",
    "outputs/mlc04_final_test_20261111/freeze/thresholds.json": "f607f117f6dc27bb2337d49032f97a6523b13fc8b1a83c1263d508cc0bd70f4a",
    "outputs/mlc03_data_quality_20261111/effect/candidates/P0/validation__seeds-42.npz": "0a48d83bcfb334200f475261e43b39c6ccc4815434b7b41cde1b2753d6348313",
    "outputs/mlc02_condition_models_20261110/candidates/P0/validation__seeds-42.npz": "fe40ab62bdb4e7ad078a389f810d6437fb7deb47b28c72b3e00364bb0ebd13d6",
    "outputs/mlc03_data_quality_20261111/effect/candidates/H-All/validation__seeds-20261110-20261111-20261112.npz": "2cc21a176eafd2b145a444832aff31041a3b158631382df2deaa84fd2abbb067",
    "outputs/mlc02_condition_models_20261110/candidates/H-All/validation__seeds-20261110-20261111-20261112.npz": "b8ab1bb08f00ce553d370d7293f3ab2c82b39a0536f437a2177c2917c12d8cc8",
    "outputs/key05_condition_keys_20261111/scores_validation.npz": "5f4af776e981c00df912175a88f5b4f7d91998ae96a8be8e7201785b7156906e",
    "outputs/mlc02_condition_models_20261110/features/xecg.npy": "355d290208d562b9990d0701f3a459986cce6a633483dd0aad59560e7bdde7e8",
    "outputs/mlc02_condition_models_20261110/features/jepa.npy": "f9ebe07d652877ac13585738697db6969470f9bc76696f386639744c05415be7",
    "outputs/mlc02_condition_models_20261110/features/cpc.npy": "9aa8a3d29b32581434d4a9eda32913374232a797be0111951d819811e30731c1",
    "outputs/mlc02_condition_models_20261110/features/measurements.npy": "75e7e762613615b2a91dfaac6c099d4e727f6a5aa3db2b9e27fe54d9570bca88",
    "outputs/mlc02_condition_models_20261110/features/rows.csv": "c6a2cd64f3cc9262630019235f4c11ab37885dbf82d701378c48b4f22e0265df",
    "outputs/mlc02_condition_models_20261110/features/measurements_status.csv": "010471c1ed714ced08d69022fb9076d4463790e5306976027f4835ec71ef6cfe",
    "outputs/mlc02_condition_models_20261110/features/receipt.json": "acc166d3a7e05523e7499485232dc7252682ba8b9cca2fe8ab8944f3c8e88073",
    "outputs/mlc04_final_test_20261111/validation/features/cpc.npy": "85a47187c1f69124f45d09d0f4bd0c72797088e05a97e6023f53b49ad3423b0b",
    "outputs/mlc04_final_test_20261111/validation/features/jepa.npy": "21e92d4b7df557fcc4b15fb562ac05e0acf5b1fe4bf60cfa9d444074465190b2",
    "outputs/mlc04_final_test_20261111/validation/features/measurements.npy": "d3d59f6596baabf92993e94842c7b01750e7cbc19f3a8cdb756d91b98ace1d0e",
    "outputs/mlc04_final_test_20261111/validation/features/measurements_status.csv": "224eb631c37c11032afca1f645cbef699a67869d3d0484682af4ff0830152b92",
    "outputs/mlc04_final_test_20261111/validation/features/rows.csv": "6aec6db6881b6c943f9dd462f295fe0dc09a29ee4a46995ce22ee6a7070964de",
    "outputs/mlc04_final_test_20261111/validation/features/xecg.npy": "63957acf22cf824cdf8b0b04b421b24fd1a1b2f7d165741491253780c1365cc8",
    "third_party/checkpoints/xecg/config.json": "23ba68fc4edc5883f4a89a55f8da3d866ab040e64a46c9c862fde4bb52459b4a",
    "third_party/checkpoints/xecg/xECG.py": "373fed5a125abea218c74a8dc5b09e6054e85172a91f99e2ed47f8221fbe266f",
    "third_party/checkpoints/xecg/model.safetensors": "812dec69ac0fbf13f39e50bde4f35f85435a66966d8785baec65c2b5c70e722c",
    "third_party/checkpoints/ecg-jepa/multiblock_epoch100.pth": "61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e",
    "third_party/checkpoints/ecg-cpc/ECG-CPC Checkpoint/last_11597276.ckpt": "bc253edc6ac279ce2caec86d3c3a21b740e43735ca03be1dc7c50e630b0ad4cf",
    "outputs/cohort25_recovery_20261028/challenge/full/rows.csv": "5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8",
    "outputs/mlc01_condition_dataset_20261110_v4/manifest.csv": "25d75f8f7e3449a635c0914cd5cdd8c2134352efa1f9b0e9d723c8d73be8f3a0",
    "outputs/mlc01_condition_dataset_20261110_v4/minimum_positive_rule.csv": "05cc3ce213cd316c71d2b3a6e6ff8fb68141f1d4865c0d00d274ab9652d261fd",
    "outputs/mlc01_condition_dataset_20261111_v5/manifest.csv": "6b3c61dc0d4039713ad2ea2a5098ba987d22055653f5c039aea533bffc95f8fe"
  },
  "dry_run": {
    "failed": [],
    "metric": {
      "stored_value": 0.9697697420282113,
      "from_stored_features": 0.9697697420282113,
      "difference_stored_features": 0.0,
      "with_reextracted_challenge_rows": 0.9697697420282113,
      "difference_reextracted": 0.0,
      "source_metric_minus_selection_metric": 0.0,
      "passed": true
    },
    "validation_primary_rule": {
      "model": "P0-v5",
      "value": 0.9610687901468651,
      "ci": [
        0.9556842122563732,
        0.9663166848638087
      ],
      "validation_value": 0.9697697420282113,
      "bound": 0.9597697420282113
    }
  },
  "test_groups": {
    "split_records": 13172
  }
}
```
