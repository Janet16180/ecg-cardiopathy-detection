# Code quality review, 29 September 2026

This review checks `ecg_experiment/`, `scripts/` and `tests/` against the project rules: the
user's global instructions, `CONTRIBUTING.md`, `AGENTS.md` and the Ruff configuration in
`pyproject.toml`. Every count below comes from a grep, an AST scan or Ruff. None is an estimate.

The code is in good shape for its size: 370 Python files and about 84,500 lines. Ruff passes with a
strict rule set (docstrings, annotations, import order, complexity 10). The test suite has 798 tests.
There are no bare `except:` clauses, no commented-out code (Ruff `ERA`: 0) and no hard-coded home
directories. The main problems are structural. Shared logic sits inside frozen experiment runners,
the same statistics helpers have been copied many times, and the lint configuration lives in a file
that experiment receipts pin by hash.

## Which files are frozen

`AGENTS.md` forbids editing frozen sources. This review builds the list first. It scanned 1,331 text
receipts (`.json`, `.jsonl`, `.txt`, `.md`, `.yaml`, `.csv`, `.log`) under `outputs/` and
`data/processed/` of the main checkout, including `outputs/refactor_pause/source_hashes.json`.
A tracked file counts as frozen in any of these cases:

- its repository-relative path appears in a receipt (201 files);
- the SHA-256 of its current bytes appears in a receipt (280 files);
- its dotted module name appears in a recorded command (18 files).

Together these give **332 frozen tracked files**. They include 223 Python files (102 in
`ecg_experiment/`, 68 in `scripts/`, 50 in `tests/`, 3 in `outputs/`), and also `pyproject.toml`,
`uv.lock`, `configs/datasets.json` and the DVC pointers.

A further 49 Python files are **historically referenced**. Their bare file name appears in
pre-refactor receipts such as `outputs/experiment009_cpc_prediction_mismatch/provenance/sources.json`
and `outputs/repository_refactor/sources.json`. Their current bytes no longer match those receipts.
This review treats them as frozen too. Only 103 Python files are unpinned, and 54 of those are tests.

Some unpinned files were still left alone:

- Files imported by the MIMIC downloader or by frozen runners: `ecg_experiment/mimic.py`,
  `processes.py`, `pool_overlap.py`, `resample.py`, `mimic_cpc_clustering.py` (docstrings only) and
  `scripts/coordination/common.py`.
- The prepared but not yet run Experiment 021 code: `ecg_experiment/code15_cpc.py`,
  `scripts/data/build_code15_cpc_cache.py` and `scripts/experiments/run_code15_supervised021.py`.
  Its protocol is frozen and its source hashes will be recorded at launch.

## Findings

"Frozen" means hash- or path-pinned. "Historical" means referenced by name in a pre-refactor receipt.

### High

**H1. Reusable logic lives in experiment runners, and runners import each other.**
`CONTRIBUTING.md` says scripts must never import from other scripts. There are 66 such imports under
`scripts/`, and 8 more imports go the wrong way, from `ecg_experiment/` into `scripts/`. Examples:

- `scripts/experiments/run_label_efficiency025b.py:31-47` and
  `scripts/experiments/run_multisource_geometry024b.py:55-61` import feature loaders and split
  helpers (`open_caches`, `sph_features`, `split_table`, `feature_identity`) from seven earlier
  runners (frozen).
- `scripts/experiments/run_calibrated_threshold027.py:39` imports from `run_sph_external022` (frozen).
- `scripts/experiments/run_xecg_adaptation.py:40,157,186,903` imports from three other scripts
  (historical).
- `ecg_experiment/xecg_readout_v7.py:13`, `xecg_encoder_motion_v10_analysis.py:15`,
  `xecg_encoder_motion_v11_verification.py:21-24`, `xecg_encoder_motion_replication_v10.py:30` and
  `_v14.py:29` import runner scripts (all frozen).

Every successor experiment that needs these helpers must either import a frozen runner or copy it.
The fix is forward-only; see recommendation 1.

**H2. The same statistics have been reimplemented many times, and they diverge.**

- There are about 20 patient-bootstrap or AUROC-interval functions. Examples:
  `evaluation.py:175`, `full_development.py:184`, `xecg_rescue_analysis.py:40`,
  `xecg_readout_v7.py:88`, `xecg_head_mechanism_v8.py:295`, `morphology_clean_audit.py:31`,
  `normal_manifold.py:128,165`, `screening_threshold.py:234`, `multisource_calibration.py:161`,
  `multisource_geometry.py:86,241`, `label_efficiency_multisource.py:131`,
  `clean_probe_fusion.py:95`, `cpc_local_readout.py:188`, `xecg_encoder_motion_v10_analysis.py:122`,
  `scripts/experiments/run_xecg_encoder_motion016_v9.py:628`,
  `scripts/experiments/run_jepa_cpc_fusion.py:368` and `scripts/reports/compare_adaptation.py:94`.
  All of them are frozen or historical.
- Two public functions share the name `patient_bootstrap` but mean different things.
  `evaluation.patient_bootstrap(y, prob, patient_ids, threshold, repeats=500)` returns metric
  intervals at a threshold. `full_development.patient_bootstrap(patients, y, first, second,
  draws=2000)` returns a paired AUROC difference. Eight runners import the second one.
- The draw counts (500 in 4 places, 2,000 in 17), the handling of single-class draws (skipped, counted or failed)
  and the seeds differ between copies. Results from different experiments are therefore not always
  interval-comparable, even when the protocols say "patient bootstrap".

**H3. The lint configuration lives in a hash-pinned file.** `pyproject.toml` holds `[tool.ruff]`.
Its current SHA-256 is recorded in `outputs/experiment022b_multisource_readout_v1/result.json`,
`experiment024b_multisource_geometry_v1/result.json` and `experiment025b_label_efficiency_multisource_v1/result.json`.
Any lint rule change therefore changes a file those receipts pin. This review did not change the
Ruff rules for that reason. See recommendation 5.

### Medium

**M1. Patient grouping is quadratic.** Six helpers build the rows of each patient with one
full-array comparison per patient, which costs O(records × patients):

- `evaluation.py:55`, `full_development.py:209`, `normal_manifold.py:153`, `cpc_local_readout.py:191`
  and `cpc_scaling_readout.py:171` (frozen);
- `scripts/reports/report_xecg_probe016.py:79` (frozen).

This is fine at 16k records. At the 100k to 1M cohorts now being built, it will dominate bootstrap
time. A `np.unique(..., return_inverse=True)` plus `argsort` split gives the same groups in
O(n log n).

**M2. Broad `except` clauses.** There are 17 broad handlers. All of them clean up or record a
status and then re-raise, so none of them silences an error:

- `ecg_experiment/staging.py:64`, `tracking.py:395`, `cpc_local_readout.py:363`,
  `xecg_encoder_motion_v11.py:128`, `xecg_encoder_motion_replication_v10.py:170` and `_v14.py:151`
  (frozen);
- `scripts/prepare_mimic_ssl.py:364` and `scripts/experiments/run_xecg_encoder_motion016_v9.py:404`
  (frozen);
- `scripts/coordination/run_cpc_coordinated.py:121`, `run_xecg_coordinated.py:121`,
  `run_xecg_adaptation_coordinated.py:106` and `run_priority_queue.py:299` (historical);
- `scripts/experiments/run_jepa_cpc_fusion.py:872`, `run_released_cpc.py:123` and
  `run_mimic_scale.py:446,532,794` (historical).

Ruff's `BLE` rule reports 0 violations for the same reason. The narrow handlers that return a
default are documented parsing fallbacks, not hidden failures, for example
`scripts/download_ptbxl_waveforms.py:117` and `scripts/validation/audit_dataset_quality.py:69`.

**M3. Public functions without NumPy sections.** Ruff's pydocstyle accepts one-line docstrings. The
user's rule asks for Parameters and Returns sections. Of about 1,880 public functions:

- 395 in frozen files lack those sections;
- 18 in historical files lack them;
- 36 in unpinned files lacked them. This review fixed 17. The other 19 are the Experiment 021 code
  in `scripts/experiments/run_code15_supervised021.py` and `scripts/data/build_code15_cpc_cache.py`,
  left alone (see above).

**M4. Deep nesting.** Forty-six functions have control-flow depth 4 or more. The deepest are all
frozen:

| File | Depth |
| --- | ---: |
| `scripts/experiments/run_local_adaptation029.py:226` (`part_a_encoder`) | 7 |
| `scripts/experiments/run_xecg_droppath_rescue016.py:349` (`_same_optimizer`) | 6 |
| `ecg_experiment/local_adaptation.py:157` | 5 |
| `ecg_experiment/morphology_clean_audit.py:139` | 5 |
| `ecg_experiment/xecg_encoder_motion_replication_v10.py:43` and `_v14.py:42` | 5 |
| `scripts/experiments/run_label_efficiency025b.py:439` | 5 |
| `scripts/experiments/run_xecg_encoder_motion016_v9.py:288` | 5 |
| `scripts/prepare_mimic_ssl.py:317` | 5 |

The unpinned ones are `ecg_experiment/mimic.py:167,272` (downloader dependency, left alone),
`scripts/experiments/probe_delta_memory011.py:25` and
`scripts/validation/verify_sampled_cpc_cache.py:23`. The last two are fixed.

**M5. Slow tests.** The suite took 143 s with 4 threads before this change:

- `tests/test_experiment017_clean.py::test_patient_bootstrap_is_paired_and_reproducible`: 38 s (frozen);
- `tests/test_run_embedding_geometry024.py`: 31 s + 10 s. This is fixed; the two tests now take 2 s;
- `tests/test_adapt_ecgfm_budget.py::test_stops_within_epoch_and_counts_examples`: 11 s (frozen).

The frozen 38 s test is now about a third of the suite.

**M6. Versioned copies.** There are 28 files with `_vN` suffixes, 24 of them for the xECG
Experiment 016 series (for example `run_xecg_encoder_motion_replication016_v10.py` to `_v14.py`).
This is how the frozen-source policy is meant to work. The cost is that fixes cannot spread, which
makes H1 and H2 more expensive over time.

### Low

- **L1. Hard-coded `/tmp` paths.** `ecg_experiment/gpu.py:10` (`/tmp/ecg_project_gpu.lock`) is a
  deliberate machine-wide lock and should stay (frozen).
  `scripts/experiments/run_released_cpc.py:35` and `scripts/features/extract_ecg_cpc.py:456-457` set
  Matplotlib and KeOps cache folders in `/tmp` (historical). There are no `/home/...` paths.
- **L2. Local `ROOT` definitions.** 111 modules define `ROOT = Path(__file__)...` instead of
  importing `ecg_experiment.ROOT`.
- **L3. Duplicated lead constants.**
  - Canonical lead names are defined in `waveforms.py:15` (tuple), in `eda/signals.py:13` (list),
    and in upper case in `scripts/validation/audit_dataset_quality.py:20`.
  - The 8-lead JEPA subset (0, 1, 6-11) is repeated in `external_encoders.py:32`,
    `lead_innovation.py:14` and `scripts/features/extract_jepa.py:31`.
  - `LEADS = 12` or `LEAD_COUNT = 12` appears in 6 modules.
- **L4. One-line `check(condition, message)` helper, five copies.** `ecg_experiment/clean_rerun.py:39`,
  `scripts/validation/audit_cpc_25k.py:18`, `audit_cpc_25k_readout.py:18`,
  `audit_cpc_scaling_readout.py:17` and `smoke_training_dataset.py:18`.
- **L5. Unused `noqa` directives.** Ruff `RUF100` finds 56 in frozen or historical files, for
  example 6 in `scripts/reports/report_experiment.py` and 1 in `ecg_experiment/xecg.py`. The two in
  unpinned files are removed.
- **L6. Library modules that print.** 13 modules in `ecg_experiment/` print progress to stdout.
  This is acceptable for batch jobs, but a library should normally leave output to its callers.
- **L7. Documented but unimported EDA code.** Nothing imports `ecg_experiment/eda/processed.py`,
  `eda/ptbxl_labels.py` or `eda/mimic.py:212` (`project_accepted`). They are not dead: they are the
  reproduction code that `docs/eda-pipeline-review.md` cites. No test covers them, because they need
  the raw data.
- **L8. Formatting is not enforced.** `ruff format --check` would reformat 280 of 371 files. Do not
  reformat frozen files. If formatting is wanted, apply it to new files only.
- **L9. Tests are exempt from annotation and docstring rules** (`tests/** = ["S101", "D", "ANN"]`).
  This is a reasonable choice, but it departs from the "type hints everywhere" rule.

These checks found nothing:

- bare `except:` clauses: 0;
- commented-out code: 0;
- home-directory paths: 0;
- empty tests: 0. `tests/test_imports.py::test_script_module_imports` is the only test without an
  assert, and it is a working import smoke test over 136 script modules.

## What was fixed

All fixes are in unpinned files, keep behaviour the same, and are one kind of change per commit.

1. `tests/test_run_embedding_geometry024.py`: two structure-only tests now use 100 bootstrap draws
   through a `monkeypatch` fixture. They take 2 s instead of 42 s.
2. Removed unused `noqa` directives in `ecg_experiment/code15_label_groups.py:26` (S324 is not
   enabled, so it became a plain comment) and `scripts/reports/plot_local_adaptation029.py:10`.
   Removed unused `tmp_path` fixtures in `tests/test_cpc_pool.py:50` and `tests/test_pilot.py:189`.
3. `scripts/experiments/probe_delta_memory011.py`: moved the per-arm timing into `measure_arm`.
   This removes a depth-4 block and a reassigned loop variable (`PLW2901`).
   `scripts/validation/verify_sampled_cpc_cache.py`: `replay_rows` now uses guard clauses. Both files
   have full NumPy docstrings and `dict[str, Any]` return types. The probe was smoke-run on CPU. The
   old and new `replay_rows` were compared on synthetic data and gave identical results and errors.
4. Added NumPy Parameters, Returns and Raises sections to 13 public functions in
   `ecg_experiment/code15_label_groups.py`, `mimic_cluster_qc.py`, `mimic_cpc_clustering.py`,
   `mimic_cpc_flags.py` and the four `scripts/validation/audit_cpc_*` scripts.
   `mimic_cpc_flags.score` now types its batch iterator instead of using `object`.

Nothing was deleted. Every candidate for dead code turned out to be cited reproduction code (L7) or a
command entry point.

## What was left, and why

- **All findings in frozen or historical files:** H1, H2, M1, M2, most of M3 and M4, the 38 s test,
  L1, L4 and L5. Editing them would change pinned bytes.
- **Ruff rule changes (H3):** they would edit the hash-pinned `pyproject.toml`.
- **Experiment 021 code:** its protocol is frozen and it has not run yet.
- **`ecg_experiment/mimic.py`, `processes.py`, `resample.py` and `pool_overlap.py`:** unpinned, but
  the downloader or frozen runners import them.
- **Consolidating duplicated helpers:** proposed below instead of done, as asked.

## Recommendations for new code

1. **Put shared readout logic in the library.** When a successor experiment next needs
   `open_caches`, `sph_features`, `split_table`, `feature_identity`, `fit_logistic` or `predict`,
   copy it once into a new unpinned module such as `ecg_experiment/readout.py`. Import it from there
   and give it a test. Frozen runners keep their own copies. New runners import only from
   `ecg_experiment/`.
2. **Use one interval module.** Create `ecg_experiment/intervals.py` for all new experiments. It
   would hold:
   - `patient_groups(patient_ids) -> list[np.ndarray]`, using the O(n log n) `np.unique` and
     `argsort` split;
   - `patient_resample(groups, rng) -> np.ndarray`;
   - `paired_auroc_difference(patients, y, first, second, draws, seed)`, which returns the
     difference, the interval and the number of single-class draws;
   - `metric_intervals(y, prob, patients, threshold, draws, seed)`.

   Fix one policy for single-class draws (skip and count) and one default draw count (2,000). Check
   bit-for-bit that `full_development.patient_bootstrap` gives the same output for the same seed, so
   that new and old results remain comparable. Do not change the frozen copies.

   Done: `ecg_experiment/intervals.py`, tested in `tests/test_intervals.py` against
   `full_development`, `evaluation`, `normal_manifold` and `screening_threshold`.
3. **Choose one resampler.** Use `ecg_experiment/resample.py` (`resample_full`) for new code, and
   `cpc_input_audit.historical_resample` only when a CPC-compatible per-half input is required.
   Today 500 to 250 Hz resampling is written separately in `echonext_readout.py:130`,
   `eda/processed.py:146-148,201-203`, `scripts/data/prepare_cpc_data.py:232` and `code15_cpc.py:80`
   (400 to 250 Hz).
4. **Keep constants in one place.** Import lead names from `ecg_experiment.waveforms.LEADS` and the
   JEPA subset from `external_encoders.JEPA_LEADS`. Import `ROOT` from `ecg_experiment`. For
   invariant checks, add a single `require(condition, message)` to an unpinned module rather than a
   sixth copy.
5. **Separate lint from the pinned project file.** At the next deliberate `pyproject.toml` change
   (a dependency update, which needs a new run identity anyway), move `[tool.ruff]` into a
   `ruff.toml`. Lint rules can then change without touching a pinned file. At that point, enable
   the rule families that already have 0 violations:
   - `BLE` (blind except), `ERA` (commented-out code), `PGH` (blanket `noqa`), `A` (shadowed
     builtins), `DTZ` (naive datetimes), `T10` (debugger calls), `PLE`, `RSE`, `TID`, `LOG` and `FLY`.

   `RUF100` can follow with per-file ignores for the 56 frozen occurrences.
6. **Keep tests fast.** Tests that only check the structure of a result should patch the bootstrap
   draw constant, as done here. Adding `--durations=10` to the CI pytest call would show slow tests
   as they appear. If the frozen 38 s test becomes a problem, a `conftest.py` marker could select
   it out of the pre-push hook without editing it.
7. **Write the full NumPy docstring for new public functions from the start.** One-line docstrings
   pass Ruff but not the project rule, so reviewers need to check for Parameters and Returns
   sections.

## Checks

| Check | Before | After |
| --- | --- | --- |
| `ruff check ecg_experiment scripts tests` | All checks passed | All checks passed |
| `pytest -q` (4 threads, CPU only) | 794 passed, 4 skipped, 143 s | 804 passed, 4 skipped, 82 s |

The "after" run includes the 10 Experiment 030 tests merged from `main` in the meantime. The
frozen 38 s test still takes 36 s. The 4 skipped tests need local model checkpoints, which are not in this worktree. CI skips them for
the same reason.
