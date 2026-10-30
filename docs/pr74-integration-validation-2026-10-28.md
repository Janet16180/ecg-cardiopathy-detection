# Corrective PR integration validation

The integrated reconstruction, evaluation and application passed the full local Python suite:
1,472 passed and 4 skipped in 206.94 seconds. The later UI-only addition comparing the two
25,000-record cohorts passed all 27 application tests again. The deliberate flat-signal fixture
produced spectral all-NaN warnings and correctly abstained before model inference.

Executed checks:

- `python -m pytest -q`: full integrated suite, with CUDA disabled and numerical thread counts set to one.
- `python -m pytest tests/test_corrected_ui.py -q`: final application regression check.
- `ruff check ecg_experiment scripts tests conftest.py ui`: passed.
- `node --test tests/test_ui_matrix.mjs`: passed.
- `uv build --offline --no-sources`: source distribution and wheel built successfully.
- `git diff --check origin/main`: no whitespace errors in changed source and documentation.

An independent real-browser/API audit at integration commit `98c93c9` verified the from-raw
artifact mode, both evaluation sites, exact frozen-threshold reset, English/Spanish labels,
mobile layout and the two-row cohort comparison. Moving the exploratory slider leaves that
fixed comparison unchanged. Missing or modified fresh artifacts were rejected without falling
back to cached artifacts. Flat synthetic input abstained; unsupported units and legacy raw-data
routes were rejected. The audit generated no clinical sample prediction and stopped its
temporary server and browser sessions afterward.

An independent data audit checked all ten published fresh-reproduction receipt hashes, all five
byte-identical cohort manifests, admitted training roles, known held-out overlap checks and exact
raw-role identity hashes. All 30 complete saved evaluation payloads and the full primary bootstrap
result match between the original and fresh studies. The separate
[matrix audit receipt](../reports/pr74-corrective-evaluation-2026-10-28/independent-audit.json)
records recalculation of the fixed points and all 6,090 saved threshold matrices.

The final audit found that the original reproduction comparison gate did not reject every possible
change to thresholds, sweeps or interval bounds. Its frozen implementation remains preserved.
An additive [strict verifier](../ecg_experiment/pr74_strict_parity_verifier.py) now compares every
field of all 30 evaluation payloads and the complete primary result. All nine regression tests
passed, including deliberate threshold, sweep, precision and interval mutations; Ruff passed.
The executed [strict parity receipt](../reports/pr74-corrective-evaluation-2026-10-28/strict-parity.json)
confirms exact agreement for both saved warm and fresh aggregates. It read no prediction arrays
or waveforms and performed no model fits or scores. The receipt CLI writes exclusively to preserve
evidence; repeat read-only checks can call `verify_saved(input_root)` from the verifier module.

These software and reproduction checks do not establish clinical performance in Italian students.
The [scientific results](pr74-corrective-evaluation-results-2026-10-28.md) and
[reproduction accounting](pr74-from-raw-reproduction-results-2026-10-28.md) retain the reused-data,
proxy-label, quality-abstention and explicit cache-reuse limitations. No scientific pipeline is
promoted after the failed primary comparison.

Remote CI remains paused under the existing user instruction. Local passing checks are not
represented as successful GitHub checks; the PR is prepared as a draft, without merging.
