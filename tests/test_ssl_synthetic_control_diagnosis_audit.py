"""Artificial independent rank, shift, activity and diagnostic-integrity checks."""

import numpy as np
import pytest

from ecg_experiment.ssl_synthetic_control_diagnosis_audit import (
    activity_mask,
    record_values,
    seed_values,
    shift_for_index,
    summarize_rows,
    top_mask,
    top_selection,
)


def fixture_arrays():
    observed = np.zeros((4, 12, 5000), dtype=np.float32)
    observed[:, :, 1000:] = 1
    probability = np.zeros_like(observed)
    probability[:, :, 991:1011] = np.float32(0.9)
    support = np.zeros_like(observed, dtype=bool)
    support[2:, 0, 991:1011] = True
    prior = np.zeros((12, 5000), dtype=np.float64)
    return probability, observed, support, prior


def test_global_rank_ties_obey_flat_indices_and_record_full_tie_population():
    scores = np.zeros((12, 5000), dtype=np.float32)
    scores.ravel()[:700] = 1
    scores.ravel()[10000:10100] = 2
    selected, cutoff = top_selection(scores)
    expected = np.r_[np.arange(500), np.arange(10000, 10100)]
    assert np.array_equal(np.flatnonzero(selected), expected)
    assert cutoff == {
        "score": 1,
        "strictly_above_count": 100,
        "equal_cutoff_count": 700,
        "selected_tied_count": 500,
    }
    assert np.array_equal(selected, top_mask(np.asfortranarray(scores)))
    scores[0, 0] = np.nan
    with pytest.raises(ValueError, match="geometry/finite"):
        top_mask(scores)


def test_uniform_rank_keeps_exact_area_and_discloses_lead_preference():
    selected, cutoff = top_selection(np.full((12, 5000), 0.5, dtype=np.float32))
    assert selected.sum() == 600
    assert np.array_equal(np.flatnonzero(selected), np.arange(600))
    assert selected[0].sum() == 600
    assert cutoff["equal_cutoff_count"] == 60000
    assert cutoff["selected_tied_count"] == 600


def test_activity_prepend_strict_quantile_and_even_window_alignment():
    assert not activity_mask(np.ones((12, 5000), dtype=np.float32)).any()
    values = np.zeros((12, 5000), dtype=np.float32)
    values[:, 1000:] = 1
    mask = activity_mask(values)
    assert np.array_equal(np.flatnonzero(mask[0]), np.arange(991, 1011))
    assert np.array_equal(mask, np.repeat(mask[:1], 12, axis=0))


def test_common_shift_and_positive_denominators_and_persistence():
    probability, observed, support, prior = fixture_arrays()
    row = record_values(probability, observed, support, prior, 80, "supervised")
    assert row["control_persistence"] == 1
    assert row["variants"]["sham"]["activity_precision"] == 240 / 600
    assert sum(row["variants"]["sham"]["lead_counts"]) == 600
    assert sum(row["variants"]["sham"]["lead_fractions"]) == pytest.approx(1)
    assert row["positives"]["focal"]["target_recall"] == 1
    assert row["positives"]["focal"]["target_precision"] == 20 / 600
    assert row["positives"]["focal"]["precision_response"] == 0
    assert row["positives"]["focal"]["nuisance_hotspot_persistence"] == 1
    combined = seed_values({"supervised": probability, "permuted": probability}, observed, support, prior, 80)
    assert 500 <= combined["common_shift"] <= 4500
    assert combined["common_shift"] == shift_for_index(80)
    assert combined["arms"]["supervised"] == combined["arms"]["permuted"]
    support[2] = False
    with pytest.raises(ValueError, match="nonempty positive support"):
        record_values(probability, observed, support, prior, 80, "supervised")


def test_primary_only_bootstrap_and_negative_arm_difference_is_retained():
    probability, observed, support, prior = fixture_arrays()
    row = seed_values({"supervised": probability, "permuted": probability}, observed, support, prior, 80)
    import copy

    rows = [copy.deepcopy(row) for _ in range(48)]
    for value in rows:
        value["arms"]["supervised"]["primary_activity_enrichment"] = -0.2
        value["arms"]["permuted"]["primary_activity_enrichment"] = 0.1
    summary = summarize_rows(rows)
    assert set(summary) == {"primary", "secondary_means"}
    assert set(summary["primary"]) == {"supervised", "permuted", "supervised_minus_permuted"}
    assert summary["primary"]["supervised_minus_permuted"]["mean"] == pytest.approx(-0.3)
    assert summary["primary"]["supervised_minus_permuted"]["ci_high"] < 0
    assert summary["primary"]["supervised"]["seed"] == 2026103701
    assert "cutoff" not in summary["secondary_means"]["supervised"]["variants"]["sham"]


def test_csv_full_field_order_and_late_corruption(tmp_path):
    import copy
    import csv

    from scripts.validation.audit_ssl_synthetic_control_diagnosis import check_csv, flatten_independent

    probability, observed, support, prior = fixture_arrays()
    row = seed_values({"supervised": probability, "permuted": probability}, observed, support, prior, 80)
    rows = [copy.deepcopy(row) for _ in range(48)]
    flattened = [
        flatten_independent(value, arm, mode)
        for value in rows
        for arm in ("supervised", "permuted")
        for mode in ("sham", "nuisance", "focal", "repeated")
    ]
    fields = list(flattened[0])
    assert len(fields) == 59
    assert fields[:5] == ["index", "seed", "common_shift", "arm", "variant"]
    assert fields[-2:] == ["control_persistence", "primary_activity_enrichment"]
    path = tmp_path / "cases.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    main = {"csv_rows": 384, "csv_columns": fields}
    assert check_csv(path, rows, main) == 384 * 59
    flattened[-1]["lead_counts_0"] = -1
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    with pytest.raises(ValueError, match="CSV integer identity"):
        check_csv(path, rows, main)


def test_numeric_access_never_converts_training_rows(monkeypatch, tmp_path):
    from scripts.validation import audit_ssl_synthetic_control_diagnosis as audit

    class EvaluationOnly:
        def __init__(self, dtype):
            self.dtype = np.dtype(dtype)
            self.shape = (128, 4, 12, 5000)
            self.requests = []

        def __getitem__(self, selected):
            assert selected == slice(80, 128)
            self.requests.append(selected)
            return np.broadcast_to(np.zeros(1, dtype=self.dtype), (48, 4, 12, 5000))

        def __array__(self):
            raise AssertionError("TRAIN rows numerically accessed")

    observed, support = EvaluationOnly(np.float32), EvaluationOnly(bool)

    def fake_load(path, **kwargs):
        if path.name == "observed.npy":
            return observed
        if path.name == "support.npy":
            return support
        if path.name == "spatial_prior.npy":
            return np.zeros((12, 5000), dtype=np.float64)
        return np.broadcast_to(np.full(1, 0.25, dtype=np.float32), (48, 4, 12, 5000))

    monkeypatch.setattr(audit.np, "load", fake_load)
    loaded = audit.arrays_from_parent(tmp_path)
    assert loaded["observed"].shape == (48, 4, 12, 5000)
    assert observed.requests == [slice(80, 128)]
    assert support.requests == [slice(80, 128)]


def test_input_allowlist_and_separate_source_output_roots(tmp_path):
    from ecg_experiment.synthetic_audit_io import file_sha256
    from scripts.validation.audit_ssl_synthetic_control_diagnosis import INPUT_NAMES, bindings

    source, study, parent = (tmp_path / value for value in ("source", "study", "parent"))
    (source / "ecg_experiment").mkdir(parents=True)
    (study / "source_snapshot/ecg_experiment").mkdir(parents=True)
    parent.mkdir()
    (source / "ecg_experiment/code.py").write_text("source")
    (study / "source_snapshot/ecg_experiment/code.py").write_text("source")
    (study / "new.json").write_text("output")
    catalog = {}
    for name in INPUT_NAMES:
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        catalog[name] = file_sha256(path)
    main = {
        "source_sha256": {"ecg_experiment/code.py": file_sha256(source / "ecg_experiment/code.py")},
        "input_sha256": catalog,
        "output_sha256": {"new.json": file_sha256(study / "new.json")},
    }
    assert bindings(source, study, parent, main, {"predecessor_sha256": catalog}, lambda: None) == 14
    main["input_sha256"] = {**catalog, "forbidden_model.pt": "declared"}
    with pytest.raises(ValueError, match="eleven-file predecessor allowlist"):
        bindings(source, study, parent, main, {"predecessor_sha256": catalog}, lambda: None)


def test_main_terminal_cost_flags_bytes_and_no_double_count(monkeypatch, tmp_path):
    import json

    from scripts.validation import audit_ssl_synthetic_control_diagnosis as audit

    study, parent = tmp_path / "diagnostic", tmp_path / "previous"
    study.mkdir()
    previous = parent / "independent_audit_v1/prospective_ledger.json"
    previous.parent.mkdir(parents=True)
    prior_cpu, prior_gpu = 32.489216225, 38.433311437256634
    previous.write_text(
        json.dumps(
            {
                "current_machine_cumulative_cpu_seconds": prior_cpu,
                "current_machine_cumulative_gpu_wall_seconds": prior_gpu,
            }
        )
    )
    original_hash = audit.file_sha256

    def fake_prior_hash(path, guard=lambda: None):
        return audit.PRIOR_LEDGER_SHA if path == previous else original_hash(path, guard)

    monkeypatch.setattr(audit, "file_sha256", fake_prior_hash)
    (study / "cases.jsonl").write_text("fixture")
    main = {"cpu_seconds": 1.0, "wall_seconds": 2.0, "output_bytes_before_receipt": 7}
    (study / "summary.json").write_text(json.dumps(main))
    ledger = {
        "receipt": "summary.json",
        "receipt_sha256": original_hash(study / "summary.json"),
        "cpu_seconds": 1.1,
        "wall_seconds": 2.1,
        "gpu_wall_seconds": 0,
        "cpu_budget_passed": True,
        "wall_budget_passed": True,
        "prior_accounting_verified": True,
        "prior_ledger_sha256": audit.PRIOR_LEDGER_SHA,
        "prior_current_machine_cpu_seconds": prior_cpu,
        "prior_current_machine_gpu_wall_seconds": prior_gpu,
        "current_machine_cumulative_cpu_seconds": prior_cpu + 1.1,
        "current_machine_cumulative_gpu_wall_seconds": prior_gpu,
        "reserved_independent_audit": {"cpu_seconds": 180, "wall_seconds": 240, "bytes": 32 * 1024**2},
        "output_bytes_before_terminal_ledger": 7 + (study / "summary.json").stat().st_size,
    }
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    result = audit.verify_resources(study, parent, main, lambda: None)
    assert result["current_machine_cpu_through_main"] == prior_cpu + 1.1
    ledger["cpu_budget_passed"] = False
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="terminal main budget flags"):
        audit.verify_resources(study, parent, main, lambda: None)
    ledger["cpu_budget_passed"] = True
    ledger["current_machine_cumulative_cpu_seconds"] += 3.505593613
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="main CPU added once"):
        audit.verify_resources(study, parent, main, lambda: None)


def test_terminal_failure_audit_ledger_adds_own_cost_once(tmp_path):
    import json
    import time

    from scripts.validation.audit_ssl_synthetic_control_diagnosis import terminal_ledger

    (tmp_path / "failure.json").write_text('{"all_passed":false}')
    costs = {
        "main_cpu_seconds": 1.1,
        "current_machine_cpu_through_main": 33.589216225,
        "current_machine_gpu_wall_seconds": 38.433311437256634,
    }
    terminal_ledger(tmp_path, "failure.json", (time.process_time() - 0.1, time.perf_counter() - 0.2), costs)
    ledger = json.loads((tmp_path / "prospective_ledger.json").read_text())
    assert ledger["cpu_seconds"] >= 0.1
    assert ledger["current_machine_cumulative_cpu_seconds"] == 33.589216225 + ledger["cpu_seconds"]
    assert ledger["gpu_wall_seconds"] == 0
    assert ledger["audit_budget_passed"]
    assert ledger["audit_output_budget_passed"]


def test_integrated_main_independent_artificial_metrics():
    import copy

    main = pytest.importorskip("ecg_experiment.ssl_synthetic_control_diagnosis")
    from ecg_experiment.synthetic_audit_io import compare_values

    probability, observed, support, prior = fixture_arrays()
    selection, cutoff = top_selection(prior)
    actual_selection, actual_cutoff = main.ranked_mask(prior)
    assert np.array_equal(selection, actual_selection)
    compare_values(cutoff, actual_cutoff, "prior cutoff")
    for index in (80, 127):
        expected = seed_values(
            {"supervised": probability, "permuted": probability}, observed, support, prior, index
        )
        actual = main.diagnose_case(
            {"supervised": probability, "permuted": probability}, observed, support, actual_selection, index
        )
        compare_values(expected, actual, "artificial diagnostic row")
    rows = [copy.deepcopy(expected) for _ in range(48)]
    for index, row in enumerate(rows):
        row["arms"]["supervised"]["primary_activity_enrichment"] = index / 48
        row["arms"]["permuted"]["primary_activity_enrichment"] = float(np.sin(index))
    compare_values(summarize_rows(rows), main.summarize(rows), "three shared-draw primary intervals")
