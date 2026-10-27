"""Artificial checks separating oracle lead timing from support-bearing lead ranking."""

import copy

import numpy as np
import pytest

from ecg_experiment.ssl_synthetic_temporal_saved_audit import (
    case_endpoints,
    crossed_intervals,
    diagnostic_summary,
)


def test_equal_active_lead_ap_does_not_area_weight():
    support = np.zeros((12, 5000), dtype=bool)
    support[0, 0] = True
    support[1, :500] = True
    probability = np.zeros((12, 5000), dtype=np.float32)
    probability[0, 0] = 1
    row = case_endpoints(probability, support)
    assert row["active_leads"] == [0, 1]
    assert row["active_lead_AP"] == pytest.approx([1, 0.1])
    assert row["active_lead_temporal_AP"] == pytest.approx(0.55)
    assert row["active_lead_temporal_AP"] != pytest.approx((1 + 500 * 0.1) / 501)


def test_oracle_temporal_rank_and_lead_selection_can_disagree():
    support = np.zeros((12, 5000), dtype=bool)
    support[0, 100] = True
    probability = np.full((12, 5000), 0.9, dtype=np.float32)
    probability[0] = 0
    probability[0, 100] = 0.1
    first = case_endpoints(probability, support)
    assert first["active_lead_temporal_AP"] == 1
    assert first["lead_presence_AP"] == pytest.approx(1 / 12)
    probability[0] += 0.85
    second = case_endpoints(probability, support)
    assert second["active_lead_temporal_AP"] == first["active_lead_temporal_AP"]
    assert second["lead_presence_AP"] == 1


def test_ties_and_degenerate_cases_are_preserved_or_rejected_without_exclusion():
    support = np.zeros((12, 5000), dtype=bool)
    support[5, :10] = True
    probability = np.full((12, 5000), 0.5, dtype=np.float32)
    row = case_endpoints(probability, support)
    assert row["active_leads"] == [5]
    assert row["active_lead_temporal_AP"] == pytest.approx(10 / 5000)
    assert row["lead_presence_AP"] == pytest.approx(1 / 12)
    with pytest.raises(ValueError, match="both lead-presence classes"):
        case_endpoints(probability, np.zeros_like(support))
    dense = support.copy()
    dense[5] = True
    with pytest.raises(ValueError, match="both within-lead temporal classes"):
        case_endpoints(probability, dense)
    dense[:, 10] = True
    with pytest.raises(ValueError, match="both lead-presence classes"):
        case_endpoints(probability, dense)
    probability[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite native probability"):
        case_endpoints(probability, support)


def artificial_rows():
    rows = []
    for fit in (1, 2, 3):
        for arm in ("B", "P", "K", "PK"):
            for index in range(2000, 2064):
                for mode in ("focal", "repeated"):
                    values = {"B": (0.5, 0.2), "P": (0.45, 0.3), "K": (0.4, 0.4), "PK": (0.3, 0.6)}[arm]
                    rows.append(
                        {
                            "fit_id": fit,
                            "arm": arm,
                            "index": index,
                            "seed": 2026103000 + index,
                            "mode": mode,
                            "active_leads": [0],
                            "active_lead_AP": [values[0]],
                            "active_lead_temporal_AP": values[0],
                            "lead_presence_AP": values[1],
                        }
                    )
    return rows


def test_disagreeing_descriptive_contrasts_retained_without_scientific_success_rule():
    rows = artificial_rows()
    summary = diagnostic_summary(rows)
    assert len(summary["descriptive_contrasts"]) == 6
    assert summary["descriptive_contrasts"]["PK_minus_B__active_lead_temporal_AP__pooled"]["ci_high"] < 0
    assert summary["descriptive_contrasts"]["PK_minus_B__lead_presence_AP__pooled"]["ci_low"] > 0
    assert "scientific_passed" not in summary
    assert set(summary["arm_fit_means"]) == {"1", "2", "3"}
    corrupted = copy.deepcopy(rows)
    corrupted[-1]["index"] = 2000
    with pytest.raises(ValueError, match="duplicate diagnostic identity"):
        diagnostic_summary(corrupted)


def test_crossed_one_shared_column_draw_and_fixed_pairing_for_all_six_endpoints():
    base = np.tile(np.arange(64, dtype=np.float64), (3, 1))
    vectors = {str(index): base + index for index in range(6)}
    summary = crossed_intervals(vectors)
    rng = np.random.default_rng(2026105601)
    draws = []
    for _ in range(2000):
        rng.integers(0, 3, 3)
        draws.append(float(rng.integers(0, 64, 64).mean()))
    low, high = np.percentile(draws, [2.5, 97.5])
    for index in range(6):
        assert summary[str(index)]["ci_low"] == pytest.approx(low + index, abs=1e-12)
        assert summary[str(index)]["ci_high"] == pytest.approx(high + index, abs=1e-12)
        assert summary[str(index)]["seed"] == 2026105601
        assert summary[str(index)]["skipped_draws"] == 0


def test_strict_allowlist_separate_roots_and_changed_payload(tmp_path):
    from ecg_experiment.synthetic_audit_io import file_sha256
    from scripts.validation.audit_ssl_synthetic_temporal_diagnosis import INPUT_NAMES, bindings

    source, study, parent = (tmp_path / value for value in ("source", "study", "parent"))
    for directory in (source, study / "source_snapshot"):
        directory.mkdir(parents=True)
        (directory / "source.py").write_text("reviewed")
    catalog = {}
    for name in INPUT_NAMES:
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        catalog[name] = file_sha256(path)
    (study / "cases.jsonl").write_text("all fixed cases")
    main = {
        "source_sha256": {"source.py": file_sha256(source / "source.py")},
        "input_sha256": catalog,
        "output_sha256": {"cases.jsonl": file_sha256(study / "cases.jsonl")},
    }
    config, admission = {"allowed_parent_files": list(INPUT_NAMES)}, {"parent": {"sha256": catalog}}
    assert len(INPUT_NAMES) == 29
    assert bindings(source, study, parent, main, config, admission, lambda: None) == 32
    changed = parent / "fit_3/PK/fresh_probabilities.npy"
    changed.write_text("corrupted last map")
    with pytest.raises(ValueError, match="allowed parent binding"):
        bindings(source, study, parent, main, config, admission, lambda: None)
    extra = copy.deepcopy(config)
    extra["allowed_parent_files"].append("fresh_observed.npy")
    with pytest.raises(ValueError, match="twenty-nine-file"):
        bindings(source, study, parent, main, extra, admission, lambda: None)


def test_arrays_open_no_signal_or_state_and_numeric_reads_positive_modes_only(tmp_path, monkeypatch):
    from ecg_experiment.ssl_synthetic_temporal_saved_audit import endpoint_rows
    from scripts.validation import audit_ssl_synthetic_temporal_diagnosis as audit

    class PositiveOnly:
        shape = (64, 4, 12, 5000)

        def __init__(self, dtype):
            self.dtype = np.dtype(dtype)
            self.requests = []

        def __array__(self, *args):
            raise AssertionError("whole map array converted")

        def __getitem__(self, key):
            assert isinstance(key, tuple)
            assert key[1] in (2, 3)
            self.requests.append(key)
            if self.dtype == np.dtype(bool):
                mask = np.zeros((12, 5000), dtype=bool)
                mask[0, :10] = True
                return mask
            return np.full((12, 5000), 0.5, dtype=np.float32)

    support, maps, opened = PositiveOnly(bool), [], []

    def load(path, **kwargs):
        opened.append(path.name)
        if path.name == "fresh_support.npy":
            return support
        assert path.name == "fresh_probabilities.npy"
        value = PositiveOnly(np.float32)
        maps.append(value)
        return value

    monkeypatch.setattr(audit.np, "load", load)
    monkeypatch.setattr(audit, "read_json", lambda path: {})
    saved_support, loaded, _ = audit.arrays_from_parent(tmp_path)
    rows = endpoint_rows(loaded[1]["B"], saved_support, 1, "B")
    assert len(rows) == 128
    assert len(maps) == 12
    assert set(opened) == {"fresh_support.npy", "fresh_probabilities.npy"}
    assert len(support.requests) == 128
    assert len(maps[0].requests) == 128


def test_late_active_lead_ap_corruption_detected_with_no_scientific_gate(tmp_path):
    import json

    from ecg_experiment.ssl_synthetic_temporal_saved_audit import endpoint_rows
    from scripts.validation.audit_ssl_synthetic_temporal_diagnosis import metric_evidence

    probability = np.broadcast_to(np.full(1, 0.5, dtype=np.float32), (64, 4, 12, 5000))
    one = np.zeros((4, 12, 5000), dtype=bool)
    one[2:, 0, :10] = True
    support = np.broadcast_to(one, (64, 4, 12, 5000))
    maps = {fit: dict.fromkeys(("B", "P", "K", "PK"), probability) for fit in (1, 2, 3)}
    rows = [
        row
        for fit in (1, 2, 3)
        for arm in ("B", "P", "K", "PK")
        for row in endpoint_rows(probability, support, fit, arm)
    ]
    main = {**diagnostic_summary(rows), "case_rows": 1536}
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert metric_evidence(tmp_path, main, support, maps, lambda: None)["case_rows"] == 1536
    rows[-1]["active_lead_AP"][0] += 0.01
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="arithmetic.*numeric"):
        metric_evidence(tmp_path, main, support, maps, lambda: None)


def test_failure_costs_preserved_without_double_charging_verified_prior(tmp_path, monkeypatch):
    import json
    import time

    from ecg_experiment.synthetic_audit_io import read_json
    from scripts.validation import audit_ssl_synthetic_temporal_diagnosis as audit

    study, parent = tmp_path / "new", tmp_path / "paired"
    study.mkdir()
    prior = parent / "independent_audit_v1/prospective_ledger.json"
    prior.parent.mkdir(parents=True)
    cpu, gpu = 390.154950634, 447.98139213304967
    prior.write_text(
        json.dumps(
            {
                "current_machine_cumulative_cpu_seconds": cpu,
                "current_machine_cumulative_gpu_wall_seconds": gpu,
            }
        )
    )
    original_hash = audit.file_sha256

    def identity(path, guard=lambda: None):
        return (
            "705cc964022f4f1d78d08d7d1eff75a665f02c150e311dacf2e1c4503ede5496"
            if path == prior
            else original_hash(path, guard)
        )

    monkeypatch.setattr(audit, "file_sha256", identity)
    main = {"cpu_seconds": 1}
    (study / "summary.json").write_text(json.dumps(main))
    ledger = {
        "receipt": "summary.json",
        "receipt_sha256": original_hash(study / "summary.json"),
        "prior_ledger_sha256": identity(prior),
        "cpu_seconds": 1.1,
        "wall_seconds": 2,
        "gpu_wall_seconds": 0.0,
        "prior_current_machine_cpu_seconds": cpu,
        "prior_current_machine_gpu_wall_seconds": gpu,
        "current_machine_cumulative_cpu_seconds": cpu + 1.1,
        "current_machine_cumulative_gpu_wall_seconds": gpu,
    }
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    admission = {
        "prior_ledger": {
            "sha256": identity(prior),
            "current_machine_cumulative_cpu_seconds": cpu,
            "current_machine_cumulative_gpu_wall_seconds": gpu,
        }
    }
    costs = audit.verified_accounting(study, parent, main, admission, lambda: None)
    destination = study / "independent_audit_v1"
    destination.mkdir()
    (destination / "failure.json").write_text(
        json.dumps({"all_passed": False, "error": "artificial late mismatch"})
    )
    audit.write_ledger(destination, "failure.json", (time.process_time(), time.perf_counter()), costs)
    own = read_json(destination / "prospective_ledger.json")
    assert own["current_machine_cumulative_cpu_seconds"] == cpu + 1.1 + own["cpu_seconds"]
    assert own["current_machine_cumulative_gpu_wall_seconds"] == gpu
    assert own["gpu_wall_seconds"] == 0
    ledger["current_machine_cumulative_cpu_seconds"] += cpu
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="once-only cumulative addition"):
        audit.verified_accounting(study, parent, main, admission, lambda: None)
