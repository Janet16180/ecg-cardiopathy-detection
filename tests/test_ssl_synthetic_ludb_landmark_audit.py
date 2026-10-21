import numpy as np
import pytest

from ecg_experiment.ssl_synthetic_ludb_landmark_audit import bootstrap, metrics, midranks


def test_empirical_midranks_have_exact_tie_mass_and_half_for_constants():
    np.testing.assert_array_equal(midranks(np.array([3.0, 1.0, 1.0, 7.0])), [0.625, 0.25, 0.25, 0.875])
    np.testing.assert_array_equal(midranks(np.ones(10)), np.full(10, 0.5))


def test_n_points_and_controls_are_within_span_not_full_record():
    score = np.zeros(5000)
    score[100:200] = np.arange(100)
    result, ranks = metrics(score, 100, 200, [150, 199])
    assert result["observed"] == 0.75
    assert result["shift_samples"] == [17, 31, 43, 59, 71, 83]
    assert result["controls"] == [
        float(np.roll(ranks, shift)[[50, 99]].mean()) for shift in result["shift_samples"]
    ]
    assert result["gain"] == result["observed"] - np.mean(result["controls"])


def test_score_at_missing_boundary_landmark_needs_no_interval_mask():
    result, _ = metrics(np.zeros(5000), 20, 4000, [100, 250, 3000])
    assert result["observed"] == 0.5
    assert result["controls"] == [0.5] * 6
    assert result["gain"] == 0
    assert result["constant_score"]


def test_reference_or_n_geometry_never_silently_drops_points():
    with pytest.raises(ValueError, match="inside"):
        metrics(np.zeros(5000), 10, 100, [5, 50])
    with pytest.raises(ValueError, match="nonempty"):
        metrics(np.zeros(5000), 10, 100, [])


def test_bootstrap_exact_whole_subject_draw_order_and_linear_bounds():
    rows = [{"record_id": n, "gain": n / 200} for n in range(1, 201) if n % 5]
    draws, bounds = bootstrap(rows)
    values = np.array([row["gain"] for row in rows])
    rng = np.random.default_rng(2026106101)
    assert draws[0] == values[rng.integers(0, 160, 160)].mean()
    assert bounds == np.percentile(draws, [2.5, 97.5], method="linear").tolist()
    with pytest.raises(ValueError, match="all160"):
        bootstrap(rows[:-1])


def test_incomplete_qrs_onset_is_retained_as_landmark_without_dense_target():
    import struct

    from ecg_experiment.ssl_synthetic_ludb_landmark_audit import annotation_row

    payload = (
        b"".join(
            struct.pack("<H", (code << 10) | delta)
            for code, delta in [(1, 100), (40, 10), (39, 20), (27, 10), (40, 10)]
        )
        + b"\0\0"
    )
    row = annotation_row(payload)
    assert row["N_samples"] == [100]
    assert row["start"] == 100
    assert row["end_exclusive"] == 151
    assert row["N_without_immediately_preceding_onset_indices"] == [0]
    assert row["event_count"] == 5
    assert row["events"][0]["symbol"] == "N"
    assert "qrs_mask" not in row


def test_non_wave_codes_and_duplicate_n_multiplicity_are_not_eligibility_filters():
    import struct

    from ecg_experiment.ssl_synthetic_ludb_landmark_audit import annotation_row

    payload = (
        b"".join(
            struct.pack("<H", (code << 10) | delta) for code, delta in [(28, 20), (1, 80), (1, 0), (40, 10)]
        )
        + b"\0\0"
    )
    row = annotation_row(payload)
    assert row["N_samples"] == [100, 100]
    assert row["start"] == 20
    assert row["event_count"] == 4
    assert row["events"][0]["symbol"] == "+"
    assert row["events"][0]["type_code"] == 28


def test_main_costs_add_once_and_exclude_new_audit_subtree(tmp_path):
    import hashlib
    import json

    from ecg_experiment.ssl_synthetic_ludb_landmark_audit_run import main_costs

    previous = {
        "current_machine_stages": 25,
        "current_machine_cumulative_cpu_seconds": 435.78864790999995,
        "current_machine_cumulative_gpu_wall_seconds": 447.98139213304967,
    }
    main = {"cpu_seconds_before_summary": 0.1, "wall_seconds_before_summary": 0.2}
    receipt = json.dumps(main)
    (tmp_path / "summary.json").write_text(receipt)
    (tmp_path / "independent_audit_v1").mkdir()
    (tmp_path / "independent_audit_v1/source.py").write_bytes(b"x" * 10000)
    ledger = {
        "receipt": "summary.json",
        "receipt_sha256": hashlib.sha256(receipt.encode()).hexdigest(),
        "prior_ledger_sha256": "prior",
        "current_machine_stages": 26,
        "cpu_seconds": 0.3,
        "wall_seconds": 0.4,
        "gpu_wall_seconds": 0,
        "peak_rss_bytes": 100000,
        "resource_gates": {"cpu": True},
        "output_bytes_before_terminal_ledger": len(receipt),
        "current_machine_cumulative_cpu_seconds": previous["current_machine_cumulative_cpu_seconds"] + 0.3,
        "current_machine_cumulative_gpu_wall_seconds": previous[
            "current_machine_cumulative_gpu_wall_seconds"
        ],
    }
    path = tmp_path / "prospective_ledger.json"
    path.write_text(json.dumps(ledger))
    assert main_costs(tmp_path, main, previous, "prior")["current_machine_stages"] == 26
    ledger["current_machine_cumulative_cpu_seconds"] += 0.3
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="once"):
        main_costs(tmp_path, main, previous, "prior")
