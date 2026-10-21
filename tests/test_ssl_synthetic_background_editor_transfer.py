"""Artificial tests for prospective background/editor transfer, with no experimental inputs."""

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_background_editor_transfer as core
from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_randomized_localization as original


def test_original_four_variants_and_targets_are_exact() -> None:
    """The new constructor must preserve the frozen generated-training editor byte for byte."""
    raw = np.random.default_rng(901).normal(size=(8, 5000)).astype(np.float32)
    standardized, _ = quality.standardize(raw)
    observed, strict, support, metadata = core.make_case(standardized, 2003)
    old, old_strict, old_support, _ = original.make_case(raw, 2003)
    assert np.array_equal(observed[:4], old)
    assert np.array_equal(strict[:2], old_strict)
    assert np.array_equal(support[:2], old_support[2:])
    assert all(row["outside_hull_cells"] == 0 for row in metadata["checks"])


def test_held_out_kernel_changes_sign_and_matches_nominal_energy() -> None:
    """New kernels differ in shape rather than receiving a new size/energy advantage."""
    for width in [61, 81, 101]:
        hann, biphasic = core.kernel(width, "hann"), core.kernel(width, "biphasic")
        assert np.any(biphasic > 0)
        assert np.any(biphasic < 0)
        assert abs(biphasic.mean()) < 1e-15
        assert np.isclose(np.sum(hann**2), np.sum(biphasic**2), rtol=0, atol=1e-13)
        assert not np.array_equal(hann, biphasic)


def test_final_canonical_difference_controls_every_support() -> None:
    """Rounded limb derivation, nuisance and every nonzero final change determine the targets."""
    native = np.random.default_rng(903).normal(size=(8, 5000)).astype(np.float32)
    observed, strict, support, _ = core.make_case(native, 2014)
    difference = observed[2:].astype(np.float64) - observed[1].astype(np.float64)
    assert np.array_equal(strict, difference != 0)
    assert np.array_equal(support, np.abs(difference) > 0.01)
    assert np.array_equal(observed[0], quality.derive_leads(native))
    assert not np.array_equal(strict, support)


def test_metrics_use_matched_control_and_reject_wrong_final_support() -> None:
    """Perfect artificial scores localize edits; a stale mask cannot enter scoring."""
    native = np.random.default_rng(904).normal(size=(8, 5000)).astype(np.float32)
    observed, _, support, _ = core.make_case(native, 2000)
    maps = np.full(observed.shape, 0.2, dtype=np.float32)
    maps[2:] = np.where(support, 0.9, 0.1)
    row = core.score_record(maps, observed, support, "real", 0, 1, "B", np.roll(support, 800, axis=2))
    for positive in row["positives"].values():
        assert positive["edited_ap"] == 1
        assert positive["nuisance_site_ap"] == positive["prevalence"]
        assert positive["response_ap"] > 0
    changed = support.copy()
    changed[0, 0, 0] = ~changed[0, 0, 0]
    with pytest.raises(ValueError, match="support identity"):
        core.score_record(maps, observed, changed, "real", 0, 1, "B", np.roll(support, 800, axis=2))


def artificial_rows() -> list[dict]:
    """Return a complete fixture with intentionally heterogeneous record effects."""
    return [
        {
            "source": source,
            "slot": slot,
            "fit_id": fit,
            "arm": arm,
            "positives": {
                name: dict.fromkeys(
                    [
                        "edited_ap",
                        "nuisance_site_ap",
                        "response_ap",
                        "amplitude_ap",
                        "amplitude_gain",
                        "donor_site_ap",
                        "donor_gain",
                        "selectivity",
                    ],
                    (slot - 20) / 100 + fit / 1000,
                )
                for name in core.POSITIVES
            },
            "controls": {
                name: dict.fromkeys(["mean", "q95", "maximum", "top600_mean", "maximum_gt_0p5"], 0.1)
                for name in core.VARIANTS[:2]
            },
        }
        for source in core.SOURCES
        for slot in range(64)
        for fit in core.FITS
        for arm in core.ARMS
    ]


def test_shared_record_bootstrap_preserves_fixed_fits_and_complete_roster() -> None:
    """All contrasts use the same resampled slots; missing/repeated fits fail."""
    rows = artificial_rows()
    config = {"bootstrap": {"seed": 2026106601, "draws": 2000}}
    statistics, draws = core.summarize(rows, config)
    assert draws.shape == (2000, len(statistics["endpoint_order"]))
    assert statistics["scientific_passed"]
    position = statistics["endpoint_order"].index("real__B__biphasic__response_ap")
    rng = np.random.default_rng(2026106601)
    manual = np.array([np.mean((rng.integers(0, 64, 64) - 20) / 100 + 0.002) for _ in range(2000)])
    np.testing.assert_allclose(draws[:, position], manual, rtol=0, atol=1e-15)
    assert all(
        value["mean"] == 0 for key, value in statistics["endpoints"].items() if key.startswith("interaction")
    )
    with pytest.raises(ValueError, match="exactly once"):
        core.summarize(rows[:-1], config)
    with pytest.raises(ValueError, match="exactly once"):
        core.summarize(rows + [rows[0]], config)


def test_donor_failure_cannot_be_rescued_by_other_primary_components() -> None:
    """The fixed conjunction retains an unfavorable alignment-control result."""
    rows = artificial_rows()
    for row in rows:
        if row["source"] == "real" and row["arm"] == "B":
            for variant in ["biphasic_focal", "biphasic_repeated"]:
                row["positives"][variant]["donor_gain"] = -0.01
    statistics, _ = core.summarize(rows, {"bootstrap": {"seed": 2026106601, "draws": 2000}})
    assert not statistics["scientific_passed"]
    assert len(statistics["scientific_gates"]) == 6
    assert sum(statistics["scientific_gates"].values()) == 4
    assert len(statistics["per_fit_mode"]) == 96


def test_runtime_preflight_rejects_missing_rosters_before_any_array_read(tmp_path, monkeypatch) -> None:
    """Incomplete source admissions stop at metadata rather than spending a failed ECG run."""
    import json

    from ecg_experiment import ssl_synthetic_background_editor_transfer_run as execution

    config = {"sources": list(core.SOURCES), "variants": list(core.VARIANTS), "prior_ledger": "prior.json"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "admission.json").write_text(json.dumps({"source_sha256": {}, "input_sha256": {}}))
    monkeypatch.setattr(
        execution,
        "from_stored",
        lambda name: tmp_path / ("config.json" if name == execution.CONFIG else name),
    )
    with pytest.raises(ValueError, match="roster incomplete"):
        execution.preflight("admission.json")


def test_cumulative_gpu_guard_charges_previous_and_active_intervals() -> None:
    """Separate profile/full lock phases cannot reset the shared GPU allowance."""
    import time

    from ecg_experiment import ssl_synthetic_background_editor_transfer_run as execution

    clock = {
        "cpu": time.process_time(),
        "wall": time.perf_counter(),
        "gpu_cost": 297.0,
        "active_gpu_start": time.perf_counter() - 2,
    }
    config = {
        "resources": {
            "cpu_seconds": 1800,
            "wall_seconds": 2400,
            "rss_bytes": 2**40,
            "output_bytes": 2**40,
            "gpu_lock_seconds": 300,
        }
    }
    with pytest.raises(TimeoutError, match="cumulative GPU-lock"):
        execution.resource_guard(config, clock)
