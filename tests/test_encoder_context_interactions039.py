"""Scientific and receipt checks for supplementary CPC factorial interactions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ecg_experiment import encoder_context_interactions039 as analysis
from ecg_experiment.intervals import patient_groups, two_class_resamples


def _arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    y = np.tile([0, 1], 8)
    arrays = {"targets": y, "patient_ids": np.repeat(np.arange(8), 2), "record_ids": np.arange(len(y))}
    for tier in analysis.TIERS:
        for encoder in analysis.ENCODERS:
            for budget in analysis.BUDGETS:
                for context in analysis.CONTEXTS:
                    arrays[f"{tier}_{encoder}_{budget}_{context}"] = rng.random((3, len(y))) + y * 0.3
    return arrays


def _independent_delta(arrays: dict[str, np.ndarray], rows: np.ndarray) -> list[float]:
    y = arrays["targets"][rows]
    scores = [
        arrays[f"25_{encoder}_limited_{context}"][:, rows]
        for encoder, context in (("patch", "xlstm"), ("cnn", "xlstm"), ("patch", "gru"), ("cnn", "gru"))
    ]
    return [
        (roc_auc_score(y, a) - roc_auc_score(y, b)) - (roc_auc_score(y, c) - roc_auc_score(y, d))
        for a, b, c, d in zip(*scores, strict=True)
    ]


def test_mean_seed_interaction_differs_from_score_ensemble() -> None:
    arrays = _arrays()
    result = analysis.analyze_interactions(arrays, draws=20)
    interaction = result["interactions"]["25k_limited_patch_by_context"]
    expected = _independent_delta(arrays, np.arange(len(arrays["targets"])))
    np.testing.assert_array_equal(interaction["seed_differences"], expected)
    assert interaction["difference"] == np.mean(expected)
    ensembled = {
        key: value.mean(axis=0, keepdims=True) if value.ndim == 2 else value for key, value in arrays.items()
    }
    assert interaction["difference"] != _independent_delta(ensembled, np.arange(len(arrays["targets"])))[0]
    assert len(result["interactions"]) == 8
    assert result["status"] == "exploratory"
    assert result["primary_decision"] is False
    assert result["training_seeds"] == [39042, 39043, 39044]


def test_shared_patient_draws_match_independent_interval_and_repeat() -> None:
    arrays = _arrays()
    result = analysis.analyze_interactions(arrays, draws=20, seed=39045)
    assert result == analysis.analyze_interactions(arrays, draws=20, seed=39045)
    rows = two_class_resamples(
        patient_groups(arrays["patient_ids"]), arrays["targets"], 20, np.random.default_rng(39045)
    )
    values = [np.mean(_independent_delta(arrays, draw)) for draw in rows]
    low, high = np.percentile(values, [2.5, 97.5])
    interaction = result["interactions"]["25k_limited_patch_by_context"]
    assert interaction["ci_low"] == low
    assert interaction["ci_high"] == high
    assert result["skipped_draws"] == 20 - len(values)
    changed = analysis.analyze_interactions(arrays, draws=20, seed=39046)
    assert changed["interactions"] != result["interactions"]


def test_rows_and_realized_seeds_remain_paired() -> None:
    arrays = _arrays()
    for tier in analysis.TIERS:
        for budget in analysis.BUDGETS:
            for context in analysis.CONTEXTS:
                baseline = arrays[f"{tier}_cnn_{budget}_{context}"]
                for encoder in analysis.ENCODERS[1:]:
                    arrays[f"{tier}_{encoder}_{budget}_{context}"] = baseline.copy()
    result = analysis.analyze_interactions(arrays, draws=20)
    for interaction in result["interactions"].values():
        assert interaction["difference"] == interaction["ci_low"] == interaction["ci_high"] == 0
        assert interaction["seed_differences"] == [0, 0, 0]


def test_single_class_draws_are_counted_and_all_invalid_draws_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    arrays = _arrays()
    arrays["patient_ids"] = arrays["targets"].copy()
    result = analysis.analyze_interactions(arrays, draws=20)
    expected = len(
        list(
            two_class_resamples(
                patient_groups(arrays["patient_ids"]), arrays["targets"], 20, np.random.default_rng(39045)
            )
        )
    )
    assert 0 < result["skipped_draws"] == 20 - expected
    assert result["accepted_draws"] == expected
    monkeypatch.setattr(analysis, "two_class_resamples", lambda *args: iter(()))
    with pytest.raises(ValueError, match="Every interaction"):
        analysis.analyze_interactions(arrays, draws=20)


@pytest.mark.parametrize("draws", [0, -1, 1.5, True])
def test_invalid_draw_counts_raise(draws: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        analysis.analyze_interactions(_arrays(), draws=draws)


@pytest.mark.parametrize("problem", ["class", "labels", "patients", "records", "seeds", "rows", "nan"])
def test_invalid_arrays_raise(problem: str) -> None:
    arrays = _arrays()
    if problem == "class":
        arrays["targets"][:] = 0
    if problem == "labels":
        arrays["targets"][0] = 2
    if problem == "patients":
        arrays["patient_ids"] = arrays["patient_ids"][:-1]
    if problem == "records":
        arrays["record_ids"][0] = arrays["record_ids"][1]
    if problem == "seeds":
        arrays["25_patch_limited_xlstm"] = arrays["25_patch_limited_xlstm"][:2]
    if problem == "rows":
        arrays["25_patch_limited_xlstm"] = arrays["25_patch_limited_xlstm"][:, :-1]
    if problem == "nan":
        arrays["25_patch_limited_xlstm"][0, 0] = np.nan
    with pytest.raises(ValueError, match="required|aligned|finite"):
        analysis.analyze_interactions(arrays, draws=2)


def test_receipt_binds_inputs_and_sources_and_refuses_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = _arrays()
    hashes = {"outputs/cell/result.json": "original"}
    monkeypatch.setattr(analysis, "load_cells", lambda root: (arrays, hashes.copy()))
    original = analysis.analyze_interactions
    monkeypatch.setattr(analysis, "analyze_interactions", lambda data: original(data, draws=10))
    receipt = analysis.write_interactions(tmp_path)
    destination = tmp_path / analysis.OUTPUT / "interactions.json"
    assert json.loads(destination.read_text()) == receipt
    assert receipt["audited_cells"] == 18
    assert receipt["analysis_role"] == "exploratory"
    assert receipt["primary_decision"] is receipt["model_selection"] is False
    assert analysis.SOURCE_FILES[0] in receipt["source_sha256"]
    assert analysis.SOURCE_FILES[1] in receipt["source_sha256"]
    assert all(not Path(name).is_absolute() for name in receipt["source_sha256"])
    assert analysis.write_interactions(tmp_path) == receipt
    hashes["outputs/cell/result.json"] = "changed"
    with pytest.raises(ValueError, match="inputs or sources changed"):
        analysis.write_interactions(tmp_path)
    hashes["outputs/cell/result.json"] = "original"
    monkeypatch.setattr(analysis, "sha256_file", lambda path: "changed-source")
    with pytest.raises(ValueError, match="inputs or sources changed"):
        analysis.write_interactions(tmp_path)
