"""Synthetic checks of the Experiment 024 runner's task, structure, audit and plotting glue."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import normalize

from scripts.experiments import run_embedding_geometry024 as runner


def frame(rows: int = 240, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    combination = np.array(["NORM", "MI", "CD+MI", "STTC"])[np.arange(rows) % 4]
    subclasses = [["IMI"] if "MI" in value else ([] if value == "NORM" else ["STTC"])
                  for value in combination]
    table = pd.DataFrame({
        "patient_id": [f"p{index // 2}" for index in range(rows)],
        "ecg_id": np.arange(rows),
        "device": np.where(np.arange(rows) % 3 == 0, "a", "b"),
        "combination": combination,
        "subclasses": subclasses,
        "standard": (combination != "NORM").astype(float),
        "male": (np.arange(rows) % 2).astype(float),
        "age": np.where(np.arange(rows) % 10 == 0, np.nan, 20 + np.arange(rows) % 60),
        "heart_rate": 50 + np.arange(rows) % 70,
        "validated_by_human": np.arange(rows) % 2 == 0,
        "scp_codes": "{}", "report": "",
    }, index=[f"ptbxl:{index}" for index in range(rows)])
    x = rng.normal(size=(rows, 6))
    x[:, 0] += 2 * table["standard"].to_numpy()
    return table, x


def test_task_membership_and_eligible_subclasses():
    table, _ = frame()
    assert runner.task_membership(table, "MI", "superclass").sum() == 120
    assert runner.task_membership(table, "IMI", "subclass").sum() == 120
    assert runner.eligible_subclasses(table, table) == ["IMI"]


def test_evaluate_task_reports_every_method_and_contrast():
    table, x = frame()
    u = normalize(x)
    y = table["standard"].to_numpy(dtype=np.int64)
    result = runner.evaluate_task(u, y, table["device"].to_numpy(), u, table, y, x[:, 0],
                                  runner.ENCODER_CONTRASTS)
    assert result["auroc"]["probe"] > 0.8
    assert set(result["bootstrap"]) == {"prototype_minus_probe", "knn_other_device_minus_knn",
                                        "prototype_device_balanced_minus_prototype"}
    assert result["query_rows"] == len(table)


def test_one_vs_norm_keeps_only_task_and_norm_rows():
    table, x = frame()
    u = normalize(x)
    result = runner.one_vs_norm(table, table, x, x, u, u, ["MI"], "superclass")
    assert result["MI"]["training_rows"] == 180
    assert result["MI"]["training_positives"] == 120


def test_nuisance_and_structure_run_on_synthetic_rows():
    table, x = frame()
    summary = runner.nuisance(x, x, table, table, ["a"])
    assert set(summary) == {"device_auroc", "sex_auroc", "age_r2", "heart_rate_r2", "rows"}
    profile = runner.cluster_profile(np.arange(len(table)) % 2, table)
    assert profile["0"]["records"] == 120
    assert profile["0"]["norm_only"] == pytest.approx(0.5)


def test_label_audit_flags_rows_surrounded_by_the_other_label():
    table, _ = frame(rows=60)
    y = table["standard"].to_numpy()
    u = normalize(np.column_stack([y, 1 - y, np.zeros(len(y))]))
    flipped = table.copy()
    flipped.iloc[1, flipped.columns.get_loc("standard")] = 0.0
    audit, summary = runner.label_audit(flipped, u, u)
    assert summary["candidates"] == 1
    assert audit["ecg_id"].tolist() == [1]


def test_plot_summary_writes_a_figure(tmp_path: Path):
    methods = ("probe", "prototype", "prototype_device_balanced", "multi_prototype_2", "multi_prototype_4",
               "multi_prototype_8", "knn", "knn_other_device")
    encoders = ("cpc", "jepa", "xecg", "released_cpc")
    comparison = {name: {"auroc": dict.fromkeys(methods, 0.9)} for name in encoders}
    runner.plot_summary(comparison, tmp_path / "summary.png")
    assert (tmp_path / "summary.png").stat().st_size > 0
