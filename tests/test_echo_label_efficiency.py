"""Checks of the Experiment 028 rows, draws, targets and runner analysis on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.echo_label_efficiency import (
    budget_summary,
    check_keys,
    draw_plans,
    gain_target,
    smallest_budget_by_draws,
    split_rows,
)
from ecg_experiment.eda.echonext import COMPOSITE
from scripts.experiments import run_echo_label_efficiency028 as run


def echo_rows() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"patient_key": np.repeat(np.arange(200), 3), "split": "train",
                          COMPOSITE: (rng.random(600) < 0.5).astype(int), "use": rng.random(600) > 0.1})
    val = pd.DataFrame({"patient_key": np.arange(1000, 1100), "split": "val",
                        COMPOSITE: (rng.random(100) < 0.4).astype(int), "use": rng.random(100) > 0.1})
    frame = pd.concat([train, val], ignore_index=True)
    frame.index = pd.Index(np.arange(len(frame)) + 5000, name="ecg_key")
    return frame


def test_split_rows_keeps_usable_training_and_all_validation_rows():
    rows = echo_rows()
    train, val_all, val = split_rows(rows)
    assert len(train) == int(rows.loc[rows["split"] == "train", "use"].sum())
    assert len(val_all) == 100
    assert val["use"].all()
    assert len(val) == int(val_all["use"].sum())


def test_split_rows_rejects_repeated_validation_patients():
    rows = echo_rows()
    rows.loc[rows["split"] == "val", "patient_key"] = 7
    with pytest.raises(ValueError, match="several usable"):
        split_rows(rows)


def test_check_keys_requires_the_same_order():
    train, val_all, _ = split_rows(echo_rows())
    check_keys(train.index.to_numpy(), val_all.index.to_numpy(), train, val_all)
    with pytest.raises(ValueError, match="Training"):
        check_keys(train.index.to_numpy()[::-1], val_all.index.to_numpy(), train, val_all)


def test_draw_plans_are_paired_across_budgets_and_end_with_the_pool():
    train, _, _ = split_rows(echo_rows())
    patients, y = train["patient_key"].to_numpy(), train[COMPOSITE].to_numpy()
    plans = draw_plans(patients, y, (20, 50), draws=3, seed=28028)
    assert [(budget, draw, seed) for budget, draw, seed, _ in plans[:3]] == [
        ("20", 0, 28028), ("20", 1, 28029), ("20", 2, 28030)]
    assert all(len(set(patients[positions])) == int(budget) for budget, _, _, positions in plans[:-1])
    assert plans[-1][0] == "all"
    assert np.array_equal(plans[-1][3], np.arange(len(y)))


def test_gain_target_keeps_the_share_of_gain_over_chance():
    assert gain_target(0.9, 0.95) == pytest.approx(0.88)
    assert gain_target(0.5, 0.95) == 0.5


def test_smallest_budget_by_draws_needs_ninety_percent():
    values = {100: np.array([0.8] * 17 + [0.6] * 3), 250: np.array([0.8] * 18 + [0.6] * 2)}
    assert smallest_budget_by_draws(values, 0.7) == 250
    assert smallest_budget_by_draws(values, 0.9) is None


def draw_table() -> pd.DataFrame:
    rows = []
    for draw in range(20):
        for name, auroc in {"a": 0.8 + draw * 1e-4, "b": 0.75, "c": 0.8 + (-1) ** draw * 1e-3}.items():
            rows.append({"budget": "100", "draw": draw, "encoder": name, "auroc": auroc,
                         "average_precision": auroc - 0.1})
    return pd.DataFrame(rows)


def test_budget_summary_reads_paired_contrasts():
    summary = budget_summary(draw_table(), "100", ("a", "b", "c"), (("a", "b"), ("c", "a")))
    assert summary["encoders"]["b"]["auroc"]["mean"] == pytest.approx(0.75)
    assert summary["contrasts"]["a_minus_b"]["auroc"]["positive_draws"] == 20
    assert summary["reading"] == {"a_minus_b": "a better", "c_minus_a": "no clear difference"}


def synthetic_study(monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"patient_key": np.repeat(np.arange(300), 2), COMPOSITE: rng.integers(0, 2, 600)})
    train.index = pd.Index(np.arange(600), name="ecg_key")
    val = pd.DataFrame({"patient_key": np.arange(1000, 1200), COMPOSITE: rng.integers(0, 2, 200)})
    signal = {"strong": 2.0, "weak": 0.5}

    def fake_inputs(name: str, train: pd.DataFrame, val: pd.DataFrame, use: np.ndarray) -> tuple:
        make = np.random.default_rng(len(name))
        pool = make.normal(size=(len(train), 3)) + signal[name] * train[COMPOSITE].to_numpy()[:, None]
        held = make.normal(size=(len(val), 3)) + signal[name] * val[COMPOSITE].to_numpy()[:, None]
        return pool, held

    monkeypatch.setattr(run, "inputs", fake_inputs)
    return train, val, np.ones(len(val), dtype=bool)


def test_evaluate_scores_every_draw_and_readout(monkeypatch: pytest.MonkeyPatch):
    train, val, use = synthetic_study(monkeypatch)
    plans = draw_plans(train["patient_key"].to_numpy(), train[COMPOSITE].to_numpy(), (50,), 2, 28028)
    rows, last, _ = run.evaluate("strong", plans, train, val, use, ("primary", "secondary"))
    assert [(row["budget"], row["readout"]) for row in rows][:2] == [("50", "primary"), ("50", "secondary")]
    assert len(rows) == 6
    assert rows[0]["records"] == rows[0]["patients"] == 50
    assert rows[-1]["records"] == 600
    assert set(last) == {"primary", "secondary"}
    assert rows[-2]["auroc"] > 0.8


def test_analyse_reports_targets_and_bootstrap(monkeypatch: pytest.MonkeyPatch):
    train, val, use = synthetic_study(monkeypatch)
    monkeypatch.setattr(run, "ENCODERS", ("strong", "weak"))
    monkeypatch.setattr(run, "CONTRASTS", (("strong", "weak"),))
    monkeypatch.setattr(run, "BUDGETS", (50, 100))
    monkeypatch.setattr(run, "READ_BUDGETS", ("50",))
    monkeypatch.setattr(run, "BOOTSTRAP_DRAWS", 50)
    plans = draw_plans(train["patient_key"].to_numpy(), train[COMPOSITE].to_numpy(), (50, 100), 20, 28028)
    outputs = {name: run.evaluate(name, plans, train, val, use, ("primary",)) for name in run.ENCODERS}
    table = pd.DataFrame([row for rows, _, _ in outputs.values() for row in rows])
    finals = {name: last["primary"] for name, (_, last, _) in outputs.items()}
    result = run.analyse(table, finals, val, tabular=0.6)
    assert result["reading"]["50"]["strong_minus_weak"] == "strong better"
    assert result["all_bootstrap"]["strong_minus_weak"]["difference"] > 0
    assert result["targets"]["strong"]["smallest_budget_mean_tabular"] == 50
    assert result["chosen_c"]["all"]["weak"] == {"0.01": 1}
