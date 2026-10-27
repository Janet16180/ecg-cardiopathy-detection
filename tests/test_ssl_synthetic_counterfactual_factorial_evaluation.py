"""Artificial fixed-gate, calibration isolation and single-input factorial tests."""

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ecg_experiment import ssl_synthetic_counterfactual_factorial_evaluation as evaluation

CONFIG = {
    "bootstrap_draws": 20,
    "bootstrap_seed": 20261027,
    "primary_direct_ap_gain_minimum": 0.02,
    "edited_ap_noninferiority_margin": 0.01,
    "control_point_increase_maximum": 0.05,
}


def _passing_summary() -> dict:
    effects = {
        "primary_average_loss_effect": {"difference": 0.02, "ci_low": 0.001},
        "paired_loss_effect_R": {"ci_low": -0.009},
        "paired_loss_effect_S": {"ci_low": -0.009},
    }
    return {
        "factorial_contrasts": {"edited_average_precision": {"all": effects}},
        "responses": {
            arm: {
                "all": {
                    name: {"ci_low": 0.01} for name in ["average_precision_change", "selectivity_contrast"]
                }
            }
            for arm in ["RP", "SP"]
        },
        "paired_loss_control_flags": {
            init: {kind: {"difference": 0.05} for kind in ["sham", "nuisance"]} for init in ["R", "S"]
        },
    }


def test_factorial_primary_simple_effects_and_interaction() -> None:
    values = {
        arm: np.full(128, value) for arm, value in zip(evaluation.ARMS, [0.1, 0.2, 0.3, 0.6], strict=True)
    }
    result = evaluation.factorial_contrasts(values, CONFIG)
    assert result["primary_average_loss_effect"]["difference"] == pytest.approx(0.3)
    assert result["paired_loss_effect_R"]["difference"] == pytest.approx(0.2)
    assert result["paired_loss_effect_S"]["difference"] == pytest.approx(0.4)
    assert result["initialization_by_loss_interaction"]["difference"] == pytest.approx(0.2)
    assert result["ssl_effect_BCE"]["difference"] == pytest.approx(0.1)
    assert result["ssl_effect_paired"]["difference"] == pytest.approx(0.3)
    assert result["primary_average_loss_effect"]["subjects"] == 128


def test_all_twelve_gates_and_no_response_only_primary() -> None:
    base = _passing_summary()
    result = evaluation.decision(base, CONFIG)
    assert len(result["gates"]) == 12
    assert result["engineering_replication_admitted"]
    failures = []
    for key, value in [("difference", 0.0199), ("ci_low", 0.0)]:
        changed = copy.deepcopy(base)
        changed["factorial_contrasts"]["edited_average_precision"]["all"]["primary_average_loss_effect"][
            key
        ] = value
        for arm in ("RP", "SP"):
            changed["responses"][arm]["all"]["average_precision_change"]["ci_low"] = 100.0
        failures.append(changed)
    for init in ["R", "S"]:
        changed = copy.deepcopy(base)
        changed["factorial_contrasts"]["edited_average_precision"]["all"][f"paired_loss_effect_{init}"][
            "ci_low"
        ] = -0.01
        failures.append(changed)
        for name in ["average_precision_change", "selectivity_contrast"]:
            changed = copy.deepcopy(base)
            changed["responses"][init + "P"]["all"][name]["ci_low"] = 0.0
            failures.append(changed)
        for kind in ["sham", "nuisance"]:
            changed = copy.deepcopy(base)
            changed["paired_loss_control_flags"][init][kind]["difference"] = 0.050001
            failures.append(changed)
    assert len(failures) == 12
    for changed in failures:
        decision = evaluation.decision(changed, CONFIG)
        assert sum(decision["gates"].values()) == 11
        assert not decision["engineering_replication_admitted"]


def test_all_four_calibrations_ignore_b_rows_and_ties_are_native() -> None:
    cases = [
        {
            "role": "calibration",
            "kind": kind,
            "role_subject_index": subject,
            "case_index": 2000 + subject * 4 + offset,
        }
        for subject in range(64)
        for kind, offset in [("sham", 2), ("nuisance", 3)]
    ]
    maps = {arm: np.zeros((768, 1, 1), np.float32) for arm in evaluation.ARMS}
    for values in maps.values():
        for subject in range(64):
            values[subject * 4 + 2] = subject / 100
            values[subject * 4 + 3] = subject / 100 + 0.001
    first = evaluation.calibration_thresholds(cases, maps)
    frozen = {arm: values[:256].tobytes() for arm, values in maps.items()}
    for values in maps.values():
        values[256:] = 999
    assert evaluation.calibration_thresholds(cases, maps) == first
    assert all(values[:256].tobytes() == frozen[arm] for arm, values in maps.items())
    assert all(value == float(np.float32(0.601)) for value in first.values())
    assert evaluation.map_row(2256) == 256
    assert evaluation.map_row(2767) == 767


def test_predict_batch_observed_only_and_nonfinite_logits_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "ecg_experiment.ssl_synthetic_counterfactual_factorial_common",
        SimpleNamespace(
            render_checked=lambda context, index: {
                "edited": np.ones((1, 4), np.float32),
                "baseline": np.full((1, 4), 999.0, np.float32),
            }
        ),
    )

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, signal):
            assert signal.shape == (2, 1, 4)
            assert torch.all(signal == 1)
            return signal + self.value

    values = evaluation.predict_batch(Spy(), {}, [2000, 2001])
    assert values.dtype == np.float32
    model = Spy()
    with torch.no_grad():
        model.value.fill_(float("inf"))
    with pytest.raises(ValueError, match="logits"):
        evaluation.predict_batch(model, {}, [2000, 2001])


def test_patient_identifiers_must_be_local_and_complete() -> None:
    rows = [
        {"method": "RP", "kind": "positive", "mode": mode, "subject": subject, "value": 0.1}
        for subject in range(128)
        for mode in ["focal", "repeated"]
    ]
    assert evaluation._patient_values(rows, "RP", "value").shape == (128,)
    rows[0]["subject"] = 564
    with pytest.raises(ValueError, match="Every evaluation patient"):
        evaluation._patient_values(rows, "RP", "value")
