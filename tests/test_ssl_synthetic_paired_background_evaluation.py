"""Artificial checks for compact map coordinates, calibration isolation and advancement safeguards."""

import copy

import numpy as np
import pytest
import torch

from ecg_experiment import ssl_synthetic_paired_background_evaluation as evaluation


def test_global_native_rows_translate_to_compact_probability_grid() -> None:
    """TRAIN cases cannot accidentally index saved evaluation probabilities."""
    assert evaluation.map_row(2000) == 0
    assert evaluation.map_row(2256) == 256
    assert evaluation.map_row(2767) == 767
    for invalid in (0, 1999, 2768):
        with pytest.raises(ValueError, match="Only global"):
            evaluation.map_row(invalid)


def test_calibration_is_independent_of_evaluation_values_and_uses_higher_quantile() -> None:
    """Only64 explicitly mapped calibration patient controls determine thresholds."""
    cases = []
    maps = {arm: np.zeros((768, 1, 1), np.float32) for arm in ("R", "S")}
    for subject in range(64):
        for offset, kind in ((2, "sham"), (3, "nuisance")):
            cases.append(
                {
                    "case_index": 2000 + 4 * subject + offset,
                    "role": "calibration",
                    "role_subject_index": subject,
                    "kind": kind,
                }
            )
            for arm in maps:
                maps[arm][4 * subject + offset] = subject / 100
    first = evaluation.calibration_thresholds(cases, maps)
    expected = float(np.quantile(np.arange(64, dtype=np.float32) / 100, 0.95, method="higher"))
    assert first == {"R": expected, "S": expected}
    for values in maps.values():
        values[256:] = 1
    assert evaluation.calibration_thresholds(cases, maps) == first
    assert not expected > first["S"]


def passing_summary() -> tuple[dict, dict]:
    """Represent plausible primary improvement with all independently specified safeguards met."""
    summary = {
        "s_minus_r_response": {"all": {"difference": 0.03, "ci_low": 0.01}},
        "s_minus_r_ap": {"all": {"ci_low": -0.005}},
        "s_minus_r_control_flags": {"sham": {"difference": 0.02}, "nuisance": {"difference": 0.01}},
        "counterfactual_response": {
            arm: {
                "all": {
                    metric: {"ci_low": 0.001}
                    for metric in ("average_precision_change", "selectivity_contrast")
                }
            }
            for arm in ("R", "S")
        },
    }
    config = {
        "primary_response_gain_minimum": 0.02,
        "edited_ap_noninferiority_margin": 0.01,
        "control_point_increase_maximum": 0.05,
    }
    return summary, config


@pytest.mark.parametrize("failure", ["response_point", "response_ci", "edited_ap", "within_arm", "control"])
def test_primary_gain_alone_cannot_override_frozen_safeguards(failure: str) -> None:
    """An apparent response win must not conceal degraded localization or worse controls."""
    summary, config = passing_summary()
    assert evaluation.decision(summary, config)["engineering_replication_admitted"]
    bad = copy.deepcopy(summary)
    if failure == "response_point":
        bad["s_minus_r_response"]["all"]["difference"] = 0.01
    elif failure == "response_ci":
        bad["s_minus_r_response"]["all"]["ci_low"] = 0
    elif failure == "edited_ap":
        bad["s_minus_r_ap"]["all"]["ci_low"] = -0.011
    elif failure == "within_arm":
        bad["counterfactual_response"]["R"]["all"]["selectivity_contrast"]["ci_low"] = 0
    else:
        bad["s_minus_r_control_flags"]["sham"]["difference"] = 0.051
    result = evaluation.decision(bad, config)
    assert not result["engineering_replication_admitted"]
    assert not result["clinical_promotion"]


def test_shared_inference_receives_only_observed_waveform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Baseline, masks and metadata never become network input features."""
    edited = np.full((12, 2500), 0.123, np.float32)
    monkeypatch.setattr(
        evaluation,
        "render_checked",
        lambda *_: {
            "edited": edited,
            "baseline": np.full_like(edited, 100),
            "support": np.ones_like(edited, bool),
        },
    )

    class Recorder(torch.nn.Module):
        """Record the waveform supplied to an otherwise trivial dense model."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.seen = None

        def forward(self, signal: torch.Tensor) -> torch.Tensor:
            """Keep the exact model input for this artificial leakage check."""
            self.seen = signal.detach().numpy().copy()
            return signal * self.weight

    model = Recorder()
    result = evaluation.predict_batch(model, {}, [2000])
    np.testing.assert_array_equal(model.seen[0], edited)
    assert result.shape == (1, 12, 2500)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("bad", [float("inf"), -float("inf"), float("nan")])
def test_infinite_logits_cannot_be_hidden_by_sigmoid(monkeypatch: pytest.MonkeyPatch, bad: float) -> None:
    """Sigmoid maps infinity to a finite probability, so raw logits need their own guard."""
    monkeypatch.setattr(evaluation, "render_checked", lambda *_: {"edited": np.zeros((12, 2500), np.float32)})
    model = torch.nn.Conv1d(12, 12, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.fill_(bad)
    with pytest.raises(ValueError, match="logits"):
        evaluation.predict_batch(model, {}, [2000])
