"""Guard the conservative machine-summary interpretation contract."""

import pytest

from ecg_experiment.mimic_machine_disagreement import machine_status


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["Sinus rhythm", "Summary: ABNORMAL ECG"], "abnormal"),
        (["Borderline ECG."], "borderline"),
        (["Normal ECG"], "normal"),
        (["Possible abnormal ECG"], "unclassified"),
        (["No evidence of abnormal ECG"], "unclassified"),
        (["Normal ECG", "Abnormal ECG"], "conflicting"),
    ],
)
def test_machine_status_requires_exact_summary(lines: list[str], expected: str) -> None:
    """Do not promote uncertain prose into an abnormal machine summary."""
    assert machine_status(lines) == expected
