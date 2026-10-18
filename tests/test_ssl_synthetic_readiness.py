"""Readiness checks must fail closed without reading patient payloads."""

import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from ecg_experiment.ssl_synthetic_readiness import inspect_prerequisites, validate_spec


def specification():
    return {
        "schema_version": 1,
        "stream": "ssl_synthetic_localization",
        "checks": [
            {"name": "payload", "path": "data/signal.npy", "kind": "file", "required": True},
            {"name": "weights", "path": "weights.pt", "kind": "file", "required": False},
        ],
    }


def test_missing_required_path_and_optional_path(tmp_path):
    report = inspect_prerequisites(specification(), tmp_path)
    assert report["missing_required"] == ["payload"]
    assert not report["required_paths_present"]


def test_present_payload_is_never_opened_or_certified(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/signal.npy").write_bytes(b"not a valid waveform")
    with patch.object(Path, "open", side_effect=AssertionError("Payload was opened")):
        report = inspect_prerequisites(specification(), tmp_path)
    assert report["required_paths_present"]
    assert report["scientific_readiness"] == "not_assessed"
    assert not report["performance_scored"]


def test_directory_cannot_satisfy_file_check(tmp_path):
    (tmp_path / "data/signal.npy").mkdir(parents=True)
    report = inspect_prerequisites(specification(), tmp_path)
    assert not report["required_paths_present"]


@pytest.mark.parametrize("value", ["../outside", "/absolute", "", "."])
def test_invalid_configuration_fails_before_any_target_inspection(value):
    spec = specification()
    spec["checks"][0]["path"] = value
    with patch.object(Path, "is_file") as inspect:
        with pytest.raises(ValueError, match="repository-relative"):
            inspect_prerequisites(spec)
        inspect.assert_not_called()


def test_duplicate_names_and_non_boolean_required_flags_fail():
    original = specification()
    duplicate = copy.deepcopy(original)
    duplicate["checks"][1]["name"] = "payload"
    numeric = copy.deepcopy(original)
    numeric["checks"][0]["required"] = 1
    for spec, message in [(duplicate, "unique nonempty name"), (numeric, "Invalid kind or required")]:
        with pytest.raises(ValueError, match=message):
            validate_spec(spec)
