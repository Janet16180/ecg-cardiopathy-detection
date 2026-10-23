"""Artificial-only qualification-runner failure propagation and artifact contracts."""

import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_edb_qualification_run as runner
from ecg_experiment.files import sha256_file


def artificial_result():
    policy = runner.operating_config(ROOT)
    result = runner.core.qualify_events([], policy)
    result["labels"][0, 0] = 1
    result["labels"][1, 0] = 2
    result["reasons"][:, 0] = 0
    return result


def decoded(issues=()):
    return {"metadata": {"decoder_issues": list(issues), "event_count": 0}, "events": []}


def test_decoder_issue_forces_complete_unknown128_and_recounts_exact_saved_grid():
    original = artificial_result()
    issue = {"kind": "WFDB_effective_view_differs_from_declared_filter"}
    result = runner.finalized_record(decoded([issue]), original)
    assert result["issues"] == [issue]
    assert not result["engineering_passed"]
    assert result["labels"].shape == result["reasons"].shape == (2, 720)
    assert result["labels"].dtype == np.uint8
    assert result["reasons"].dtype == np.uint16
    assert np.all(result["labels"] == 0)
    assert np.all(result["reasons"] == 128)
    assert result["counts"]["ST_positive_windows"] == 0
    assert result["counts"]["reference_windows"] == 0
    assert result["counts"]["unknown_windows"] == 1440
    assert result["counts"]["issues"] == 1
    assert original["labels"][0, 0] == 1
    assert original["labels"][1, 0] == 2


def test_semantic_and_decoder_issues_are_both_preserved_without_double_counting_cells():
    original = artificial_result()
    semantic = {"kind": "unclosed_episode", "event_index": 4}
    decoder = {"kind": "WFDB_unsafe_definition_scan", "rdann_comparison_run": False}
    original["issues"] = [semantic]
    result = runner.finalized_record(decoded([decoder]), original)
    assert result["issues"] == [semantic, decoder]
    assert result["counts"]["issues"] == 2
    assert result["counts"]["unknown_windows"] == 1440
    assert np.all(result["reasons"] == 128)


def test_no_issues_preserves_eligible_cells_and_recounts_stale_tentative_counts():
    result = runner.finalized_record(decoded(), artificial_result())
    assert result["engineering_passed"]
    assert result["labels"][0, 0] == 1
    assert result["labels"][1, 0] == 2
    assert result["counts"]["ST_positive_windows"] == 1
    assert result["counts"]["reference_windows"] == 1
    assert result["counts"]["unknown_windows"] == 1438
    assert (
        sum(result["counts"][key] for key in ["ST_positive_windows", "reference_windows", "unknown_windows"])
        == 1440
    )


@pytest.mark.parametrize("decoder_issue", [False, True])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("labels", np.zeros((2, 719), dtype=np.uint8)),
        ("labels", np.zeros((2, 720), dtype=np.int16)),
        ("reasons", np.ones((1, 720), dtype=np.uint16)),
        ("reasons", np.ones((2, 720), dtype=np.uint8)),
    ],
)
def test_malformed_core_array_shape_or_dtype_is_not_hidden_by_record_issues(field, value, decoder_issue):
    result = artificial_result()
    result[field] = value
    issues = [{"kind": "known_decoder_issue"}] if decoder_issue else []
    with pytest.raises(ValueError, match="geometry|dtype"):
        runner.finalized_record(decoded(issues), result)


@pytest.mark.parametrize("label", [3, 255])
def test_noncontract_label_values_cannot_escape_census_accounting(label):
    result = artificial_result()
    result["labels"][0, 0] = label
    with pytest.raises(ValueError, match="label|Label"):
        runner.finalized_record(decoded(), result)


@pytest.mark.parametrize(("label", "reason"), [(1, 1), (2, 64), (0, 0), (0, 256)])
def test_invalid_label_reason_relationships_are_rejected(label, reason):
    result = artificial_result()
    result["labels"][0, 0] = label
    result["reasons"][0, 0] = reason
    with pytest.raises(ValueError, match="reason|Reason"):
        runner.finalized_record(decoded(), result)


def test_operating_overlay_preserves_every_other_base_setting_and_all_source_bytes():
    sources = [runner.CONFIG, runner.CONTRACT, runner.EXECUTION, runner.GROUPING]
    before = {name: sha256_file(ROOT / name) for name in sources}
    base = json.loads((ROOT / runner.CONFIG).read_text())
    expected = json.loads(json.dumps(base))
    expected["contract"] = json.loads((ROOT / runner.CONTRACT).read_text())
    expected["grouping"] = json.loads((ROOT / runner.GROUPING).read_text())["grouping"]
    expected["resources"]["qualification"] = json.loads((ROOT / runner.EXECUTION).read_text())["resources"][
        "qualification"
    ]
    policy = runner.operating_config(ROOT)
    assert policy == expected
    assert policy["grouping"]["expected_document_components"] == 79
    assert policy["grouping"]["explicit_groups"][0] == ["e0118", "e0119", "e0121", "e0122"]
    assert policy["resources"]["qualification"] == {
        "cpu_seconds": 120,
        "wall_seconds": 300,
        "rss_bytes": 2 * 1024**3,
        "output_bytes": 16 * 1024**2,
    }
    assert {name: sha256_file(ROOT / name) for name in sources} == before


def test_actual_installed_package_identity_binds_real_extension_and_shared_alias():
    identity = runner.package_identity(ROOT)
    assert identity["numpy"] == np.__version__
    assert identity["wfdb"] == importlib.metadata.version("wfdb")
    files = identity["file_sha256"]
    prefix = f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/"
    assert len(files) == 7
    assert all(name.startswith(prefix) and not Path(name).is_absolute() for name in files)
    assert prefix + "wfdb/io/annotation.py" in files
    assert prefix + "wfdb/io/record.py" in files
    assert any("numpy/_core/_multiarray_umath" in name and name.endswith(".so") for name in files)
    assert all(sha256_file(ROOT / name) == digest for name, digest in files.items())


def test_role_summary_counts_linked_records_once_as_capable_components():
    policy = runner.operating_config(ROOT)
    positive = runner.finalized_record(decoded(), artificial_result())["counts"]
    unknown = runner.finalized_record(decoded([{"kind": "issue"}]), artificial_result())["counts"]
    role = policy["roles"]["names"][0]
    rows = [
        {"record": "artificial1", "component": "linked", "role": role, "counts": positive},
        {"record": "artificial2", "component": "linked", "role": role, "counts": positive},
        {"record": "artificial3", "component": "other", "role": role, "counts": unknown},
    ]
    summary = runner.record_summary(rows, policy)
    assert (summary["records"], summary["components"], summary["channels"]) == (3, 2, 6)
    assert summary["roles"][role]["ST_capable_components"] == 1
    assert summary["roles"][role]["reference_capable_components"] == 1
    assert summary["roles"][role]["ST_positive_windows"] == 2
    assert summary["roles"][role]["unknown_windows"] == 4316
    assert all(summary["roles"][name]["records"] == 0 for name in policy["roles"]["names"][1:])


@pytest.mark.parametrize("reason", [128, 129, 255])
def test_record_issue_reason_bit_requires_an_actual_record_issue(reason):
    result = artificial_result()
    result["labels"][0, 0] = 0
    result["reasons"][0, 0] = reason
    with pytest.raises(ValueError, match="record|reason|Reason"):
        runner.finalized_record(decoded(), result)


def test_descriptives_preserve_signed_episode_multiplicity_and_raw_quality_seconds():
    result = artificial_result()
    result["labels"][:] = 0
    result["reasons"][:] = 1
    result["reasons"][0, 0] = 7
    result["reasons"][0, 1] = 72
    result["episodes"] = [
        {"category": "real", "family": "ST", "channel": 0, "sign": "+", "extreme": False},
        {"category": "real", "family": "ST", "channel": 0, "sign": "+", "extreme": False},
        {"category": "axis", "family": "T", "channel": 1, "sign": "-", "extreme": True},
    ]
    result["quality_spans"] = [
        {"start": 0, "end": 250, "states": ["unknown", "c"]},
        {"start": 250, "end": 500, "states": ["n", "u"]},
        {"start": 500, "end": 1800000, "states": ["c", "c"]},
    ]
    summary = runner.record_descriptives(result, 250)
    assert summary["unknown_reason_bit_cells"] == {
        "1": 1439,
        "2": 1,
        "4": 1,
        "8": 1,
        "16": 0,
        "32": 0,
        "64": 1,
        "128": 0,
    }
    assert summary["episode_class_counts"] == {"real:ST:0:+:False": 2, "axis:T:1:-:True": 1}
    assert summary["raw_quality_seconds_by_channel"] == [
        {"unknown": 1, "clean": 7198, "noisy": 1, "unreadable": 0},
        {"unknown": 0, "clean": 7199, "noisy": 0, "unreadable": 1},
    ]
    assert summary["quality_seconds_are_not_qualified_label_coverage"]


def test_decoder_failure_descriptives_have_only_record_issue_reasons():
    result = runner.finalized_record(decoded([{"kind": "decoder_issue"}]), artificial_result())
    summary = runner.record_descriptives(result, 250)
    assert summary["unknown_reason_bit_cells"]["128"] == 1440
    assert all(value == 0 for key, value in summary["unknown_reason_bit_cells"].items() if key != "128")


def test_full_artificial_run_saves_complete_grids_and_separates_execution_from_engineering(
    tmp_path, monkeypatch
):
    policy = runner.operating_config(ROOT)
    for name in [runner.CONFIG, runner.CONTRACT, runner.EXECUTION, runner.GROUPING]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    linked = [name for group in policy["grouping"]["explicit_groups"] for name in group]
    names = linked + [f"e{number:04d}" for number in range(1000, 1000 + 90 - len(linked))]
    roles = runner.shared.record_roles(("\n".join(names) + "\n").encode(), policy)
    prior = {
        "current_machine_stages": 43,
        "current_machine_cumulative_cpu_seconds": 1.0,
        "current_machine_cumulative_gpu_wall_seconds": 0.0,
    }
    admission = {
        "prior_ledger": {"sha256": "a" * 64},
        "source_sha256": {},
        "package_runtime": {"artificial": True},
        "acquisition_root": "artificial_input",
    }
    admission_path = tmp_path / "artificial_admission.json"
    admission_path.write_text(json.dumps(admission))
    monkeypatch.setattr(runner, "preflight", lambda *args: (policy, admission, prior, roles))
    monkeypatch.setattr(runner, "verified_inputs", lambda *args: (prior, roles))
    monkeypatch.setattr(runner.shared, "verify_sources", lambda *args: None)
    monkeypatch.setattr(runner, "package_identity", lambda *args: admission["package_runtime"])
    calls = []

    def artificial_decode(record, scratch, config):
        calls.append(record.name)
        issues = [{"kind": "artificial_decoder_issue"}] if record.name == names[0] else []
        return {
            "events": [],
            "metadata": {"decoder_issues": issues, "wfdb_effective_view_verified": not issues},
        }

    monkeypatch.setattr(runner.core, "decode_record", artificial_decode)
    import time

    ledger = runner.run(tmp_path, admission_path, (time.perf_counter(), time.process_time()))
    assert ledger["technical_passed"]
    assert ledger["current_machine_stages"] == 44
    assert calls == names
    output = tmp_path / runner.ROOT_OUTPUT
    labels = np.load(output / "labels.npy", allow_pickle=False)
    reasons = np.load(output / "reasons.npy", allow_pickle=False)
    assert labels.shape == reasons.shape == (90, 2, 720)
    assert labels.dtype == np.uint8
    assert reasons.dtype == np.uint16
    assert np.all(labels == 0)
    assert np.all(reasons[0] == 128)
    assert np.all(reasons[1:] == 1)
    summary = json.loads((output / "summary.json").read_text())
    assert not summary["engineering_passed_before_independent_audit"]
    assert summary["engineering_gates"]["complete90records"]
    assert summary["engineering_gates"]["published_grouping79"]
    assert not summary["engineering_gates"]["all_record_contracts"]
    assert summary["record_decode_attempts"] == summary["record_decodes_completed"] == 90
    assert summary["new_annotation_decodes"] == 90
    assert summary["requests"] == summary["new_waveform_decodes"] == summary["new_model_scores"] == 0
    rows = json.loads((output / "record_index.json").read_text())
    assert len(rows) == 90
    assert rows[0]["counts"]["unknown_windows"] == 1440
    assert rows[0]["counts"]["issues"] == 1
