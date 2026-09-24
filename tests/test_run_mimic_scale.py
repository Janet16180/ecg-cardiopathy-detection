"""Orchestrator guards against incomplete outputs and mismatched update budgets."""

import json
from unittest import mock

import pytest

from scripts.experiments import run_mimic_scale
from scripts.experiments.finetune_pretrained import check_adaptation_budget
from scripts.experiments.run_mimic_scale import ensure_stage, validate_comparison, wait_for_artifact


def test_bounded_checkpoint_is_accepted_only_at_exact_budget():
    bounded = {"max_updates": 7000, "batch_size": 32, "epochs": 14}
    check_adaptation_budget(bounded, [{"updates": 7000, "seen_examples": 224000}])
    check_adaptation_budget({"epochs": 2}, [{}, {}])
    with pytest.raises(ValueError, match="fixed epoch budget"):
        check_adaptation_budget({"epochs": 2}, [{}])


@pytest.mark.parametrize("history", [[], [{"updates": 6999, "seen_examples": 223968}],
                                     [{"updates": 7000, "seen_examples": 223968}]])
def test_bounded_checkpoint_rejects_incomplete_budget(history):
    bounded = {"max_updates": 7000, "batch_size": 32, "epochs": 14}
    with pytest.raises(ValueError, match="exact optimizer-update budget"):
        check_adaptation_budget(bounded, history)


def test_existing_incomplete_stage_is_not_relaunched(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "history.json").write_text("[]")
    launch = mock.Mock()
    monkeypatch.setattr(run_mimic_scale, "run_stage", launch)
    with pytest.raises(RuntimeError, match="Incomplete existing"):
        ensure_stage(tmp_path, "ptb_adaptation", stage, ["python"], lambda: {}, resumable=True)
    launch.assert_not_called()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["stages"]["ptb_adaptation"]["state"] == "failed"


def test_completed_stage_is_reused(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "metrics.json").write_text("{}")
    launch = mock.Mock()
    monkeypatch.setattr(run_mimic_scale, "run_stage", launch)
    assert ensure_stage(tmp_path, "direct", stage, ["python"], lambda: {"auroc": 0.8}) == {"auroc": 0.8}
    launch.assert_not_called()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["stages"]["direct"]["skipped_existing"] is True


def test_reused_wait_pid_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mimic_scale, "process_cmdline", lambda pid: "python another_task.py")
    with pytest.raises(RuntimeError, match="no longer identifies"):
        wait_for_artifact(4607, "finetune_pretrained", tmp_path / "metrics.json", lambda: None)


def test_wait_requires_artifact_without_pid(tmp_path):
    with pytest.raises(FileNotFoundError, match="supply its active wait PID"):
        wait_for_artifact(None, "finetune_pretrained", tmp_path / "metrics.json", lambda: None)


def test_wait_polls_until_producer_exits_with_valid_artifact(tmp_path, monkeypatch):
    artifact = tmp_path / "metrics.json"
    commands = iter(["python -m finetune_pretrained"] * 2 + [None])

    def sleep(_):
        artifact.write_text("{}")

    monkeypatch.setattr(run_mimic_scale, "process_cmdline", lambda pid: next(commands))
    monkeypatch.setattr(run_mimic_scale.time, "sleep", sleep)
    validate = mock.Mock(side_effect=[ValueError("still writing"), None])
    wait_for_artifact(4607, "finetune_pretrained", artifact, validate)
    assert validate.call_count == 2


def test_wait_rejects_exit_without_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mimic_scale, "process_cmdline", lambda pid: None)
    with pytest.raises(RuntimeError, match="ended without required artifact"):
        wait_for_artifact(4607, "finetune_pretrained", tmp_path / "metrics.json", lambda: None)


def test_finetune_stage_resumes_from_epoch_checkpoint(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "resume.pt").write_bytes(b"checkpoint")
    (stage / "history.json").write_text("[]")
    launch = mock.Mock(return_value={"ok": True})
    monkeypatch.setattr(run_mimic_scale, "run_stage", launch)
    result = ensure_stage(tmp_path, "ptb_finetune", stage, ["python"], lambda: {},
                          resumable=True, completion_marker="metrics.json")
    assert result == {"ok": True}
    assert launch.call_args.args[2] == ["python", "--resume"]


def test_partial_finetune_metrics_do_not_block_resume(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "resume.pt").write_bytes(b"checkpoint")
    (stage / "metrics.json").write_text("{}")
    launch = mock.Mock(return_value={})
    monkeypatch.setattr(run_mimic_scale, "run_stage", launch)
    ensure_stage(tmp_path, "ptb_finetune", stage, ["python"], mock.Mock(side_effect=FileNotFoundError),
                 resumable=True, completion_marker="metrics.json")
    assert launch.call_args.args[2] == ["python", "--resume"]


def test_paired_comparison_must_match_reference_and_models(tmp_path):
    path = tmp_path / "comparison.json"
    items = [{"model": "a", "reference": "base"}, {"model": "b", "reference": "base"}]
    path.write_text(json.dumps({"test_records": 1896, "comparisons": items}))
    assert validate_comparison(path, "base", ("a", "b")) == {"comparisons": 2, "test_records": 1896}
    with pytest.raises(ValueError, match="Invalid paired comparison"):
        validate_comparison(path, "other", ("a", "b"))
    with pytest.raises(ValueError, match="Invalid paired comparison"):
        validate_comparison(path, "base", ("a",))
