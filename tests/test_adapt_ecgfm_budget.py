"""Bounded adaptation must honor optimizer and example budgets without a GPU."""

import json
import sys

import pytest
import torch

from scripts.experiments import adapt_ecgfm


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def extract_features(self, source, **kwargs):
        # Depend on the parameter and each ECG so the contrastive loss has gradients.
        return {"x": source * self.scale}


class InterruptingBackbone(TinyBackbone):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def extract_features(self, source, **kwargs):
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("simulated interruption")
        return super().extract_features(source, **kwargs)


def test_budget_requires_enough_complete_batches():
    assert adapt_ecgfm.update_budget(5, 2, 2, 3) == (2, True)
    assert adapt_ecgfm.update_budget(5, 2, 2, None) == (3, False)


@pytest.mark.parametrize("settings", [(5, 2, 2, 5), (1, 2, 2, 1), (5, 2, 2, 0)])
def test_budget_rejects_unreachable_updates(settings):
    with pytest.raises(ValueError, match="max-updates"):
        adapt_ecgfm.update_budget(*settings)


def test_contrastive_loss_is_invariant_to_record_order():
    features = torch.randn(4, 2, 3, generator=torch.Generator().manual_seed(1))
    patients = torch.tensor([0, 1, 1, 2])
    order = torch.tensor([2, 0, 3, 1])
    torch.testing.assert_close(adapt_ecgfm.temporal_contrastive_loss(features, patients),
                               adapt_ecgfm.temporal_contrastive_loss(features[order], patients[order]))
    with pytest.raises(ValueError, match="at least two ECGs"):
        adapt_ecgfm.temporal_contrastive_loss(features[:1], patients[:1])


@pytest.fixture
def mocked_adaptation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    metadata = tmp_path / "checkpoint.json"
    checkpoint.write_bytes(b"checkpoint")
    metadata.write_text(json.dumps({"checkpoint": {"sha256": "fake"}}))
    rows = [{"ecg_id": str(i), "patient_id": str(i), "filename_hr": str(i)} for i in range(5)]

    def sample(_, index):
        value = float(index + 1)
        views = torch.tensor([[[value, 1.0]], [[1.0, value + 1.0]]])
        return views, index

    monkeypatch.setattr(adapt_ecgfm, "training_rows", lambda *args: (rows, {}))
    monkeypatch.setattr(adapt_ecgfm, "sha256_file", lambda path: "fake")
    monkeypatch.setattr(adapt_ecgfm, "git_head", lambda path: "fake")
    monkeypatch.setattr(adapt_ecgfm.TrainingWaveforms, "__getitem__", sample)

    def run(output, backbone, *extra):
        monkeypatch.setattr(adapt_ecgfm, "load_model", lambda *args: (backbone(), None))
        monkeypatch.setattr(sys, "argv", [
            "adapt_ecgfm", "--manifest-dir", str(tmp_path), "--raw-dir", str(tmp_path),
            "--checkpoint", str(checkpoint), "--checkpoint-metadata", str(metadata),
            "--output-dir", str(output), "--device", "cpu", "--workers", "0",
            "--batch-size", "2", "--epochs", "2", "--max-updates", "3", *extra])
        adapt_ecgfm.main()

    return run


def test_stops_within_epoch_and_counts_examples(tmp_path, mocked_adaptation):
    output = tmp_path / "output"
    mocked_adaptation(output, TinyBackbone)
    history = json.loads((output / "history.json").read_text())
    completion = json.loads((output / "completion.json").read_text())
    assert [item["updates"] for item in history] == [2, 3]
    assert [item["records"] for item in history] == [4, 2]
    assert [item["complete_epoch"] for item in history] == [True, False]
    assert completion["updates"] == 3
    assert completion["seen_examples"] == 6
    assert completion["completed_epochs"] == 1
    assert not (output / "resume.pt").exists()


def test_resume_after_interruption_matches_uninterrupted_run(tmp_path, mocked_adaptation):
    mocked_adaptation(tmp_path / "output", TinyBackbone)
    reference = torch.load(tmp_path / "output" / "adapted_backbone.pt", map_location="cpu",
                           weights_only=True)["backbone"]
    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        mocked_adaptation(interrupted, InterruptingBackbone)
    assert (interrupted / "resume.pt").is_file()
    mocked_adaptation(interrupted, TinyBackbone, "--resume")
    resumed = torch.load(interrupted / "adapted_backbone.pt", map_location="cpu",
                         weights_only=True)["backbone"]
    for name in reference:
        assert torch.equal(reference[name], resumed[name]), name


def test_fresh_run_refuses_nonempty_output(tmp_path, mocked_adaptation):
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.txt").write_text("x")
    with pytest.raises(FileExistsError, match="not empty"):
        mocked_adaptation(output, TinyBackbone)
