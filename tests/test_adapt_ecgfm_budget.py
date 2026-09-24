"""Bounded adaptation must honor optimizer and example budgets without a GPU."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class AdaptBudgetTest(unittest.TestCase):
    def test_budget_requires_enough_complete_batches(self):
        self.assertEqual(adapt_ecgfm.update_budget(5, 2, 2, 3), (2, True))
        self.assertEqual(adapt_ecgfm.update_budget(5, 2, 2, None), (3, False))
        for settings in [(5, 2, 2, 5), (1, 2, 2, 1), (5, 2, 2, 0)]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                adapt_ecgfm.update_budget(*settings)

    def test_stops_within_epoch_and_counts_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            metadata = root / "checkpoint.json"
            checkpoint.write_bytes(b"checkpoint")
            metadata.write_text(json.dumps({"checkpoint": {"sha256": "fake"}}))
            rows = [{"ecg_id": str(i), "patient_id": str(i), "filename_hr": str(i)}
                    for i in range(5)]
            args = ["adapt_ecgfm", "--manifest-dir", directory, "--raw-dir", directory,
                    "--checkpoint", str(checkpoint), "--checkpoint-metadata", str(metadata),
                    "--output-dir", str(root / "output"), "--device", "cpu", "--workers", "0",
                    "--batch-size", "2", "--epochs", "2", "--max-updates", "3"]

            def sample(_, index):
                value = float(index + 1)
                views = torch.tensor([[[value, 1.0]], [[1.0, value + 1.0]]])
                return views, index

            with mock.patch.object(sys, "argv", args), \
                 mock.patch.object(adapt_ecgfm, "training_rows", return_value=(rows, {})), \
                 mock.patch.object(adapt_ecgfm, "sha256_file", return_value="fake"), \
                 mock.patch.object(adapt_ecgfm, "git_head", return_value="fake"), \
                 mock.patch.object(adapt_ecgfm, "load_model", side_effect=lambda *a: (TinyBackbone(), None)), \
                 mock.patch.object(adapt_ecgfm.TrainingWaveforms, "__getitem__", sample):
                adapt_ecgfm.main()

            output = root / "output"
            history = json.loads((output / "history.json").read_text())
            completion = json.loads((output / "completion.json").read_text())
            self.assertEqual([item["updates"] for item in history], [2, 3])
            self.assertEqual([item["records"] for item in history], [4, 2])
            self.assertEqual([item["complete_epoch"] for item in history], [True, False])
            self.assertEqual(completion["updates"], 3)
            self.assertEqual(completion["seen_examples"], 6)
            self.assertEqual(completion["completed_epochs"], 1)
            self.assertFalse((output / "resume.pt").exists())

            reference = torch.load(output / "adapted_backbone.pt", map_location="cpu",
                                   weights_only=True)["backbone"]
            interrupted_args = args.copy()
            interrupted_args[interrupted_args.index("--output-dir") + 1] = str(root / "interrupted")
            with mock.patch.object(sys, "argv", interrupted_args), \
                 mock.patch.object(adapt_ecgfm, "training_rows", return_value=(rows, {})), \
                 mock.patch.object(adapt_ecgfm, "sha256_file", return_value="fake"), \
                 mock.patch.object(adapt_ecgfm, "git_head", return_value="fake"), \
                 mock.patch.object(adapt_ecgfm, "load_model", side_effect=lambda *a: (InterruptingBackbone(), None)), \
                 mock.patch.object(adapt_ecgfm.TrainingWaveforms, "__getitem__", sample):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    adapt_ecgfm.main()
            self.assertTrue((root / "interrupted" / "resume.pt").is_file())

            with mock.patch.object(sys, "argv", interrupted_args + ["--resume"]), \
                 mock.patch.object(adapt_ecgfm, "training_rows", return_value=(rows, {})), \
                 mock.patch.object(adapt_ecgfm, "sha256_file", return_value="fake"), \
                 mock.patch.object(adapt_ecgfm, "git_head", return_value="fake"), \
                 mock.patch.object(adapt_ecgfm, "load_model", side_effect=lambda *a: (TinyBackbone(), None)), \
                 mock.patch.object(adapt_ecgfm.TrainingWaveforms, "__getitem__", sample):
                adapt_ecgfm.main()
            resumed = torch.load(root / "interrupted" / "adapted_backbone.pt",
                                 map_location="cpu", weights_only=True)["backbone"]
            for name in reference:
                self.assertTrue(torch.equal(reference[name], resumed[name]), name)


if __name__ == "__main__":
    unittest.main()
