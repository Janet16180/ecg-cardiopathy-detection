"""Epoch-boundary fine-tuning checkpoints preserve optimizer and RNG state."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ecg_experiment.reproducibility import cpu_state
from scripts.experiments.finetune_pretrained import load_resume_checkpoint, save_resume_checkpoint


class FineTuneResumeTest(unittest.TestCase):
    def test_interrupted_training_matches_uninterrupted_training(self):
        def new_training():
            model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(8, 1))
            return model, torch.optim.AdamW(model.parameters(), lr=0.01)

        def train_epoch(model, optimizer):
            model.train()
            inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
            for index in torch.randperm(8).split(4):
                scale = random.random() + float(np.random.random())
                prediction = model(inputs[index] * scale)
                loss = prediction.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            fingerprint = {"manifest_sha256": {"labeled_train.csv": "test"}, "epochs": 4}
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            uninterrupted, uninterrupted_optimizer = new_training()
            for _ in range(4):
                train_epoch(uninterrupted, uninterrupted_optimizer)
            expected = cpu_state(uninterrupted)
            expected_draws = (random.random(), float(np.random.random()), torch.rand(1))

            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            interrupted, interrupted_optimizer = new_training()
            for _ in range(2):
                train_epoch(interrupted, interrupted_optimizer)
            save_resume_checkpoint(path, fingerprint, interrupted, interrupted_optimizer,
                                   cpu_state(interrupted), 2, 0.7,
                                   [{"epoch": 1}, {"epoch": 2}], 12.0)
            self.assertFalse(path.with_name("resume.pt.partial").exists())

            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            resumed, resumed_optimizer = new_training()
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                load_resume_checkpoint(path, {"epochs": 5}, resumed, resumed_optimizer)
            checkpoint = load_resume_checkpoint(path, fingerprint, resumed, resumed_optimizer)
            self.assertEqual(checkpoint["elapsed_seconds"], 12.0)
            for _ in range(2):
                train_epoch(resumed, resumed_optimizer)
            for name, parameter in expected.items():
                torch.testing.assert_close(resumed.state_dict()[name], parameter, rtol=0, atol=0)
            self.assertEqual(random.random(), expected_draws[0])
            self.assertEqual(float(np.random.random()), expected_draws[1])
            torch.testing.assert_close(torch.rand(1), expected_draws[2], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
