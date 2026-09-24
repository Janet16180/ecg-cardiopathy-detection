"""Exercise xECG budget routing and exact epoch-boundary continuation on CPU."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.run_xecg_finetune import (
    budgets, cpu_state, layerwise_parameter_groups, load_resume, make_scheduler,
    resume_for_budget,
    save_resume, train_epoch,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.patch_embedding = nn.Linear(3, 3)
        self.backbone.core = nn.Module()
        self.backbone.core.model = nn.Module()
        self.backbone.core.model.blocks = nn.ModuleList(nn.Linear(3, 3) for _ in range(9))
        self.backbone.norm = nn.LayerNorm(3)
        self.head = nn.Linear(3, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.backbone.patch_embedding(x)
        for block in self.backbone.core.model.blocks:
            x = torch.tanh(block(x))
        return self.head(self.dropout(self.backbone.norm(x))).squeeze(-1)


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = make_scheduler(optimizer, steps_per_epoch=2, epochs=4)
    generator = torch.Generator().manual_seed(seed)
    x = torch.arange(45, dtype=torch.float32).reshape(15, 3) / 10
    y = torch.tensor([0, 1] * 7 + [1], dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=True, generator=generator)
    return model, optimizer, scheduler, generator, loader


class XECGFineTuneTest(unittest.TestCase):
    def test_mixed_resume_and_fresh_budget_routing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "xecg_full_seed42"
            second = root / "xecg_ten_percent_seed42"
            first.mkdir()
            (first / "resume.pt").write_bytes(b"epoch boundary")
            self.assertTrue(resume_for_budget(first, requested=True))
            self.assertFalse(resume_for_budget(second, requested=True))
            with self.assertRaisesRegex(FileExistsError, "use --resume"):
                resume_for_budget(first, requested=False)
            second.mkdir()
            (second / "history.json").write_text("[]")
            with self.assertRaisesRegex(FileExistsError, "lacks an epoch checkpoint"):
                resume_for_budget(second, requested=True)

    def test_budget_routing_and_layerwise_optimizer_groups(self):
        self.assertEqual(budgets("all"), ("full", "ten_percent"))
        self.assertEqual(budgets("full"), ("full",))
        self.assertEqual(budgets("ten_percent"), ("ten_percent",))
        with self.assertRaises(ValueError):
            budgets("unknown")
        groups = layerwise_parameter_groups(TinyClassifier())
        self.assertEqual(groups[0]["name"], "patch_embedding")
        self.assertAlmostEqual(groups[0]["lr"], 3e-5 * .75 ** 10)
        self.assertAlmostEqual(groups[1]["lr"], 3e-5 * .75 ** 9)
        self.assertAlmostEqual(groups[9]["lr"], 3e-5 * .75)
        self.assertEqual(groups[-1]["name"], "binary_head")
        self.assertAlmostEqual(groups[-1]["lr"], 1e-3)
        ids = [id(param) for group in groups for param in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_epoch_resume_preserves_parameters_optimizer_scheduler_and_rng(self):
        fingerprint = {"checkpoint_sha256": "synthetic", "budget": "full", "epochs": 4}
        uninterrupted = setup(17)
        for _ in range(4):
            train_epoch(uninterrupted[0], uninterrupted[4], uninterrupted[1],
                        uninterrupted[2], "cpu", effective_batch_size=9)
        expected = cpu_state(uninterrupted[0])
        expected_draws = (random.random(), float(np.random.random()), torch.rand(1))

        interrupted = setup(17)
        for _ in range(2):
            train_epoch(interrupted[0], interrupted[4], interrupted[1],
                        interrupted[2], "cpu", effective_batch_size=9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            save_resume(path, fingerprint, interrupted[0], interrupted[1], interrupted[2],
                        interrupted[3], cpu_state(interrupted[0]), 2, .8,
                        [{"epoch": 1}, {"epoch": 2}], 12.0)
            self.assertFalse(path.with_name("resume.pt.partial").exists())
            resumed = setup(999)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                load_resume(path, {"checkpoint_sha256": "changed"}, *resumed[:4])
            saved = load_resume(path, fingerprint, *resumed[:4])
            self.assertEqual(saved["elapsed_seconds"], 12.0)
            for _ in range(2):
                train_epoch(resumed[0], resumed[4], resumed[1], resumed[2],
                            "cpu", effective_batch_size=9)
            for name, parameter in expected.items():
                torch.testing.assert_close(resumed[0].state_dict()[name], parameter, rtol=0, atol=0)
            self.assertEqual(random.random(), expected_draws[0])
            self.assertEqual(float(np.random.random()), expected_draws[1])
            torch.testing.assert_close(torch.rand(1), expected_draws[2], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
