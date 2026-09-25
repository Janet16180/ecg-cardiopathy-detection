"""Exercise xECG budget routing and exact epoch-boundary continuation on CPU."""

import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_experiment.receipts import artifact_hashes, verified_completion
from ecg_experiment.reproducibility import cpu_state
from scripts.experiments.run_xecg_finetune import (
    COMPLETION_FILES,
    budgets,
    check_cache_coverage,
    layerwise_parameter_groups,
    load_resume,
    make_scheduler,
    resume_for_budget,
    save_resume,
    train_epoch,
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


def test_mixed_resume_and_fresh_budget_routing(tmp_path):
    first = tmp_path / "xecg_full_seed42"
    second = tmp_path / "xecg_ten_percent_seed42"
    first.mkdir()
    (first / "resume.pt").write_bytes(b"epoch boundary")
    assert resume_for_budget(first, requested=True)
    assert not resume_for_budget(second, requested=True)
    with pytest.raises(FileExistsError, match="use --resume"):
        resume_for_budget(first, requested=False)
    second.mkdir()
    (second / "history.json").write_text("[]")
    with pytest.raises(FileExistsError, match="lacks an epoch checkpoint"):
        resume_for_budget(second, requested=True)


def test_budget_routing():
    assert budgets("all") == ("full", "ten_percent")
    assert budgets("full") == ("full",)
    assert budgets("ten_percent") == ("ten_percent",)
    with pytest.raises(ValueError, match="Unknown label budget"):
        budgets("unknown")


def test_layerwise_optimizer_groups_cover_each_parameter_once():
    groups = layerwise_parameter_groups(TinyClassifier())
    assert groups[0]["name"] == "patch_embedding"
    assert groups[0]["lr"] == pytest.approx(3e-5 * .75 ** 10)
    assert groups[1]["lr"] == pytest.approx(3e-5 * .75 ** 9)
    assert groups[9]["lr"] == pytest.approx(3e-5 * .75)
    assert groups[-1]["name"] == "binary_head"
    assert groups[-1]["lr"] == pytest.approx(1e-3)
    ids = [id(param) for group in groups for param in group["params"]]
    assert len(ids) == len(set(ids))


def test_layerwise_groups_reject_another_block_count():
    model = TinyClassifier()
    model.backbone.core.model.blocks = nn.ModuleList(nn.Linear(3, 3) for _ in range(8))
    with pytest.raises(ValueError, match="nine xECG backbone blocks"):
        layerwise_parameter_groups(model)


def test_scheduler_warms_up_then_decays_to_zero():
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = make_scheduler(optimizer, steps_per_epoch=2, epochs=3)
    rates = []
    for _ in range(6):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[:3] == pytest.approx([0.0, 0.5, 1.0])
    assert rates[3] == pytest.approx(0.5 * (1 + np.cos(np.pi / 4)))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_epoch_resume_preserves_parameters_optimizer_scheduler_and_rng(tmp_path):
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
    path = tmp_path / "resume.pt"
    save_resume(path, fingerprint, interrupted[0], interrupted[1], interrupted[2],
                interrupted[3], cpu_state(interrupted[0]), 2, .8,
                [{"epoch": 1}, {"epoch": 2}], 12.0)
    assert [item.name for item in tmp_path.iterdir()] == ["resume.pt"]
    resumed = setup(999)
    with pytest.raises(ValueError, match="configuration differs"):
        load_resume(path, {"checkpoint_sha256": "changed"}, *resumed[:4])
    saved = load_resume(path, fingerprint, *resumed[:4])
    assert saved["elapsed_seconds"] == 12.0
    for _ in range(2):
        train_epoch(resumed[0], resumed[4], resumed[1], resumed[2],
                    "cpu", effective_batch_size=9)
    for name, parameter in expected.items():
        torch.testing.assert_close(resumed[0].state_dict()[name], parameter, rtol=0, atol=0)
    assert random.random() == expected_draws[0]
    assert float(np.random.random()) == expected_draws[1]
    torch.testing.assert_close(torch.rand(1), expected_draws[2], rtol=0, atol=0)


def test_resume_rejects_malformed_history(tmp_path):
    model, optimizer, scheduler, generator, _ = setup(3)
    path = tmp_path / "resume.pt"
    save_resume(path, {}, model, optimizer, scheduler, generator, cpu_state(model), 3, .5,
                [{"epoch": 1}, {"epoch": 2}], 1.0)
    with pytest.raises(ValueError, match="Malformed"):
        load_resume(path, {}, model, optimizer, scheduler, generator)


def test_train_epoch_counts_one_update_per_effective_batch():
    model, optimizer, scheduler, _, loader = setup(5)
    loss, updates = train_epoch(model, loader, optimizer, scheduler, "cpu", effective_batch_size=6)
    assert updates == 3
    assert np.isfinite(loss)


def test_completion_is_verified_against_fingerprint_and_artifacts(tmp_path):
    fingerprint = {"budget": "full"}
    assert verified_completion(tmp_path, fingerprint, COMPLETION_FILES) is None
    for name in COMPLETION_FILES:
        (tmp_path / name).write_text(name)
    receipt = {"fingerprint": fingerprint, "sha256": artifact_hashes(tmp_path, COMPLETION_FILES)}
    (tmp_path / "complete.json").write_text(json.dumps(receipt))
    assert verified_completion(tmp_path, fingerprint, COMPLETION_FILES) == receipt
    with pytest.raises(ValueError, match="differs from requested inputs"):
        verified_completion(tmp_path, {"budget": "ten_percent"}, COMPLETION_FILES)
    (tmp_path / "metrics.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verified_completion(tmp_path, fingerprint, COMPLETION_FILES)


def test_cache_coverage_requires_every_selected_record():
    rows = {"labeled_train": [{"ecg_id": "1"}], "test": [{"ecg_id": "4"}]}
    manifests = {"full": (rows, [{"ecg_id": "2"}], [{"ecg_id": "3"}], {})}
    check_cache_coverage(manifests, {1: 0, 2: 1, 3: 2, 4: 3})
    with pytest.raises(ValueError, match="missing ECG 4"):
        check_cache_coverage(manifests, {1: 0, 2: 1, 3: 2})
