"""CPU-only real-data loader/backpropagation check; not a model experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from ecg_experiment.training_dataset import TrainingECGDataset, file_hash, read_csv

HELDOUT_SPLITS = ("development", "calibration", "test")


def require(condition: bool, message: str) -> None:
    """
    Fail the smoke check with a clear message.

    Parameters
    ----------
    condition : bool
        Check that must hold.
    message : str
        Description of the failed check.

    Raises
    ------
    RuntimeError
        If ``condition`` is false.
    """
    if not condition:
        raise RuntimeError(f"Smoke check failed: {message}")


def check_patient_separation(ssl_rows: list[dict[str, str]], references: list[dict[str, str]]) -> None:
    """
    Check that PTB-XL training and held-out patient sets are disjoint.

    Parameters
    ----------
    ssl_rows : list[dict[str, str]]
        Self-supervised training rows.
    references : list[dict[str, str]]
        Held-out reference rows with ``split`` and ``patient_id``.

    Raises
    ------
    RuntimeError
        If any patient appears in more than one of the sets.
    """
    train_patients = {row["patient_id"] for row in ssl_rows if row["source"] == "ptbxl"}
    heldout = {split: {row["patient_id"] for row in references if row["split"] == split}
               for split in HELDOUT_SPLITS}
    for split, patients in heldout.items():
        require(not train_patients & patients, f"training patients overlap {split}")
    for index, split in enumerate(HELDOUT_SPLITS):
        for other in HELDOUT_SPLITS[index + 1:]:
            require(not heldout[split] & heldout[other], f"{split} patients overlap {other}")


def check_backward_step(signal: torch.Tensor, target: torch.Tensor) -> None:
    """
    Check that a tiny model gets finite gradients and an optimizer update.

    Parameters
    ----------
    signal : torch.Tensor
        Batch of waveforms shaped ``[batch, 12, samples]``.
    target : torch.Tensor
        Binary targets for the batch.

    Raises
    ------
    RuntimeError
        If the loss or gradients are not finite or no parameter changes.
    """
    model = torch.nn.Sequential(torch.nn.Conv1d(12, 4, 25, stride=25), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(4, 1))
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(signal).squeeze(1), target.float())
    loss.backward()
    require(bool(torch.isfinite(loss)), "loss is not finite")
    require(all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()), "gradients are missing or not finite")
    optimizer.step()
    require(any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)),
            "optimizer step changed no parameter")


def smoke(directory: Path) -> dict[str, Any]:
    """
    Load real batches, run one backward step, and check patient separation.

    Parameters
    ----------
    directory : Path
        Built training dataset directory.

    Returns
    -------
    dict[str, Any]
        Receipt describing the checks that passed.

    Raises
    ------
    RuntimeError
        If any check fails.
    """
    directory = Path(directory)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    ssl = TrainingECGDataset(directory)
    representatives: dict[str, int] = {}
    for index, row in enumerate(ssl.rows):
        representatives.setdefault(row["source"], index)
    batch = next(iter(DataLoader(Subset(ssl, list(representatives.values())), batch_size=len(representatives))))
    require(tuple(batch["signal"].shape) == (len(representatives), 12, 5000), "unexpected SSL batch shape")
    require(not batch["target_available"].any() and bool(torch.all(batch["target"] == -1)),
            "SSL targets are not masked")

    labeled = TrainingECGDataset(directory, purpose="supervised", label_budget="0.1")
    batch = next(iter(DataLoader(labeled, batch_size=8)))
    require(bool(batch["target_available"].all()) and set(batch["source"]) == {"ptbxl"},
            "supervised batch lacks PTB-XL targets")
    check_backward_step(batch["signal"], batch["target"])

    references = read_csv(directory / "heldout_references.csv")
    check_patient_separation(ssl.rows, references)
    return {"status": "passed_cpu_loader_smoke_not_performance_result", "device": "cpu",
            "dataset_metadata_sha256": file_hash(directory / "metadata.json"),
            "smoke_source_sha256": file_hash(Path(__file__)), "ssl_sources_checked": list(representatives),
            "ssl_records": len(ssl), "ssl_targets_masked": True, "supervised_fraction0.1_records": len(labeled),
            "supervised_fraction1_records": len(TrainingECGDataset(directory, "supervised", "1")),
            "real_batch_shape": list(batch["signal"].shape), "finite_backward_and_optimizer_step": True,
            "ptb_train_development_calibration_test_patients_disjoint": True,
            "heldout_reference_records": len(references)}


def main() -> None:
    """Run the smoke check from the command line and print its receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(smoke(args.dataset_dir), indent=2))


if __name__ == "__main__":
    main()
