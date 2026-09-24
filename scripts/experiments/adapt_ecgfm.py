#!/usr/bin/env python3
"""Continue ECG-FM with a bounded patient-aware temporal contrastive objective.

This CMSC-style adaptation is not a reproduction of ECG-FM's full WCR objective.
Only official PTB-XL folds 1--8 enter optimization; diagnostic annotations are
never returned by the streaming dataset. The final fixed-epoch state is used
unless --max-updates requests an exact optimizer-update budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from scripts.extract_pretrained import git_head, load_model, preprocess_ecg_fm, read_record, sha256


def training_rows(manifest_dir: Path, raw_dir: Path):
    """Check exact official training membership and patient isolation without labels."""
    path = manifest_dir / "all_train_ssl.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"ecg_id", "patient_id", "filename_lr", "filename_hr"}:
            raise ValueError("SSL manifest must contain waveform identifiers only")
        rows = list(reader)
    ids = [row["ecg_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicate SSL ECG identifiers")
    metadata_path = raw_dir / "ptbxl_database.csv"
    with metadata_path.open(newline="") as handle:
        # Retain identity/fold fields only; SCP statements/reports are not inspected.
        official = {r["ecg_id"]: {k: r[k] for k in
                    ("patient_id", "strat_fold", "filename_hr")} for r in csv.DictReader(handle)}
    expected_ids = {key for key, row in official.items() if 1 <= int(row["strat_fold"]) <= 8}
    if set(ids) != expected_ids:
        raise ValueError("SSL manifest must contain exactly all official fold 1--8 ECGs")
    for row in rows:
        source = official[row["ecg_id"]]
        if any(row[key] != source[key] for key in ("patient_id", "filename_hr")):
            raise ValueError("SSL patient identity or waveform differs from official metadata")
    heldout_patients = {r["patient_id"] for r in official.values() if int(r["strat_fold"]) >= 9}
    if heldout_patients & {r["patient_id"] for r in rows}:
        raise ValueError("Held-out patient present in SSL data")
    hashes = {path.name: sha256(path), "ptbxl_database.csv": sha256(metadata_path)}
    return rows, hashes


def extra_training_rows(path: Path | None, existing_ids):
    """Read an explicitly unlabeled, namespaced external waveform manifest."""
    if path is None:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ecg_id", "patient_id", "raw_dir", "filename_hr", "source"}
        if set(reader.fieldnames or []) != required:
            raise ValueError("External SSL manifest must contain only " + ", ".join(sorted(required)))
        rows = list(reader)
    seen = set(existing_ids)
    for row in rows:
        if not all(row.values()):
            raise ValueError("Missing external waveform identity or path")
        prefix = row["source"] + ":"
        if not row["ecg_id"].startswith(prefix) or not row["patient_id"].startswith(prefix):
            raise ValueError("External ECG and grouping identifiers must use a source: namespace")
        if row["ecg_id"] in seen:
            raise ValueError("Duplicate external ECG identifier")
        seen.add(row["ecg_id"])
        root = Path(row["raw_dir"]).resolve()
        waveform = (root / row["filename_hr"]).resolve()
        if not waveform.is_relative_to(root) or not waveform.with_suffix(".hea").is_file():
            raise ValueError("External waveform is absent or escapes its dataset root")
        row["raw_dir"] = str(root)
    if not rows:
        raise ValueError("External manifest is empty")
    return rows


class TrainingWaveforms(Dataset):
    def __init__(self, rows, raw_dir):
        self.rows, self.raw_dir = rows, raw_dir
        patients = {value: i for i, value in enumerate(sorted({r["patient_id"] for r in rows}))}
        self.patient_index = [patients[r["patient_id"]] for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        waveform = read_record(Path(row.get("raw_dir", self.raw_dir)), row["filename_hr"])
        return torch.from_numpy(preprocess_ecg_fm(waveform)), self.patient_index[index]


def temporal_contrastive_loss(features, patient_ids, temperature=0.1):
    """Symmetric cross-view SupCon; all same-patient cross-view pairs are positive.

    Each half-record anchors against every opposite half-record in the batch.
    Repeated ECGs from one patient are never treated as negative pairs. We average
    log probabilities over a patient's positives, then over both anchor directions.
    """
    if features.ndim != 3 or features.shape[1] != 2 or len(features) < 2:
        raise ValueError("Need at least two ECGs, each with two feature vectors")
    if not temperature > 0 or len(patient_ids) != len(features):
        raise ValueError("Invalid temperature or patient identifiers")
    embedding = F.normalize(features.float(), dim=-1)
    similarities = embedding[:, 0] @ embedding[:, 1].T / temperature
    positives = patient_ids[:, None].eq(patient_ids[None, :]).to(similarities.dtype)
    forward = -(F.log_softmax(similarities, dim=1) * positives).sum(1) / positives.sum(1)
    backward = -(F.log_softmax(similarities.T, dim=1) * positives.T).sum(1) / positives.T.sum(1)
    return (forward.mean() + backward.mean()) / 2


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def update_budget(records: int, batch_size: int, epochs: int, max_updates: int | None):
    """Return steps per epoch and whether the final batch must be omitted.

    Bounded runs use full batches so every update sees the same number of ECGs.
    Legacy epoch runs retain their original partial final batch behavior.
    """
    if max_updates is None:
        return math.ceil(records / batch_size), False
    if max_updates < 1:
        raise ValueError("--max-updates must be positive")
    steps_per_epoch = records // batch_size
    if steps_per_epoch < 1:
        raise ValueError("--max-updates requires at least one full batch per epoch")
    if max_updates > epochs * steps_per_epoch:
        raise ValueError("--epochs cannot supply --max-updates full-batch optimizer updates")
    return steps_per_epoch, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True,
                        help="Existing extraction metadata identifying the official checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-updates", type=int,
                        help="Stop after exactly this many optimizer updates, even within an epoch; "
                             "uses full batches only and resumes from completed epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extra-ssl-manifest", type=Path,
                        help="Unlabeled external CSV: ecg_id,patient_id,raw_dir,filename_hr,source")
    parser.add_argument("--extra-grouping-description", default="Source-specific grouping; patient identity not independently verified")
    parser.add_argument("--protocol-note", default="")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error("Epochs/threads must be positive; batch >=2 and workers >=0")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or not np.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Learning rate and temperature must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptbxl_rows, hashes = training_rows(args.manifest_dir, args.raw_dir)
    extra_rows = extra_training_rows(args.extra_ssl_manifest, [r["ecg_id"] for r in ptbxl_rows])
    rows = ptbxl_rows + extra_rows
    try:
        steps_per_epoch, drop_last = update_budget(
            len(rows), args.batch_size, args.epochs, args.max_updates)
    except ValueError as exc:
        parser.error(str(exc))
    original_meta = json.loads(args.checkpoint_metadata.read_text())["checkpoint"]
    if original_meta["sha256"] != sha256(args.checkpoint):
        raise ValueError("Official checkpoint differs from recorded extraction metadata")
    config = {
        "model": "ecg-fm", "objective": "CMSC-style symmetric cross-view patient-aware contrastive adaptation; not full WCR",
        "official_checkpoint": original_meta, "manifest_sha256": hashes,
        "training_ecg_ids": [r["ecg_id"] for r in ptbxl_rows],
        "training_patient_ids": sorted({r["patient_id"] for r in rows}),
        "training_records": len(rows), "heldout_patient_overlap": None if extra_rows else 0,
        "ptbxl_heldout_patient_overlap": 0,
        "ptbxl_training_records": len(ptbxl_rows),
        "external_ssl": ({"manifest_sha256": sha256(args.extra_ssl_manifest),
                          "records": len(extra_rows), "source_counts": dict(Counter(r["source"] for r in extra_rows)),
                          "grouping_description": args.extra_grouping_description,
                          "waveforms": extra_rows,
                          "limitation": "PTB-XL patient isolation is verified; cross-source patient overlap is not independently verifiable"}
                         if extra_rows else None),
        "preprocessing": "Official 500 Hz 12-lead per-lead z-score over 10 seconds; two nonoverlapping 5-second views",
        "representation": "Mean final-layer tokens per view, L2 normalized for loss; no projection head",
        "positive_pairs": "All same-patient pairs across the two temporal views; averaged positive log probabilities",
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "temperature": args.temperature, "weight_decay": 0.01, "gradient_clip": 1.0,
        "seed": args.seed, "workers": args.workers, "threads": args.threads,
        "checkpoint_selection": "Final state after fixed epoch budget; no validation/test selection",
        "protocol_note": args.protocol_note,
        "source_sha256": sha256(Path(__file__)),
        "extractor_source_sha256": sha256(Path(__file__).resolve().parents[2] / "scripts/extract_pretrained.py"),
        "official_source_commit": git_head(Path(__file__).resolve().parents[2] / "third_party" / "fairseq-signals"),
    }
    if args.max_updates is not None:
        config.update({"max_updates": args.max_updates, "updates_per_epoch": steps_per_epoch,
                       "drop_last": True,
                       "checkpoint_selection": "Final state after exact optimizer-update budget; no validation/test selection"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.output_dir / "resume.pt"
    if (args.output_dir / "adapted_backbone.pt").exists():
        raise FileExistsError("Completed adaptation exists")
    if any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError("Output directory is not empty; use --resume for identical settings")
    if args.resume and (json.loads((args.output_dir / "config.json").read_text()) != config):
        raise ValueError("Resume configuration differs")
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError("No completed-epoch resume checkpoint exists; restart this run in a clean output directory")
    save_json(args.output_dir / "config.json", config)
    (args.output_dir / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    model, _ = load_model("ecg-fm", args.checkpoint, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TrainingWaveforms(rows, args.raw_dir), batch_size=args.batch_size,
                        shuffle=True, drop_last=drop_last, num_workers=args.workers,
                        pin_memory=args.device == "cuda", persistent_workers=args.workers > 0,
                        generator=generator)
    history, elapsed_before = [], 0.0
    if args.resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["backbone"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        if args.max_updates is not None and any(not item.get("complete_epoch", True) for item in history):
            raise ValueError("Cannot resume after a partial final epoch")
        elapsed_before = history[-1]["seconds"]
        generator.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])
        del state
    started = time.monotonic()
    updates_completed = len(history) * steps_per_epoch if args.max_updates is not None else 0
    seen_examples = sum(item["records"] for item in history) if args.max_updates is not None else 0
    for epoch in range(len(history), args.epochs):
        model.train()
        total_loss, observations, maximum_grad = 0.0, 0, 0.0
        for step, (views, patients) in enumerate(loader):
            if len(views) < 2:
                raise ValueError("Singleton final contrastive batch; choose another batch size")
            views = views.to(args.device, non_blocking=True)
            patients = patients.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            tokens = model.extract_features(source=views.flatten(0, 1), padding_mask=None, mask=False)["x"]
            features = tokens.mean(dim=1).reshape(len(views), 2, -1)
            loss = temporal_contrastive_loss(features, patients, args.temperature)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite contrastive loss")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            maximum_grad = max(maximum_grad, float(gradient_norm))
            optimizer.step()
            if args.max_updates is not None:
                updates_completed += 1
                seen_examples += len(views)
            total_loss += float(loss.detach()) * len(views)
            observations += len(views)
            if (step + 1) % 100 == 0:
                print(json.dumps({"epoch": epoch + 1, "records": observations,
                                  "running_loss": total_loss / observations,
                                  **({"updates": updates_completed, "seen_examples": seen_examples}
                                     if args.max_updates is not None else {}),
                                  "seconds": elapsed_before + time.monotonic() - started}), flush=True)
            if args.max_updates is not None and updates_completed == args.max_updates:
                break
        complete_epoch = observations == steps_per_epoch * args.batch_size if drop_last else observations == len(rows)
        if (args.max_updates is None and not complete_epoch) or maximum_grad <= 0:
            raise RuntimeError("Incomplete epoch or no gradient updates")
        entry = {"epoch": epoch + 1, "records": observations, "loss": total_loss / observations,
                 "maximum_gradient_norm_before_clipping": maximum_grad,
                 "seconds": elapsed_before + time.monotonic() - started}
        if args.max_updates is not None:
            entry.update({"updates": updates_completed, "seen_examples": seen_examples,
                          "complete_epoch": complete_epoch})
        history.append(entry)
        save_json(args.output_dir / "history.json", history)
        print(json.dumps({"stage": "ecg-fm_cmsc_adaptation", **entry}), flush=True)
        if complete_epoch:
            temporary = args.output_dir / "resume.partial.pt"
            torch.save({"backbone": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "optimizer": optimizer.state_dict(), "history": history,
                        "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                        "python_rng": random.getstate()}, temporary)
            os.replace(temporary, resume_path)
        if args.max_updates is not None and updates_completed == args.max_updates:
            break
    backbone = {k: value.detach().cpu() for k, value in model.state_dict().items()}
    if not all(torch.isfinite(value).all() for value in backbone.values() if value.is_floating_point()):
        raise RuntimeError("Nonfinite adapted checkpoint")
    path = args.output_dir / "adapted_backbone.pt"
    torch.save({"backbone": backbone, "metadata": config, "history": history}, path)
    completion = {"checkpoint": path.name, "sha256": sha256(path),
              "completed_epochs": len(history), "seconds": elapsed_before + time.monotonic() - started,
              "torch_version": str(torch.__version__),
              "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None}
    if args.max_updates is not None:
        completion.update({"updates": updates_completed, "seen_examples": seen_examples,
                           "completed_epochs": sum(item["complete_epoch"] for item in history)})
    save_json(args.output_dir / "completion.json", completion)
    # The completed portable backbone supersedes this run's optimizer resume file.
    resume_path.unlink(missing_ok=True)
    print(f"Completed adaptation: {path}", flush=True)


if __name__ == "__main__":
    main()
