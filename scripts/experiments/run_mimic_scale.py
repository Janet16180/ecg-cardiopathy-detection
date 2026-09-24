#!/usr/bin/env python3
"""Run the prespecified Experiment 003 GPU stages sequentially.

Nothing starts on import. Invoke with the pretrained Python environment after
the direct baseline has finished. Optional PIDs let this runner wait for an
already running baseline or MIMIC downloader without starting either process.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/processed/ptbxl/seed42_fraction1"
PTB_RAW = ROOT / "data/raw/ptb-xl/1.0.3"
MIMIC = ROOT / "data/processed/mimic_ssl_200k"
CHECKPOINT = ROOT / "third_party/checkpoints/ecg-fm/mimic_iv_ecg_physionet_pretrained.pt"
CHECKPOINT_META = ROOT / "data/processed/pretrained/ecg-fm/metadata.json"
CACHE = ROOT / "data/processed/ptbxl/pretrained_views/ecg-fm_full"
OUTPUT = ROOT / "outputs/experiment003_mimic"
BASELINE = "ecg-fm_direct_full_seed42/ecg-fm_finetuned_seed42"
PTB_ADAPT = "ecg-fm_ptb_ssl_7000"
POOLED_ADAPT = "ecg-fm_pooled_ssl_7000"
PTB_FINE = "ecg-fm_adapted_finetuned_seed42"
POOLED_FINE = "ecg-fm_pooled_adapted_finetuned_seed42"
UPDATES = 7000
BATCH = 32


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def csv_rows(path: Path, fields: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != fields:
            raise ValueError(f"Unexpected columns in {path}")
        return list(reader)


def environment() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the runner's Python environment")
    packages = {}
    for name in ("torch", "numpy", "scikit-learn", "wfdb", "transformers", "fairseq-signals"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": packages,
            "torch_cuda_version": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(),
            "cuda_device_count": torch.cuda.device_count(), "checked_at": now()}


def process_cmdline(pid: int) -> str | None:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    if not raw:
        return None
    return raw.replace(b"\0", b" ").decode(errors="replace")


def wait_for_artifact(pid: int | None, process_marker: str, artifact: Path, validate) -> None:
    """Never accept a reused PID or a process exit without its final artifact."""
    if artifact.is_file():
        try:
            validate()
            return
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            if pid is None:
                raise
    if pid is None:
        raise FileNotFoundError(f"Required artifact is absent: {artifact}; supply its active wait PID")
    while True:
        command = process_cmdline(pid)
        if command is None:
            if not artifact.is_file():
                raise RuntimeError(f"PID {pid} ended without required artifact {artifact}")
            validate()
            return
        if process_marker not in command:
            raise RuntimeError(f"PID {pid} no longer identifies {process_marker}: {command}")
        if artifact.is_file():
            try:
                validate()
                return
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
                # The producer may still be writing its other completion files.
                pass
        print(f"Waiting for PID {pid}: {artifact.name}", flush=True)
        time.sleep(30)


def validate_mimic() -> dict:
    selection = read_json(MIMIC / "selection.json")
    metadata = read_json(MIMIC / "metadata.json")
    selected_path = MIMIC / "selected_records.csv"
    manifest_path = MIMIC / "ssl_manifest.csv"
    selected = csv_rows(selected_path, {"subject_id", "study_id", "path"})
    accepted = csv_rows(manifest_path, {"ecg_id", "patient_id", "raw_dir", "filename_hr", "source"})
    if (selection["max_records"] != 200000 or metadata["max_records"] != 200000
            or selection["seed"] != 42 or metadata["seed"] != 42
            or not 0 < len(accepted) <= len(selected) <= 200000
            or len(selected) != selection["selected_records"]
            or len(selected) != metadata["selected_records"]
            or len(accepted) != metadata["accepted_records"]
            or metadata["accepted_patients"] != len({row["patient_id"] for row in accepted})
            or metadata["selection_sha256"] != selection["selection_sha256"]
            or metadata["selected_records_sha256"] != digest(selected_path)
            or metadata["manifest_sha256"] != digest(manifest_path)
            or metadata["exclusions_sha256"] != digest(MIMIC / "exclusions.csv")
            or len({row["ecg_id"] for row in accepted}) != len(accepted)
            or any(row["source"] != "mimic" or not row["ecg_id"].startswith("mimic:")
                   or not row["patient_id"].startswith("mimic:") for row in accepted)):
        raise ValueError("MIMIC selection/manifest/metadata integrity validation failed")
    ptb_count = len(csv_rows(MANIFEST / "all_train_ssl.csv",
                             {"ecg_id", "patient_id", "filename_lr", "filename_hr"}))
    if ptb_count != 17418 or 2 * ((ptb_count + len(accepted)) // BATCH) < UPDATES:
        raise ValueError("Accepted pool cannot supply the prespecified 7,000 updates in two epochs")
    return {"selected_records": len(selected), "accepted_records": len(accepted),
            "accepted_patients": metadata["accepted_patients"], "ptbxl_records": ptb_count,
            "pool_records": ptb_count + len(accepted), "manifest_sha256": metadata["manifest_sha256"],
            "selection_sha256": metadata["selection_sha256"],
            "exclusion_counts": metadata["exclusion_counts"]}


def validate_adaptation(directory: Path, pooled: bool) -> dict:
    config = read_json(directory / "config.json")
    history = read_json(directory / "history.json")
    completion = read_json(directory / "completion.json")
    checkpoint = directory / "adapted_backbone.pt"
    if (config.get("max_updates") != UPDATES or config.get("batch_size") != BATCH
            or config.get("epochs") != (2 if pooled else 14)
            or config.get("drop_last") is not True
            or config.get("manifest_sha256", {}).get("all_train_ssl.csv")
            != digest(MANIFEST / "all_train_ssl.csv")
            or config.get("official_checkpoint", {}).get("sha256") != digest(CHECKPOINT)
            or bool(config.get("external_ssl")) != pooled
            or (pooled and config["external_ssl"]["manifest_sha256"] != digest(MIMIC / "ssl_manifest.csv"))
            or not history or history[-1].get("updates") != UPDATES
            or history[-1].get("seen_examples") != UPDATES * BATCH
            or completion.get("updates") != UPDATES
            or completion.get("seen_examples") != UPDATES * BATCH
            or completion.get("sha256") != digest(checkpoint)):
        raise ValueError(f"Invalid or incomplete adaptation: {directory}")
    return {"updates": UPDATES, "seen_examples": UPDATES * BATCH,
            "completed_epochs": completion["completed_epochs"], "checkpoint_sha256": completion["sha256"]}


def validate_finetune(directory: Path, adaptation: Path | None = None) -> dict:
    config = read_json(directory / "config.json")
    metrics = read_json(directory / "metrics.json")
    predictions = csv_rows(directory / "test_predictions.csv",
                           {"ecg_id", "patient_id", "target", "raw_logit", "probability", "prediction"})
    expected_hash = digest(adaptation) if adaptation else None
    observed_adaptation = config.get("adaptation")
    if (not (directory / "model.pt").is_file()
            or config.get("model") != "ecg-fm" or config.get("label_seed") != 42
            or any(config.get("manifest_sha256", {}).get(f"{name}.csv")
                   != digest(MANIFEST / f"{name}.csv")
                   for name in ("labeled_train", "validation", "test"))
            or config.get("official_checkpoint", {}).get("sha256") != digest(CHECKPOINT)
            or config.get("batch_size") != 16 or config.get("epochs_budget") != 20
            or config.get("patience") != 5 or config.get("records", {}).get("train") != 15360
            or config.get("records", {}).get("test") != 1896
            or bool(observed_adaptation) != bool(adaptation)
            or (adaptation and observed_adaptation.get("checkpoint_sha256") != expected_hash)
            or metrics.get("label_seed") != 42 or metrics.get("test", {}).get("n") != 1896
            or len(predictions) != 1896 or len({row["ecg_id"] for row in predictions}) != 1896):
        raise ValueError(f"Invalid or incomplete fine-tuning result: {directory}")
    return {"test_auroc": metrics["test"]["auroc"], "test_records": len(predictions),
            "model_sha256": digest(directory / "model.pt")}


def record_status(output: Path, stage: str, state: str, **details) -> None:
    path = output / "status.json"
    status = read_json(path) if path.is_file() else {"created_at": now(), "stages": {}}
    status["stages"][stage] = {"state": state, "updated_at": now(), **details}
    atomic_json(path, status)


def run_stage(output: Path, name: str, command: list[str], validate, *, resume: bool = False) -> dict:
    log = output / f"{name}.log"
    record_status(output, name, "running", command=command, log=str(log), resume=resume)
    with log.open("a", encoding="utf-8") as logfile:
        logfile.write(f"\n[{now()}] {' '.join(command)}\n")
        logfile.flush()
        process = None
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            atomic_json(output / "pid.json", {"stage": name, "pid": process.pid,
                                               "started_at": now(), "command": command})
            assert process.stdout is not None
            for line in process.stdout:
                logfile.write(line)
                logfile.flush()
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise RuntimeError(f"{name} exited with code {code}; see {log}")
            result = validate()
        except BaseException as exc:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
            record_status(output, name, "failed", error=str(exc), log=str(log))
            raise
    record_status(output, name, "complete", result=result, log=str(log))
    return result


def ensure_stage(output: Path, name: str, directory: Path, command: list[str], validate,
                 *, resumable: bool = False, completion_marker: str | None = None) -> dict:
    try:
        marker = completion_marker or ("completion.json" if resumable else "metrics.json")
        if (directory / marker).is_file():
            try:
                result = validate()
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
                if (not resumable or not (directory / "resume.pt").is_file()
                        or (directory / "config.json").is_file()):
                    raise
            else:
                record_status(output, name, "complete", result=result, skipped_existing=True)
                return result
        resume = False
        if directory.exists() and any(directory.iterdir()):
            if resumable and (directory / "resume.pt").is_file():
                resume = True
                command = [*command, "--resume"]
            else:
                raise RuntimeError(f"Incomplete existing {name} output at {directory}; inspect it before rerunning")
        return run_stage(output, name, command, validate, resume=resume)
    except BaseException as exc:
        record_status(output, name, "failed", error=str(exc))
        raise


def adaptation_command(directory: Path, epochs: int, extra: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", "scripts.experiments.adapt_ecgfm",
               "--manifest-dir", str(MANIFEST), "--raw-dir", str(PTB_RAW),
               "--checkpoint", str(CHECKPOINT), "--checkpoint-metadata", str(CHECKPOINT_META),
               "--output-dir", str(directory), "--epochs", str(epochs),
               "--max-updates", str(UPDATES), "--batch-size", str(BATCH),
               "--seed", "42", "--device", "cuda", "--protocol-note",
               "Experiment 003; 7000 matched full-batch optimizer updates"]
    if extra:
        command.extend(("--extra-ssl-manifest", str(extra)))
    return command


def finetune_command(output: Path, checkpoint: Path) -> list[str]:
    return [sys.executable, "-m", "scripts.experiments.finetune_pretrained", "--model", "ecg-fm",
            "--manifest-dir", str(MANIFEST), "--raw-dir", str(PTB_RAW),
            "--output-dir", str(output), "--cache-dir", str(CACHE),
            "--seed", "42", "--epochs", "20", "--patience", "5",
            "--batch-size", "16", "--adapted-backbone", str(checkpoint)]


def execute(args) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status_path = OUTPUT / "status.json"
    checked_environment = environment()
    status = read_json(status_path) if status_path.is_file() else {"created_at": now(), "stages": {}}
    previous_environment = status.get("environment")
    if previous_environment:
        for key in ("executable", "packages", "torch_cuda_version", "cuda_device"):
            if previous_environment.get(key) != checked_environment[key]:
                raise ValueError(f"Runner environment changed since the first stage: {key}")
    status["environment"] = checked_environment
    atomic_json(status_path, status)
    record_status(OUTPUT, "runner", "running")

    baseline_dir = OUTPUT / BASELINE
    record_status(OUTPUT, "direct_baseline", "waiting", pid=args.wait_baseline_pid,
                  artifact=str(baseline_dir / "metrics.json"))
    wait_for_artifact(args.wait_baseline_pid, "finetune_pretrained",
                      baseline_dir / "metrics.json", lambda: validate_finetune(baseline_dir))
    record_status(OUTPUT, "direct_baseline", "complete", result=validate_finetune(baseline_dir))

    ptb_dir = OUTPUT / PTB_ADAPT
    ensure_stage(OUTPUT, "ptb_adaptation", ptb_dir,
                 adaptation_command(ptb_dir, 14), lambda: validate_adaptation(ptb_dir, False),
                 resumable=True)
    ptb_checkpoint = ptb_dir / "adapted_backbone.pt"
    ensure_stage(OUTPUT, "ptb_finetune", OUTPUT / PTB_FINE,
                 finetune_command(OUTPUT, ptb_checkpoint),
                 lambda: validate_finetune(OUTPUT / PTB_FINE, ptb_checkpoint),
                 resumable=True, completion_marker="metrics.json")

    record_status(OUTPUT, "mimic_preparation", "waiting", pid=args.wait_download_pid,
                  artifact=str(MIMIC / "metadata.json"))
    wait_for_artifact(args.wait_download_pid, "prepare_mimic_ssl",
                      MIMIC / "metadata.json", validate_mimic)
    pool = validate_mimic()
    record_status(OUTPUT, "mimic_preparation", "complete", result=pool)

    pooled_dir = OUTPUT / POOLED_ADAPT
    ensure_stage(OUTPUT, "pooled_adaptation", pooled_dir,
                 adaptation_command(pooled_dir, 2, MIMIC / "ssl_manifest.csv"),
                 lambda: validate_adaptation(pooled_dir, True), resumable=True)
    pooled_checkpoint = pooled_dir / "adapted_backbone.pt"
    ensure_stage(OUTPUT, "pooled_finetune", OUTPUT / POOLED_FINE,
                 finetune_command(OUTPUT, pooled_checkpoint),
                 lambda: validate_finetune(OUTPUT / POOLED_FINE, pooled_checkpoint),
                 resumable=True, completion_marker="metrics.json")

    def comparison(name: str, filename: str, reference: str, models: tuple[str, ...]) -> None:
        path = OUTPUT / filename

        def validate():
            result = read_json(path)
            items = result.get("comparisons", [])
            if (result.get("test_records") != 1896 or len(items) != len(models)
                    or {item["model"] for item in items} != set(models)
                    or any(item["reference"] != reference for item in items)):
                raise ValueError(f"Invalid paired comparison output: {path}")
            return {"comparisons": len(items), "test_records": result["test_records"]}

        if path.is_file():
            record_status(OUTPUT, name, "complete", result=validate(), skipped_existing=True)
        else:
            run_stage(OUTPUT, name, [sys.executable, "-m", "scripts.reports.compare_adaptation",
                      "--input-dir", str(OUTPUT), "--reference", reference,
                      "--models", *models, "--repeats", "500",
                      "--output-json", str(path)], validate)

    comparison("paired_direct_comparisons", "paired_adaptation_comparisons.json",
               BASELINE, (PTB_FINE, POOLED_FINE))
    comparison("paired_pooling_comparison", "paired_pooling_comparison.json",
               PTB_FINE, (POOLED_FINE,))
    print(json.dumps({"stage": "complete", "output": str(OUTPUT),
                      "adaptation_updates_per_arm": UPDATES, "examples_per_arm": UPDATES * BATCH,
                      "mimic_pool": pool}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-baseline-pid", type=int,
                        help="Existing direct baseline process to wait for")
    parser.add_argument("--wait-download-pid", type=int,
                        help="Existing MIMIC preparation process to wait for")
    args = parser.parse_args()
    if any(pid is not None and pid < 1 for pid in (args.wait_baseline_pid, args.wait_download_pid)):
        parser.error("Wait PIDs must be positive")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".runner.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Experiment 003 runner is active") from exc
        try:
            execute(args)
        except BaseException as exc:
            record_status(OUTPUT, "runner", "failed", error=str(exc))
            raise
        record_status(OUTPUT, "runner", "complete")


if __name__ == "__main__":
    main()
