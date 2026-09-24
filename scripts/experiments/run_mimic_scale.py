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
import importlib.metadata
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ecg_experiment.files import sha256_file, write_json_atomic

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
PTB_ADAPTATION_EPOCHS = 14
POOLED_ADAPTATION_EPOCHS = 2
MIMIC_MAX_RECORDS = 200000
PTB_SSL_RECORDS = 17418
SEED = 42
FINETUNE_BATCH = 16
FINETUNE_EPOCHS = 20
FINETUNE_PATIENCE = 5
TRAIN_RECORDS = 15360
TEST_RECORDS = 1896
COMPARISON_REPEATS = 500
WAIT_SECONDS = 30
PACKAGES = ("torch", "numpy", "scikit-learn", "wfdb", "transformers", "fairseq-signals")
MANIFEST_FIELDS = {"ecg_id", "patient_id", "raw_dir", "filename_hr", "source"}
ENVIRONMENT_KEYS = ("executable", "packages", "torch_cuda_version", "cuda_device")
# Validation failures that mean a producer has not finished writing its outputs.
INCOMPLETE_ERRORS = (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError)

Validator = Callable[[], Any]


def now() -> str:
    """
    Return the current UTC time.

    Returns
    -------
    str
        ISO 8601 timestamp.
    """
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    """
    Read a UTF-8 JSON file.

    Parameters
    ----------
    path : Path
        JSON file.

    Returns
    -------
    Any
        Parsed value.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path, fields: set[str]) -> list[dict[str, str]]:
    """
    Read a CSV file whose header must equal the expected columns.

    Parameters
    ----------
    path : Path
        CSV file.
    fields : set[str]
        Required column names.

    Returns
    -------
    list[dict[str, str]]
        Rows keyed by column name.

    Raises
    ------
    ValueError
        If the columns differ.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != fields:
            raise ValueError(f"Unexpected columns in {path}")
        return list(reader)


def package_version(name: str) -> str | None:
    """
    Return an installed distribution's version.

    Parameters
    ----------
    name : str
        Distribution name.

    Returns
    -------
    str | None
        Version, or None when the distribution is not installed.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    """
    Describe the runner's Python, packages, and CUDA device.

    Returns
    -------
    dict[str, Any]
        Environment record.

    Raises
    ------
    RuntimeError
        If CUDA is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the runner's Python environment")
    return {"python": sys.version, "executable": sys.executable,
            "packages": {name: package_version(name) for name in PACKAGES},
            "torch_cuda_version": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(),
            "cuda_device_count": torch.cuda.device_count(), "checked_at": now()}


def process_cmdline(pid: int) -> str | None:
    """
    Return a process's command line.

    Parameters
    ----------
    pid : int
        Process identifier.

    Returns
    -------
    str | None
        Space-joined arguments, or None when the process does not exist.
    """
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    return raw.replace(b"\0", b" ").decode(errors="replace") if raw else None


def _validates(validate: Validator) -> bool:
    """Return whether the artifact validates, treating incomplete outputs as not yet ready."""
    try:
        validate()
    except INCOMPLETE_ERRORS:
        return False
    return True


def _wait_while_running(pid: int, process_marker: str, artifact: Path, validate: Validator) -> bool:
    """Poll until the artifact validates (True) or the process ends (False)."""
    if artifact.is_file() and _validates(validate):
        return True
    while True:
        command = process_cmdline(pid)
        if command is None:
            return False
        if process_marker not in command:
            raise RuntimeError(f"PID {pid} no longer identifies {process_marker}: {command}")
        if artifact.is_file() and _validates(validate):
            return True
        print(f"Waiting for PID {pid}: {artifact.name}", flush=True)
        time.sleep(WAIT_SECONDS)


def wait_for_artifact(pid: int | None, process_marker: str, artifact: Path, validate: Validator) -> None:
    """
    Never accept a reused PID or a process exit without its final artifact.

    Parameters
    ----------
    pid : int | None
        Producer process to wait for, or None when the artifact must exist.
    process_marker : str
        Text that the producer's command line must contain.
    artifact : Path
        Final artifact of the producer.
    validate : Callable[[], Any]
        Validator of the producer's outputs.

    Raises
    ------
    FileNotFoundError
        If no PID is given and the artifact is absent.
    RuntimeError
        If the PID identifies another process, or the producer exits without
        its artifact.
    """
    if pid is None:
        if not artifact.is_file():
            raise FileNotFoundError(f"Required artifact is absent: {artifact}; supply its active wait PID")
        validate()
        return
    if _wait_while_running(pid, process_marker, artifact, validate):
        return
    if not artifact.is_file():
        raise RuntimeError(f"PID {pid} ended without required artifact {artifact}")
    validate()


def _mimic_metadata_consistent(selection: dict[str, Any], metadata: dict[str, Any],
                               selected: list[dict[str, str]], accepted: list[dict[str, str]]) -> bool:
    """Check selection, manifest, and metadata counts and digests against each other."""
    return (selection["max_records"] == MIMIC_MAX_RECORDS and metadata["max_records"] == MIMIC_MAX_RECORDS
            and selection["seed"] == SEED and metadata["seed"] == SEED
            and 0 < len(accepted) <= len(selected) <= MIMIC_MAX_RECORDS
            and len(selected) == selection["selected_records"]
            and len(selected) == metadata["selected_records"]
            and len(accepted) == metadata["accepted_records"]
            and metadata["accepted_patients"] == len({row["patient_id"] for row in accepted})
            and metadata["selection_sha256"] == selection["selection_sha256"]
            and metadata["selected_records_sha256"] == sha256_file(MIMIC / "selected_records.csv")
            and metadata["manifest_sha256"] == sha256_file(MIMIC / "ssl_manifest.csv")
            and metadata["exclusions_sha256"] == sha256_file(MIMIC / "exclusions.csv"))


def _mimic_rows_namespaced(accepted: list[dict[str, str]]) -> bool:
    """Check that accepted rows are unique and carry the ``mimic:`` namespace."""
    return (len({row["ecg_id"] for row in accepted}) == len(accepted)
            and all(row["source"] == "mimic" and row["ecg_id"].startswith("mimic:")
                    and row["patient_id"].startswith("mimic:") for row in accepted))


def validate_mimic() -> dict[str, Any]:
    """
    Validate the prepared MIMIC pool and its capacity for the update budget.

    Returns
    -------
    dict[str, Any]
        Pool counts and digests.

    Raises
    ------
    ValueError
        If the pool is inconsistent or cannot supply the update budget.
    """
    selection = read_json(MIMIC / "selection.json")
    metadata = read_json(MIMIC / "metadata.json")
    selected = csv_rows(MIMIC / "selected_records.csv", {"subject_id", "study_id", "path"})
    accepted = csv_rows(MIMIC / "ssl_manifest.csv", MANIFEST_FIELDS)
    if (not _mimic_metadata_consistent(selection, metadata, selected, accepted)
            or not _mimic_rows_namespaced(accepted)):
        raise ValueError("MIMIC selection/manifest/metadata integrity validation failed")
    ptb_count = len(csv_rows(MANIFEST / "all_train_ssl.csv",
                             {"ecg_id", "patient_id", "filename_lr", "filename_hr"}))
    pool = ptb_count + len(accepted)
    if ptb_count != PTB_SSL_RECORDS or POOLED_ADAPTATION_EPOCHS * (pool // BATCH) < UPDATES:
        raise ValueError("Accepted pool cannot supply the prespecified 7,000 updates in two epochs")
    return {"selected_records": len(selected), "accepted_records": len(accepted),
            "accepted_patients": metadata["accepted_patients"], "ptbxl_records": ptb_count,
            "pool_records": pool, "manifest_sha256": metadata["manifest_sha256"],
            "selection_sha256": metadata["selection_sha256"],
            "exclusion_counts": metadata["exclusion_counts"]}


def _adaptation_config_valid(config: dict[str, Any], pooled: bool) -> bool:
    """Check the recorded adaptation protocol and its input digests."""
    return (config.get("max_updates") == UPDATES and config.get("batch_size") == BATCH
            and config.get("epochs") == (POOLED_ADAPTATION_EPOCHS if pooled else PTB_ADAPTATION_EPOCHS)
            and config.get("drop_last") is True
            and config.get("manifest_sha256", {}).get("all_train_ssl.csv")
            == sha256_file(MANIFEST / "all_train_ssl.csv")
            and config.get("official_checkpoint", {}).get("sha256") == sha256_file(CHECKPOINT)
            and bool(config.get("external_ssl")) == pooled
            and (not pooled
                 or config["external_ssl"]["manifest_sha256"] == sha256_file(MIMIC / "ssl_manifest.csv")))


def validate_adaptation(directory: Path, pooled: bool) -> dict[str, Any]:
    """
    Validate a completed bounded ECG-FM adaptation.

    Parameters
    ----------
    directory : Path
        Adaptation output directory.
    pooled : bool
        Whether the adaptation includes the MIMIC pool.

    Returns
    -------
    dict[str, Any]
        Update counts and the checkpoint digest.

    Raises
    ------
    ValueError
        If the adaptation used another protocol or did not finish its budget.
    """
    config = read_json(directory / "config.json")
    history = read_json(directory / "history.json")
    completion = read_json(directory / "completion.json")
    checkpoint = directory / "adapted_backbone.pt"
    if (not _adaptation_config_valid(config, pooled)
            or not history or history[-1].get("updates") != UPDATES
            or history[-1].get("seen_examples") != UPDATES * BATCH
            or completion.get("updates") != UPDATES
            or completion.get("seen_examples") != UPDATES * BATCH
            or completion.get("sha256") != sha256_file(checkpoint)):
        raise ValueError(f"Invalid or incomplete adaptation: {directory}")
    return {"updates": UPDATES, "seen_examples": UPDATES * BATCH,
            "completed_epochs": completion["completed_epochs"], "checkpoint_sha256": completion["sha256"]}


def _finetune_config_valid(config: dict[str, Any], adaptation_sha256: str | None) -> bool:
    """Check the recorded fine-tuning protocol, inputs, and adapted checkpoint digest."""
    observed_adaptation = config.get("adaptation")
    manifests = config.get("manifest_sha256", {})
    return (config.get("model") == "ecg-fm" and config.get("label_seed") == SEED
            and all(manifests.get(f"{name}.csv") == sha256_file(MANIFEST / f"{name}.csv")
                    for name in ("labeled_train", "validation", "test"))
            and config.get("official_checkpoint", {}).get("sha256") == sha256_file(CHECKPOINT)
            and config.get("batch_size") == FINETUNE_BATCH and config.get("epochs_budget") == FINETUNE_EPOCHS
            and config.get("patience") == FINETUNE_PATIENCE
            and config.get("records", {}).get("train") == TRAIN_RECORDS
            and config.get("records", {}).get("test") == TEST_RECORDS
            and bool(observed_adaptation) == bool(adaptation_sha256)
            and (not adaptation_sha256 or observed_adaptation.get("checkpoint_sha256") == adaptation_sha256))


def validate_finetune(directory: Path, adaptation: Path | None = None) -> dict[str, Any]:
    """
    Validate a completed ECG-FM fine-tuning run.

    Parameters
    ----------
    directory : Path
        Fine-tuning run directory.
    adaptation : Path | None
        Adapted backbone the run must have loaded, or None for direct runs.

    Returns
    -------
    dict[str, Any]
        Test AUROC, record count, and the model digest.

    Raises
    ------
    ValueError
        If the run used another protocol or its outputs are incomplete.
    """
    config = read_json(directory / "config.json")
    metrics = read_json(directory / "metrics.json")
    predictions = csv_rows(directory / "test_predictions.csv",
                           {"ecg_id", "patient_id", "target", "raw_logit", "probability", "prediction"})
    adaptation_sha256 = sha256_file(adaptation) if adaptation else None
    if (not (directory / "model.pt").is_file() or not _finetune_config_valid(config, adaptation_sha256)
            or metrics.get("label_seed") != SEED or metrics.get("test", {}).get("n") != TEST_RECORDS
            or len(predictions) != TEST_RECORDS
            or len({row["ecg_id"] for row in predictions}) != TEST_RECORDS):
        raise ValueError(f"Invalid or incomplete fine-tuning result: {directory}")
    return {"test_auroc": metrics["test"]["auroc"], "test_records": len(predictions),
            "model_sha256": sha256_file(directory / "model.pt")}


def record_status(output: Path, stage: str, state: str, **details: Any) -> None:
    """
    Record a stage state in the runner's ``status.json``.

    Parameters
    ----------
    output : Path
        Runner output directory.
    stage : str
        Stage name.
    state : str
        New state, such as ``"running"`` or ``"complete"``.
    **details : Any
        Extra fields stored with the state.
    """
    path = output / "status.json"
    status = read_json(path) if path.is_file() else {"created_at": now(), "stages": {}}
    status["stages"][stage] = {"state": state, "updated_at": now(), **details}
    write_json_atomic(path, status)


def _stream_output(process: subprocess.Popen, logfile: Any) -> None:
    """Copy the child's combined output to the log and to stdout, line by line."""
    if process.stdout is None:
        raise RuntimeError("Stage process was started without an output pipe")
    for line in process.stdout:
        logfile.write(line)
        logfile.flush()
        print(line, end="", flush=True)


def run_stage(output: Path, name: str, command: list[str], validate: Validator, *,
              resume: bool = False) -> Any:
    """
    Run one stage as a logged child process and validate its outputs.

    Parameters
    ----------
    output : Path
        Runner output directory.
    name : str
        Stage name.
    command : list[str]
        Child command.
    validate : Callable[[], Any]
        Validator of the stage outputs.
    resume : bool
        Whether the command resumes a partial run; recorded in the status.

    Returns
    -------
    Any
        Validator result.

    Raises
    ------
    RuntimeError
        If the child exits with a nonzero code.
    """
    log = output / f"{name}.log"
    record_status(output, name, "running", command=command, log=str(log), resume=resume)
    with log.open("a", encoding="utf-8") as logfile:
        logfile.write(f"\n[{now()}] {' '.join(command)}\n")
        logfile.flush()
        process = None
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            write_json_atomic(output / "pid.json", {"stage": name, "pid": process.pid,
                                                    "started_at": now(), "command": command})
            _stream_output(process, logfile)
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


def existing_stage(name: str, directory: Path, command: list[str], validate: Validator, *, resumable: bool,
                   marker: str) -> tuple[Any | None, list[str], bool]:
    """
    Inspect a stage directory before running it.

    Parameters
    ----------
    name : str
        Stage name used in error messages.
    directory : Path
        Stage output directory.
    command : list[str]
        Command that runs the stage.
    validate : Callable[[], Any]
        Validator for completed stage outputs.
    resumable : bool
        Whether the stage can continue from ``resume.pt``.
    marker : str
        File whose presence marks a completed stage.

    Returns
    -------
    tuple[Any | None, list[str], bool]
        Validated result of a completed stage (or None), the command to run,
        and whether the command resumes a partial run.

    Raises
    ------
    RuntimeError
        If the directory holds incomplete output that cannot be resumed.
    """
    if (directory / marker).is_file():
        try:
            return validate(), command, False
        except INCOMPLETE_ERRORS:
            if (not resumable or not (directory / "resume.pt").is_file()
                    or (directory / "config.json").is_file()):
                raise
    if not directory.exists() or not any(directory.iterdir()):
        return None, command, False
    if not resumable or not (directory / "resume.pt").is_file():
        raise RuntimeError(f"Incomplete existing {name} output at {directory}; inspect it before rerunning")
    return None, [*command, "--resume"], True


def ensure_stage(output: Path, name: str, directory: Path, command: list[str], validate: Validator,
                 *, resumable: bool = False, completion_marker: str | None = None) -> Any:
    """
    Reuse a validated completed stage, or run (or resume) it.

    Parameters
    ----------
    output : Path
        Runner output directory.
    name : str
        Stage name.
    directory : Path
        Stage output directory.
    command : list[str]
        Command that runs the stage.
    validate : Callable[[], Any]
        Validator of the stage outputs.
    resumable : bool
        Whether the stage can continue from ``resume.pt``.
    completion_marker : str | None
        File that marks completion; defaults by stage kind.

    Returns
    -------
    Any
        Validator result.
    """
    marker = completion_marker or ("completion.json" if resumable else "metrics.json")
    try:
        result, command, resume = existing_stage(name, directory, command, validate,
                                                 resumable=resumable, marker=marker)
    except BaseException as exc:
        record_status(output, name, "failed", error=str(exc))
        raise
    if result is not None:
        record_status(output, name, "complete", result=result, skipped_existing=True)
        return result
    return run_stage(output, name, command, validate, resume=resume)


def adaptation_command(directory: Path, epochs: int, extra: Path | None = None) -> list[str]:
    """
    Build the bounded ECG-FM adaptation command.

    Parameters
    ----------
    directory : Path
        Adaptation output directory.
    epochs : int
        Epoch ceiling that supplies the update budget.
    extra : Path | None
        External SSL manifest to pool, if any.

    Returns
    -------
    list[str]
        Child command.
    """
    command = [sys.executable, "-m", "scripts.experiments.adapt_ecgfm",
               "--manifest-dir", str(MANIFEST), "--raw-dir", str(PTB_RAW),
               "--checkpoint", str(CHECKPOINT), "--checkpoint-metadata", str(CHECKPOINT_META),
               "--output-dir", str(directory), "--epochs", str(epochs),
               "--max-updates", str(UPDATES), "--batch-size", str(BATCH),
               "--seed", str(SEED), "--device", "cuda", "--protocol-note",
               "Experiment 003; 7000 matched full-batch optimizer updates"]
    if extra:
        command.extend(("--extra-ssl-manifest", str(extra)))
    return command


def finetune_command(output: Path, checkpoint: Path) -> list[str]:
    """
    Build the ECG-FM fine-tuning command for an adapted backbone.

    Parameters
    ----------
    output : Path
        Fine-tuning output root.
    checkpoint : Path
        Adapted backbone.

    Returns
    -------
    list[str]
        Child command.
    """
    return [sys.executable, "-m", "scripts.experiments.finetune_pretrained", "--model", "ecg-fm",
            "--manifest-dir", str(MANIFEST), "--raw-dir", str(PTB_RAW),
            "--output-dir", str(output), "--cache-dir", str(CACHE),
            "--seed", str(SEED), "--epochs", str(FINETUNE_EPOCHS), "--patience", str(FINETUNE_PATIENCE),
            "--batch-size", str(FINETUNE_BATCH), "--adapted-backbone", str(checkpoint)]


def validate_comparison(path: Path, reference: str, models: tuple[str, ...]) -> dict[str, int]:
    """
    Validate a paired comparison output.

    Parameters
    ----------
    path : Path
        Comparison JSON.
    reference : str
        Reference run every comparison must use.
    models : tuple[str, ...]
        Compared runs.

    Returns
    -------
    dict[str, int]
        Number of comparisons and test records.

    Raises
    ------
    ValueError
        If the comparisons or test records differ from the request.
    """
    result = read_json(path)
    items = result.get("comparisons", [])
    if (result.get("test_records") != TEST_RECORDS or len(items) != len(models)
            or {item["model"] for item in items} != set(models)
            or any(item["reference"] != reference for item in items)):
        raise ValueError(f"Invalid paired comparison output: {path}")
    return {"comparisons": len(items), "test_records": result["test_records"]}


def run_comparison(name: str, filename: str, reference: str, models: tuple[str, ...]) -> None:
    """
    Reuse or compute one paired test comparison.

    Parameters
    ----------
    name : str
        Stage name.
    filename : str
        Output file name inside ``OUTPUT``.
    reference : str
        Reference run.
    models : tuple[str, ...]
        Runs compared against the reference.
    """
    path = OUTPUT / filename

    def validate() -> dict[str, int]:
        return validate_comparison(path, reference, models)

    if path.is_file():
        record_status(OUTPUT, name, "complete", result=validate(), skipped_existing=True)
        return
    run_stage(OUTPUT, name, [sys.executable, "-m", "scripts.reports.compare_adaptation",
                             "--input-dir", str(OUTPUT), "--reference", reference,
                             "--models", *models, "--repeats", str(COMPARISON_REPEATS),
                             "--output-json", str(path)], validate)


def check_environment() -> None:
    """
    Record the runner environment, refusing a change since the first stage.

    Raises
    ------
    ValueError
        If the executable, packages, or CUDA setup changed.
    """
    status_path = OUTPUT / "status.json"
    checked_environment = environment()
    status = read_json(status_path) if status_path.is_file() else {"created_at": now(), "stages": {}}
    previous_environment = status.get("environment")
    for key in ENVIRONMENT_KEYS:
        if previous_environment and previous_environment.get(key) != checked_environment[key]:
            raise ValueError(f"Runner environment changed since the first stage: {key}")
    status["environment"] = checked_environment
    write_json_atomic(status_path, status)


def adapt_and_finetune(stage: str, adaptation_dir: Path, finetune_dir: Path, command: list[str],
                       pooled: bool) -> None:
    """
    Run one adaptation arm and then fine-tune its final backbone.

    Parameters
    ----------
    stage : str
        Stage prefix, ``"ptb"`` or ``"pooled"``.
    adaptation_dir : Path
        Adaptation output directory.
    finetune_dir : Path
        Fine-tuning run directory.
    command : list[str]
        Adaptation command.
    pooled : bool
        Whether the adaptation pools MIMIC.
    """
    ensure_stage(OUTPUT, f"{stage}_adaptation", adaptation_dir, command,
                 lambda: validate_adaptation(adaptation_dir, pooled), resumable=True)
    checkpoint = adaptation_dir / "adapted_backbone.pt"
    ensure_stage(OUTPUT, f"{stage}_finetune", finetune_dir, finetune_command(OUTPUT, checkpoint),
                 lambda: validate_finetune(finetune_dir, checkpoint),
                 resumable=True, completion_marker="metrics.json")


def execute(args: argparse.Namespace) -> None:
    """
    Run every Experiment 003 stage in order, reusing completed stages.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with the optional wait PIDs.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    check_environment()
    record_status(OUTPUT, "runner", "running")

    baseline_dir = OUTPUT / BASELINE
    record_status(OUTPUT, "direct_baseline", "waiting", pid=args.wait_baseline_pid,
                  artifact=str(baseline_dir / "metrics.json"))
    wait_for_artifact(args.wait_baseline_pid, "finetune_pretrained",
                      baseline_dir / "metrics.json", lambda: validate_finetune(baseline_dir))
    record_status(OUTPUT, "direct_baseline", "complete", result=validate_finetune(baseline_dir))

    ptb_dir = OUTPUT / PTB_ADAPT
    adapt_and_finetune("ptb", ptb_dir, OUTPUT / PTB_FINE, adaptation_command(ptb_dir, PTB_ADAPTATION_EPOCHS),
                       pooled=False)

    record_status(OUTPUT, "mimic_preparation", "waiting", pid=args.wait_download_pid,
                  artifact=str(MIMIC / "metadata.json"))
    wait_for_artifact(args.wait_download_pid, "prepare_mimic_ssl",
                      MIMIC / "metadata.json", validate_mimic)
    pool = validate_mimic()
    record_status(OUTPUT, "mimic_preparation", "complete", result=pool)

    pooled_dir = OUTPUT / POOLED_ADAPT
    adapt_and_finetune("pooled", pooled_dir, OUTPUT / POOLED_FINE,
                       adaptation_command(pooled_dir, POOLED_ADAPTATION_EPOCHS, MIMIC / "ssl_manifest.csv"),
                       pooled=True)

    run_comparison("paired_direct_comparisons", "paired_adaptation_comparisons.json",
                   BASELINE, (PTB_FINE, POOLED_FINE))
    run_comparison("paired_pooling_comparison", "paired_pooling_comparison.json",
                   PTB_FINE, (POOLED_FINE,))
    print(json.dumps({"stage": "complete", "output": str(OUTPUT),
                      "adaptation_updates_per_arm": UPDATES, "examples_per_arm": UPDATES * BATCH,
                      "mimic_pool": pool}), flush=True)


def main() -> None:
    """
    Parse wait PIDs and run the experiment while holding the runner lock.

    Raises
    ------
    RuntimeError
        If another Experiment 003 runner holds the lock.
    """
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
