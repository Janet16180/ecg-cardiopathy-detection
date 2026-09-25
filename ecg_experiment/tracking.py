"""
Read-only historical result import for an optional MLflow registry.

This module never starts training and never changes an existing result file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .files import sha256_file

IMPORT_SCHEMA = "historical-v1"
EXPERIMENT_NAME = "ecg-cardiopathy-historical"
MAX_ARTIFACT_BYTES = 256_000
TEST_METRICS = frozenset({
    "auroc", "average_precision", "sensitivity", "specificity", "precision",
    "npv", "f1", "accuracy", "brier", "log_loss", "tp", "tn", "fp", "fn",
    "n", "prevalence", "predicted_positive_rate", "ece_10_bins",
})
DEVELOPMENT_METRICS = frozenset({"auroc", "sensitivity", "specificity"})
METRIC_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FUTURE_EXPERIMENT_NAME = "ecg-cardiopathy"
CONFIG_SCALARS = frozenset({
    "labeled_training_records", "exposed_training_labels", "model_parameters",
    "encoder_parameters", "parameters", "epochs_budget", "batch_size",
    "microbatch_size", "effective_batch_size", "backbone_lr", "head_lr",
    "core_lr", "lr", "learning_rate", "weight_decay", "optimizer",
    "feature_dimension", "seed", "budget", "epochs", "arm",
})
SOURCE_HASH_FIELDS = frozenset({
    "label_manifest_sha256", "runner_source_sha256", "finetune_source_sha256",
    "extractor_source_sha256", "evaluation_source_sha256",
})
MAX_CONFIG_PARAM_CHARS = 128
MAX_VERIFIED_PARAM_CHARS = 256
RESULT_STAGES = frozenset({"development_screen", "calibration", "held_out_test", "execution_failed"})
RUN_STATUSES = frozenset({"FINISHED", "FAILED"})


@dataclass(frozen=True)
class HistoricalRun:
    """
    One aggregate result ready for registry import.

    Attributes
    ----------
    source : Path
        Result or receipt file the record was read from.
    source_id : str
        ``source`` relative to the project root; the registry identity.
    source_sha256 : str
        Digest of ``source`` at discovery time.
    name : str
        Registry run name.
    stage : str
        Evaluation stage, such as ``held_out_test`` or ``development_screen``.
    metrics : dict[str, float]
        Finite aggregate metrics.
    params : dict[str, str]
        Allowlisted scalar parameters.
    tags : dict[str, str]
        Provenance tags.
    """

    source: Path
    source_id: str
    source_sha256: str
    name: str
    stage: str
    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _numeric_map(prefix: str, data: dict[str, Any], allowed: frozenset[str]) -> dict[str, float]:
    return {f"{prefix}.{key}": number for key, value in data.items()
            if key in allowed and (number := _metric(value)) is not None}


def _receipt_for_metrics(path: Path) -> Path | None:
    receipts = [path.with_name(name) for name in ("complete.json", "completion.json")]
    receipt = next((candidate for candidate in receipts if candidate.is_file()), None)
    if receipt is None:
        return None
    data = _json(receipt)
    hashes = data.get("sha256", data.get("artifacts", {}))
    if isinstance(hashes, dict) and "metrics.json" in hashes and hashes["metrics.json"] != sha256_file(path):
        raise ValueError(f"Receipt hash mismatch: {path}")
    return receipt


def _lookup(config: dict[str, Any], key: str) -> Any:
    """Read a config value, falling back to a nested fingerprint dictionary."""
    value = config.get(key)
    fingerprint = config.get("fingerprint")
    if value is None and isinstance(fingerprint, dict):
        value = fingerprint.get(key)
    return value


def _config_param(value: Any) -> str | None:
    """Return a short finite scalar as text, or None when it is not indexable."""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = str(value)
    if len(text) > MAX_CONFIG_PARAM_CHARS or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return text


def _labeled_record_count(config: dict[str, Any]) -> str | None:
    records = config.get("records")
    if not isinstance(records, dict):
        return None
    count = records.get("labeled_train", records.get("train"))
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        return None
    return str(count)


def _config_metadata(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read only a small allowlist of aggregate config scalars and hashes."""
    if not path.is_file():
        return {}, {}
    config = _json(path)
    fingerprint = config.get("fingerprint")
    params = {}
    tags = {"ecg.config_sha256": sha256_file(path)}
    for key in CONFIG_SCALARS:
        text = _config_param(_lookup(config, key))
        if text is not None:
            params[key] = text
    if "labeled_training_records" not in params:
        count = _labeled_record_count(config)
        if count is not None:
            params["labeled_training_records"] = count
    if isinstance(fingerprint, str) and SHA256.fullmatch(fingerprint):
        tags["ecg.historical_fingerprint_sha256"] = fingerprint
    for key in SOURCE_HASH_FIELDS:
        value = _lookup(config, key)
        if isinstance(value, str) and SHA256.fullmatch(value):
            tags[f"ecg.{key}"] = value
    return params, tags


def _interval_metrics(ci: Any) -> dict[str, float]:
    """Flatten finite bootstrap bounds of allowlisted test metrics."""
    metrics: dict[str, float] = {}
    if not isinstance(ci, dict):
        return metrics
    for name, bounds in ci.items():
        if name not in TEST_METRICS or not isinstance(bounds, list) or len(bounds) != 2:
            continue
        for label, value in zip(("lower", "upper"), bounds, strict=True):
            if (number := _metric(value)) is not None:
                metrics[f"test.ci95.{name}.{label}"] = number
    return metrics


def _legacy_run(path: Path, root: Path) -> HistoricalRun:
    data = _json(path)
    test = data.get("test")
    if not isinstance(test, dict):
        raise ValueError(f"Missing aggregate test metrics: {path}")
    receipt = _receipt_for_metrics(path)
    metrics = _numeric_map("test", test, TEST_METRICS)
    metrics.update(_interval_metrics(data.get("test_ci95_patient_bootstrap", {})))
    params = {"model": str(data.get("model", path.parent.name))}
    config_params, config_tags = _config_metadata(path.with_name("config.json"))
    for key, value in config_params.items():
        if key not in params:
            params[key] = value
    if "label_seed" in data:
        params["label_seed"] = str(data["label_seed"])
    source_id = path.relative_to(root).as_posix()
    tags = {
        "ecg.evaluation_stage": "held_out_test",
        "ecg.result_basis": "historical_metrics_json",
        "ecg.receipt": "present" if receipt else "absent",
        **config_tags,
    }
    return HistoricalRun(path, source_id, sha256_file(path), path.parent.name,
                         "held_out_test", metrics, params, tags)


def _pilot_arm_run(arm_receipt: Path, receipt_hashes: Any, root: Path) -> HistoricalRun | None:
    """Read one pilot arm's development result; None when it records no AUROC."""
    arm_name = arm_receipt.parent.name
    if (isinstance(receipt_hashes, dict) and arm_name in receipt_hashes
            and receipt_hashes[arm_name] != sha256_file(arm_receipt)):
        raise ValueError(f"Parent receipt hash mismatch: {arm_receipt}")
    arm = _json(arm_receipt)
    if "best_development_auroc" not in arm:
        return None
    auroc = _metric(arm["best_development_auroc"])
    if auroc is None:
        raise ValueError(f"Invalid development AUROC: {arm_receipt}")
    config_path = arm_receipt.with_name("config.json")
    config = _json(config_path) if config_path.is_file() else {}
    config_params, config_tags = _config_metadata(config_path)
    params = {"arm": arm_name, **config_params}
    for key in ("model_parameters", "inference_parameters", "exposed_training_labels"):
        if key in config:
            params[key] = str(config[key])
    for key in ("best_epoch", "fingerprint"):
        if key in arm:
            params[key] = str(arm[key])
    source_id = arm_receipt.relative_to(root).as_posix()
    return HistoricalRun(
        arm_receipt, source_id, sha256_file(arm_receipt), arm_name,
        "development_screen", {"development.best_auroc": auroc},
        params,
        {"ecg.evaluation_stage": "development_screen",
         "ecg.result_basis": "historical_completion_json",
         "ecg.calibration_test_opened": "false", **config_tags},
    )


def _pilot_runs(root: Path, number: str) -> Iterable[HistoricalRun]:
    for directory in (root / "outputs").glob(f"experiment{number}_*"):
        parent_receipt = directory / "completion.json"
        if not parent_receipt.is_file():
            continue
        parent_data = _json(parent_receipt)
        if parent_data.get("status") != "development_pilot_complete":
            continue
        receipt_hashes = parent_data.get("arm_completions_sha256", {})
        for arm_receipt in sorted(directory.glob("*/completion.json")):
            run = _pilot_arm_run(arm_receipt, receipt_hashes, root)
            if run is not None:
                yield run


def _fusion_runs(root: Path) -> Iterable[HistoricalRun]:
    directory = root / "outputs" / "experiment014_jepa_cpc_fusion"
    for name in ("full", "ten_percent"):
        path = directory / f"{name}.json"
        if not path.is_file():
            continue
        data = _json(path)
        if data.get("completed") is not True:
            continue
        development = data.get("development", {})
        selected = str(development.get("selected_alpha"))
        selected_grid = development.get("grid", {}).get(selected, {})
        metrics = _numeric_map("development.selected", selected_grid, DEVELOPMENT_METRICS)
        gate = development.get("gate", {})
        if isinstance(gate, dict):
            metrics.update(_numeric_map("development.gate", gate,
                                        frozenset({"auroc_difference_vs_best_endpoint",
                                                   "specificity_gain_vs_best_endpoint"})))
        source_id = path.relative_to(root).as_posix()
        yield HistoricalRun(
            path, source_id, sha256_file(path), name, "development_screen", metrics,
            {"budget": name, "selected_alpha": selected},
            {"ecg.evaluation_stage": "development_screen",
             "ecg.result_basis": "historical_completed_json",
             "ecg.calibration_test_opened": str(data.get("calibration_test_opened", "unknown")).lower()},
        )


def discover_historical_runs(root: Path) -> list[HistoricalRun]:
    """
    Discover immutable historical summaries, without reading raw ECG data.

    Parameters
    ----------
    root : Path
        Project root holding ``outputs``.

    Returns
    -------
    list[HistoricalRun]
        Held-out test and development-screen results, sorted by source.

    Raises
    ------
    ValueError
        If a result file or receipt is malformed or fails its recorded hash.
    """
    root = root.resolve()
    result: list[HistoricalRun] = []
    for directory in sorted((root / "outputs").glob("experiment*")):
        if directory.name.startswith(("experiment014_", "experiment015_", "experiment017_")):
            continue
        for path in sorted(directory.glob("*/metrics.json")):
            result.append(_legacy_run(path, root))
    result.extend(_fusion_runs(root))
    result.extend(_pilot_runs(root, "015"))
    result.extend(_pilot_runs(root, "017"))
    return sorted(result, key=lambda run: run.source_id)


def _client(tracking_uri: str, client: Any) -> Any:
    if client is not None:
        return client
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("Install the uv tracking dependency group to use MLflow") from exc
    return MlflowClient(tracking_uri=tracking_uri)


def _experiment_id(client: Any, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    return experiment.experiment_id if experiment else client.create_experiment(experiment_name)


def _check_registered_values(existing: Any, record: HistoricalRun, tags: dict[str, str]) -> None:
    for kind, expected, logged in (("tag", tags, existing.data.tags),
                                   ("parameter", record.params, existing.data.params)):
        for key, value in expected.items():
            old = logged.get(key)
            if old is not None and old != value:
                raise ValueError(f"Registry {kind} conflict for {record.source_id}: {key}")
    if set(existing.data.metrics) != set(record.metrics) or any(
        existing.data.metrics[key] != value for key, value in record.metrics.items()
    ):
        raise ValueError(f"Registry metric conflict for {record.source_id}")


def _reconcile_existing_run(matches: list[Any], record: HistoricalRun, tags: dict[str, str],
                            client: Any, status: str) -> None:
    """
    Check an already registered run against the record and add missing fields.

    A previous version indexed fewer safe config fields. Additive enrichment
    is allowed; a conflicting value is never overwritten.
    """
    if any(run.data.tags.get("ecg.source_sha256") != record.source_sha256 for run in matches):
        raise ValueError(f"Source changed after registry import: {record.source_id}")
    if len(matches) != 1 or matches[0].info.status != status:
        run_ids = ", ".join(run.info.run_id for run in matches)
        raise RuntimeError(
            f"Existing registry run {run_ids} for {record.source_id} is incomplete, "
            "failed, or duplicated; inspect it before retrying"
        )
    existing = matches[0]
    _check_registered_values(existing, record, tags)
    for key, value in tags.items():
        if key not in existing.data.tags:
            client.set_tag(existing.info.run_id, key, value)
    for key, value in record.params.items():
        if key not in existing.data.params:
            client.log_param(existing.info.run_id, key, value)


def _log_aggregate_artifact(client: Any, run_id: str, record: HistoricalRun) -> None:
    # Construct this from reviewed numeric metrics. Never copy a source
    # JSON file, which may contain paths, identifiers or predictions.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "aggregate_metrics.json"
        path.write_text(json.dumps({"stage": record.stage, "metrics": record.metrics}, sort_keys=True),
                        encoding="utf-8")
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("Aggregate metrics artifact exceeds size limit")
        client.log_artifact(run_id, str(path), "aggregate")


def _create_run(record: HistoricalRun, tags: dict[str, str], client: Any, experiment_id: str, *,
                status: str, log_aggregate_artifacts: bool) -> None:
    run = client.create_run(experiment_id, tags=tags)
    try:
        for key, value in record.params.items():
            client.log_param(run.info.run_id, key, value)
        for key, value in record.metrics.items():
            client.log_metric(run.info.run_id, key, value)
        if log_aggregate_artifacts:
            _log_aggregate_artifact(client, run.info.run_id, record)
        client.set_terminated(run.info.run_id, status=status)
    except Exception:
        client.set_terminated(run.info.run_id, status="FAILED")
        raise


def _register_one(record: HistoricalRun, client: Any, experiment_id: str, *,
                  status: str, log_aggregate_artifacts: bool) -> bool:
    """Create a registry run for a new source; return False when it already exists."""
    source_key = hashlib.sha256(record.source_id.encode("utf-8")).hexdigest()
    tags = {
        "mlflow.runName": record.name,
        "ecg.import_schema": IMPORT_SCHEMA,
        "ecg.source_id": record.source_id,
        "ecg.source_key": source_key,
        "ecg.source_sha256": record.source_sha256,
        "ecg.provenance": "historical_source_receipts; current_git_revision_not_claimed",
        **record.tags,
    }
    matches = client.search_runs(
        [experiment_id], filter_string=f"tags.`ecg.source_key` = '{source_key}'",
        max_results=10,
    )
    if matches:
        _reconcile_existing_run(matches, record, tags, client, status)
    else:
        _create_run(record, tags, client, experiment_id, status=status,
                    log_aggregate_artifacts=log_aggregate_artifacts)
    return not matches


def import_historical_runs(root: Path, tracking_uri: str, *,
                           experiment_name: str = EXPERIMENT_NAME,
                           log_aggregate_artifacts: bool = False,
                           client: Any = None) -> tuple[int, int]:
    """
    Import each historical source once; reject changed or incomplete registry records.

    The caller chooses the tracking URI. MLflow is an optional, lazy dependency.

    Parameters
    ----------
    root : Path
        Project root holding ``outputs``.
    tracking_uri : str
        MLflow tracking URI.
    experiment_name : str
        Registry experiment.
    log_aggregate_artifacts : bool
        Also upload a sanitized aggregate-metrics JSON per run.
    client : Any
        MLflow client; created from ``tracking_uri`` when omitted.

    Returns
    -------
    tuple[int, int]
        Imported and already-registered run counts.

    Raises
    ------
    ValueError
        If a source changed since import or conflicts with the registry.
    RuntimeError
        If MLflow is unavailable, or an existing run is incomplete or duplicated.
    """
    client = _client(tracking_uri, client)
    experiment_id = _experiment_id(client, experiment_name)
    imported = skipped = 0
    for record in discover_historical_runs(root):
        if _register_one(record, client, experiment_id, status="FINISHED",
                         log_aggregate_artifacts=log_aggregate_artifacts):
            imported += 1
        else:
            skipped += 1
    return imported, skipped


def _check_verified_provenance(stage: str, status: str, hashes: dict[str, str]) -> None:
    if stage not in RESULT_STAGES:
        raise ValueError(f"Unknown result stage: {stage}")
    if status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status: {status}")
    if (status == "FAILED") != (stage == "execution_failed"):
        raise ValueError("Failed status must use execution_failed stage")
    for key, digest in hashes.items():
        if not SHA256.fullmatch(digest):
            raise ValueError(f"Invalid {key} SHA-256")


def _aggregate_name_valid(key: str) -> bool:
    """Aggregate names are short snake case and never per-record identifiers."""
    return bool(METRIC_KEY.fullmatch(key)) and not key.endswith("_id")


def _normalized_metrics(stage: str, metrics: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if not _aggregate_name_valid(key):
            raise ValueError(f"Invalid aggregate metric name: {key}")
        number = _metric(value)
        if number is None:
            raise ValueError(f"Metric must be finite: {key}")
        normalized[f"{stage}.{key}"] = number
    return normalized


def _normalized_params(params: dict[str, str | int | float | bool]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in params.items():
        if not _aggregate_name_valid(key):
            raise ValueError(f"Invalid parameter name: {key}")
        if not isinstance(value, (str, int, float, bool)) or len(str(value)) > MAX_VERIFIED_PARAM_CHARS:
            raise ValueError(f"Parameter must be a short scalar: {key}")
        normalized[key] = str(value)
    return normalized


def record_verified_result(root: Path, receipt: Path, tracking_uri: str, *,
                           name: str, stage: str, metrics: dict[str, float],
                           params: dict[str, str | int | float | bool],
                           manifest_sha256: str, code_sha256: str,
                           data_sha256: str,
                           status: str = "FINISHED",
                           experiment_name: str = FUTURE_EXPERIMENT_NAME,
                           client: Any = None) -> bool:
    """
    Opt in a future, frozen run after its local receipt has been written.

    Explicit frozen hashes identify the actual execution; the current checkout
    is never inferred. A failed execution may be recorded with
    ``status="FAILED"`` and no metrics.

    Parameters
    ----------
    root : Path
        Project root.
    receipt : Path
        JSON receipt under ``root / "outputs"``.
    tracking_uri : str
        MLflow tracking URI chosen by the caller.
    name : str
        Registry run name.
    stage : str
        One of ``RESULT_STAGES``.
    metrics : dict[str, float]
        Finite aggregate metrics; names are prefixed with ``stage``.
    params : dict[str, str | int | float | bool]
        Short scalar parameters.
    manifest_sha256 : str
        Digest of the executable manifest.
    code_sha256 : str
        Digest of the executed code.
    data_sha256 : str
        Digest of the input data.
    status : str
        ``"FINISHED"``, or ``"FAILED"`` with stage ``execution_failed``.
    experiment_name : str
        Registry experiment.
    client : Any
        MLflow client; created from ``tracking_uri`` when omitted.

    Returns
    -------
    bool
        False when this receipt was already registered.

    Raises
    ------
    ValueError
        If the stage, status, hashes, receipt, metric or parameter names or
        values are invalid, or the registry holds conflicting values.
    RuntimeError
        If an existing registry run is incomplete or duplicated.
    """
    _check_verified_provenance(stage, status, {"manifest": manifest_sha256, "code": code_sha256,
                                               "data": data_sha256})
    root = root.resolve()
    receipt = receipt.resolve()
    if receipt.suffix != ".json" or not receipt.is_relative_to(root / "outputs"):
        raise ValueError("Receipt must be a JSON file under the project's outputs directory")
    _json(receipt)
    normalized_metrics = _normalized_metrics(stage, metrics)
    normalized_params = _normalized_params(params)
    record = HistoricalRun(
        receipt, receipt.relative_to(root).as_posix(), sha256_file(receipt),
        name, stage, normalized_metrics, normalized_params,
        {"ecg.import_schema": "verified-result-v1",
         "ecg.evaluation_stage": stage,
         "ecg.result_basis": "explicit_frozen_provenance",
         "ecg.provenance": "explicit_frozen_hashes",
         "ecg.manifest_sha256": manifest_sha256,
         "ecg.code_sha256": code_sha256,
         "ecg.data_sha256": data_sha256},
    )
    client = _client(tracking_uri, client)
    return _register_one(record, client, _experiment_id(client, experiment_name),
                         status=status, log_aggregate_artifacts=False)
