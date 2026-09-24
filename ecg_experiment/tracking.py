"""Read-only historical result import for an optional MLflow registry.

This module never starts training and never changes an existing result file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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


@dataclass(frozen=True)
class HistoricalRun:
    source: Path
    source_id: str
    source_sha256: str
    name: str
    stage: str
    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]
    artifacts: tuple[Path, ...]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _numeric_map(prefix: str, data: dict[str, Any], allowed: frozenset[str]) -> dict[str, float]:
    return {f"{prefix}.{key}": number for key, value in data.items()
            if key in allowed and (number := _metric(value)) is not None}


def _receipt_for_metrics(path: Path) -> Path | None:
    for name in ("complete.json", "completion.json"):
        receipt = path.with_name(name)
        if receipt.is_file():
            data = _json(receipt)
            hashes = data.get("sha256", data.get("artifacts", {}))
            if isinstance(hashes, dict) and "metrics.json" in hashes:
                if hashes["metrics.json"] != _digest(path):
                    raise ValueError(f"Receipt hash mismatch: {path}")
            return receipt
    return None


def _config_metadata(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read only a small allowlist of aggregate config scalars and hashes."""
    if not path.is_file():
        return {}, {}
    config = _json(path)
    fingerprint = config.get("fingerprint")
    params = {}
    tags = {"ecg.config_sha256": _digest(path)}
    for key in CONFIG_SCALARS:
        value = config.get(key)
        if value is None and isinstance(fingerprint, dict):
            value = fingerprint.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            text = str(value)
            if len(text) <= 128 and (not isinstance(value, float) or math.isfinite(value)):
                params[key] = text
    records = config.get("records")
    if "labeled_training_records" not in params and isinstance(records, dict):
        count = records.get("labeled_train", records.get("train"))
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            params["labeled_training_records"] = str(count)
    if isinstance(fingerprint, str) and SHA256.fullmatch(fingerprint):
        tags["ecg.historical_fingerprint_sha256"] = fingerprint
    for key in SOURCE_HASH_FIELDS:
        value = config.get(key)
        if value is None and isinstance(fingerprint, dict):
            value = fingerprint.get(key)
        if isinstance(value, str) and SHA256.fullmatch(value):
            tags[f"ecg.{key}"] = value
    return params, tags


def _legacy_run(path: Path, root: Path) -> HistoricalRun:
    data = _json(path)
    test = data.get("test")
    if not isinstance(test, dict):
        raise ValueError(f"Missing aggregate test metrics: {path}")
    receipt = _receipt_for_metrics(path)
    metrics = _numeric_map("test", test, TEST_METRICS)
    ci = data.get("test_ci95_patient_bootstrap", {})
    if isinstance(ci, dict):
        for name, bounds in ci.items():
            if name not in TEST_METRICS:
                continue
            if isinstance(bounds, list) and len(bounds) == 2:
                for label, value in zip(("lower", "upper"), bounds):
                    if (number := _metric(value)) is not None:
                        metrics[f"test.ci95.{name}.{label}"] = number
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
    return HistoricalRun(path, source_id, _digest(path), path.parent.name,
                         "held_out_test", metrics, params, tags,
                         (path, receipt) if receipt else (path,))


def _pilot_runs(root: Path, number: str) -> Iterable[HistoricalRun]:
    directories = list((root / "outputs").glob(f"experiment{number}_*"))
    for directory in directories:
        parent_receipt = directory / "completion.json"
        if not parent_receipt.is_file():
            continue
        parent_data = _json(parent_receipt)
        if parent_data.get("status") != "development_pilot_complete":
            continue
        receipt_hashes = parent_data.get("arm_completions_sha256", {})
        for arm_receipt in sorted(directory.glob("*/completion.json")):
            arm_name = arm_receipt.parent.name
            if isinstance(receipt_hashes, dict) and arm_name in receipt_hashes:
                if receipt_hashes[arm_name] != _digest(arm_receipt):
                    raise ValueError(f"Parent receipt hash mismatch: {arm_receipt}")
            arm = _json(arm_receipt)
            if "best_development_auroc" not in arm:
                continue
            number_value = _metric(arm["best_development_auroc"])
            if number_value is None:
                raise ValueError(f"Invalid development AUROC: {arm_receipt}")
            config_path = arm_receipt.with_name("config.json")
            config = _json(config_path) if config_path.is_file() else {}
            config_params, config_tags = _config_metadata(config_path)
            params = {"arm": arm_name, **config_params}
            for key in ("model_parameters", "inference_parameters", "exposed_training_labels"):
                if key in config:
                    params[key] = str(config[key])
            if "best_epoch" in arm:
                params["best_epoch"] = str(arm["best_epoch"])
            if "fingerprint" in arm:
                params["fingerprint"] = str(arm["fingerprint"])
            source_id = arm_receipt.relative_to(root).as_posix()
            yield HistoricalRun(
                arm_receipt, source_id, _digest(arm_receipt), arm_name,
                "development_screen", {"development.best_auroc": number_value},
                params,
                {"ecg.evaluation_stage": "development_screen",
                 "ecg.result_basis": "historical_completion_json",
                 "ecg.calibration_test_opened": "false", **config_tags},
                (arm_receipt,),
            )


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
            path, source_id, _digest(path), name, "development_screen", metrics,
            {"budget": name, "selected_alpha": selected},
            {"ecg.evaluation_stage": "development_screen",
             "ecg.result_basis": "historical_completed_json",
             "ecg.calibration_test_opened": str(data.get("calibration_test_opened", "unknown")).lower()},
            (),  # Fusion files contain split metadata; never upload them as artifacts.
        )


def discover_historical_runs(root: Path) -> list[HistoricalRun]:
    """Discover immutable historical summaries, without reading raw ECG data."""
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


def _register_one(record: HistoricalRun, client: Any, experiment_id: str, *,
                  status: str, log_aggregate_artifacts: bool) -> bool:
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
        if any(run.data.tags.get("ecg.source_sha256") != record.source_sha256
               for run in matches):
            raise ValueError(f"Source changed after registry import: {record.source_id}")
        if len(matches) != 1 or matches[0].info.status != status:
            run_ids = ", ".join(run.info.run_id for run in matches)
            raise RuntimeError(
                f"Existing registry run {run_ids} for {record.source_id} is incomplete, "
                "failed, or duplicated; inspect it before retrying"
            )
        existing = matches[0]
        for key, value in tags.items():
            old = existing.data.tags.get(key)
            if old is not None and old != value:
                raise ValueError(f"Registry tag conflict for {record.source_id}: {key}")
        for key, value in record.params.items():
            old = existing.data.params.get(key)
            if old is not None and old != value:
                raise ValueError(f"Registry parameter conflict for {record.source_id}: {key}")
        if set(existing.data.metrics) != set(record.metrics) or any(
            existing.data.metrics[key] != value for key, value in record.metrics.items()
        ):
            raise ValueError(f"Registry metric conflict for {record.source_id}")
        # A previous version indexed fewer safe config fields. Additive
        # enrichment is allowed; never overwrite a conflicting value.
        for key, value in tags.items():
            if key not in existing.data.tags:
                client.set_tag(existing.info.run_id, key, value)
        for key, value in record.params.items():
            if key not in existing.data.params:
                client.log_param(existing.info.run_id, key, value)
        return False
    run = client.create_run(experiment_id, tags=tags)
    try:
        for key, value in record.params.items():
            client.log_param(run.info.run_id, key, value)
        for key, value in record.metrics.items():
            client.log_metric(run.info.run_id, key, value)
        if log_aggregate_artifacts:
            # Construct this from reviewed numeric metrics. Never copy a source
            # JSON file, which may contain paths, identifiers or predictions.
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "aggregate_metrics.json"
                path.write_text(json.dumps({"stage": record.stage,
                                            "metrics": record.metrics}, sort_keys=True),
                                encoding="utf-8")
                if path.stat().st_size > MAX_ARTIFACT_BYTES:
                    raise ValueError("Aggregate metrics artifact exceeds size limit")
                client.log_artifact(run.info.run_id, str(path), "aggregate")
        client.set_terminated(run.info.run_id, status=status)
    except Exception:
        client.set_terminated(run.info.run_id, status="FAILED")
        raise
    return True


def import_historical_runs(root: Path, tracking_uri: str, *,
                           experiment_name: str = EXPERIMENT_NAME,
                           log_aggregate_artifacts: bool = False,
                           client: Any = None) -> tuple[int, int]:
    """Import once per source; reject changed or incomplete registry records.

    The caller chooses the tracking URI. MLflow is an optional, lazy dependency.
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


def record_verified_result(root: Path, receipt: Path, tracking_uri: str, *,
                           name: str, stage: str, metrics: dict[str, float],
                           params: dict[str, str | int | float | bool],
                           manifest_sha256: str, code_sha256: str,
                           data_sha256: str,
                           status: str = "FINISHED",
                           experiment_name: str = FUTURE_EXPERIMENT_NAME,
                           client: Any = None) -> bool:
    """Opt in a future, frozen run after its local receipt has been written.

    Explicit frozen hashes identify the actual execution. The current checkout
    is never inferred. Returns False when this receipt was already registered.
    A failed execution may be recorded with status='FAILED' and no metrics.
    """
    if stage not in {"development_screen", "calibration", "held_out_test", "execution_failed"}:
        raise ValueError(f"Unknown result stage: {stage}")
    if status not in {"FINISHED", "FAILED"}:
        raise ValueError(f"Unknown run status: {status}")
    if (status == "FAILED") != (stage == "execution_failed"):
        raise ValueError("Failed status must use execution_failed stage")
    for key, digest in {"manifest": manifest_sha256, "code": code_sha256,
                        "data": data_sha256}.items():
        if not SHA256.fullmatch(digest):
            raise ValueError(f"Invalid {key} SHA-256")
    root = root.resolve()
    receipt = receipt.resolve()
    if receipt.suffix != ".json" or not receipt.is_relative_to(root / "outputs"):
        raise ValueError("Receipt must be a JSON file under the project's outputs directory")
    _json(receipt)
    normalized_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if not METRIC_KEY.fullmatch(key) or key.endswith("_id"):
            raise ValueError(f"Invalid aggregate metric name: {key}")
        number = _metric(value)
        if number is None:
            raise ValueError(f"Metric must be finite: {key}")
        normalized_metrics[f"{stage}.{key}"] = number
    normalized_params: dict[str, str] = {}
    for key, value in params.items():
        if not METRIC_KEY.fullmatch(key) or key.endswith("_id"):
            raise ValueError(f"Invalid parameter name: {key}")
        if not isinstance(value, (str, int, float, bool)) or len(str(value)) > 256:
            raise ValueError(f"Parameter must be a short scalar: {key}")
        normalized_params[key] = str(value)
    record = HistoricalRun(
        receipt, receipt.relative_to(root).as_posix(), _digest(receipt),
        name, stage, normalized_metrics, normalized_params,
        {"ecg.import_schema": "verified-result-v1",
         "ecg.evaluation_stage": stage,
         "ecg.result_basis": "explicit_frozen_provenance",
         "ecg.provenance": "explicit_frozen_hashes",
         "ecg.manifest_sha256": manifest_sha256,
         "ecg.code_sha256": code_sha256,
         "ecg.data_sha256": data_sha256},
        (),
    )
    client = _client(tracking_uri, client)
    return _register_one(record, client, _experiment_id(client, experiment_name),
                         status=status, log_aggregate_artifacts=False)
