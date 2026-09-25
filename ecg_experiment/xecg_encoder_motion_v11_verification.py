"""Transitive byte verification of frozen xECG predecessor evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment.evaluation import CALIBRATION_RECORDS, partition_validation
from ecg_experiment.files import read_csv, write_json_atomic
from ecg_experiment.xecg_encoder_motion_v11 import OUT, ROOT, SHA256, safe_leaf, stable_sha256
from scripts.experiments import run_xecg_droppath_rescue016 as v5
from scripts.experiments import run_xecg_droppath_rescue016_v6 as v6
from scripts.experiments import run_xecg_encoder_motion016_v9 as v9
from scripts.experiments import run_xecg_probe_finetune016 as base

V9_PROFILE = "outputs/experiment_queue_016_encoder_motion_v9_profile"
V9_FULL = "outputs/experiment_queue_016_encoder_motion_v9_full"
V9_OUT = "outputs/experiment016_encoder_motion_v9"
V10_OUT = "outputs/experiment016_encoder_motion_replication_v10"
ROOT_HASHES = {
    f"{V9_PROFILE}/queue.json": "9d6a2b033db352c9974adfbbf77a1b2d3cada5c5bb4107c3f5095fefddb1a676",
    f"{V9_PROFILE}/sources.json": "9cb55fc92d1930fb3d10539a157213609c17190078900b27d96d068d6eff2a09",
    f"{V9_PROFILE}/job/priority_queue_completion.json": (
        "2ac8f18986d94e8884ad74503409e678371bd6ef6551679fac9d9d4b9b7ddf51"
    ),
    f"{V9_FULL}/queue.json": "0b3b2456e94ea1a21a6bae24d720d893460aa78f1d2351406d2c47659396d45f",
    f"{V9_FULL}/sources.json": "f52010d13e4f3b03a06605a17d5ba12721e59e35df2f2e9d81d8fa64e310b405",
    f"{V9_FULL}/job/priority_queue_completion.json": (
        "29ceceace4579b00d8cc468dc84e0cef8ab2af7a743925d16dbb799d3d3044ad"
    ),
    f"{V9_OUT}/report.json": "938bc656167a01afa5cf38100a1f042a98ad9d2a843390fcb2f1f8c7f7eaf50b",
    f"{V10_OUT}/check.json": "5279da4935c79b63d78b0247acf7e49e55c16ed1e81bfdec0174aca94b51f03a",
    f"{V10_OUT}/compatibility.json": "892a7d9a8ebeeba6f15213f4b2d3bfbce10d909053c5c93a5bde9f72dea9ac67",
    "outputs/experiment_queue_016_rescue_profile_v5/sources.json": (
        "e0fa710b871695c3f43f77489d9f3769f6255d8261e728d392c37be73f1934ac"
    ),
    "data/processed/ptbxl/seed42_fraction1/test.csv": (
        "45e57546e179b58f1e1eaf04cf83a60ae5975c6a34c16f88ab64b7671566bf76"
    ),
    "data/processed/ptbxl/seed42_fraction0.1/test.csv": (
        "45e57546e179b58f1e1eaf04cf83a60ae5975c6a34c16f88ab64b7671566bf76"
    ),
    "data/processed/ptbxl/seed42_fraction0.1/labeled_train.csv": (
        "8e790dfb093e3e1f176337df1fba6ad2735ac04ec5bbcda1ed9ccfd2a29b843a"
    ),
    "data/processed/ptbxl/seed42_fraction0.1/validation.csv": (
        "347a83d308ef80d0c450979655e0e22bb7a798a2984c0d1b5de1a747383f4e8d"
    ),
    "outputs/refactor_pause/source_before_refactor.tar.gz": (
        "4911551226c3df2d2297895aef65430132a54199ce0edaac63d97b47d2a2f8ed"
    ),
}


class LeafGraph:
    """Collect every original hash obligation and read each unique leaf once."""

    def __init__(self) -> None:
        """Create an empty graph with no trusted old receipt cache."""
        self.expected: dict[str, str | None] = {}
        self.coverage: dict[str, list[str]] = defaultdict(list)
        self.resolved: set[str] = set()
        self.read_bytes = 0
        self.verified: dict[str, str] = {}
        self.edges: list[dict[str, str]] = []

    def add(self, name: str, digest: str, obligation: str) -> None:
        """Register a path and reject conflicting claimed bytes."""
        safe_leaf(name)
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError(f"Malformed expected SHA-256 for {name}")
        previous = self.expected.get(name)
        if previous is not None and previous != digest:
            raise ValueError(f"Conflicting expected SHA-256 for {name}")
        self.expected[name] = digest
        self.coverage[name].append(obligation)
        self.edges.append({"obligation": obligation, "leaf": name})

    def add_aggregate_member(self, name: str) -> None:
        """Register an unpinned leaf whose aggregate digest is pinned."""
        safe_leaf(name)
        self.expected.setdefault(name, None)
        obligation = "v9_fingerprint:vendored_xlstm_source_tree_aggregate"
        self.coverage[name].append(obligation)
        self.edges.append({"obligation": obligation, "leaf": name})

    def _expand_json(self, name: str) -> None:  # noqa: C901 - three receipt schemas are resolved here
        """Expand known source-map, queue-completion and arm-completion edges."""
        if name in self.resolved:
            return
        self.resolved.add(name)
        if not name.endswith(".json"):
            return
        path = safe_leaf(name)
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read()
            after_fd = os.fstat(handle.fileno())
        after = path.stat()
        def identity(row: Any) -> tuple[int, ...]:
            """Retain attributes that prove the parsed bytes stayed stable."""
            return (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        if identity(before) != identity(after_fd) or identity(before) != identity(after):
            raise ValueError(f"Concurrent modification: {name}")
        digest = hashlib.sha256(raw).hexdigest()
        expected = self.expected[name]
        if expected is not None and digest != expected:
            raise ValueError(f"Current bytes differ: {name}")
        self.verified[name] = digest
        self.read_bytes += len(raw)
        row = json.loads(raw)
        if path.name == "sources.json":
            for child, child_digest in row.items():
                self.add(child, child_digest, f"{name}:sources[{child}]")
        elif path.name == "priority_queue_completion.json":
            if not isinstance(row.get("artifacts"), dict):
                raise ValueError(f"Completion lacks artifacts: {name}")
            for child, child_digest in row["artifacts"].items():
                self.add(child, child_digest, f"{name}:artifacts[{child}]")
        elif path.name == "complete.json" and isinstance(row.get("sha256"), dict):
            for relative, child_digest in row["sha256"].items():
                child = str((Path(name).parent / relative).as_posix())
                self.add(child, child_digest, f"{name}:sha256[{relative}]")

    def expand(self) -> None:
        """Resolve all transitive references reachable from the pinned roots."""
        while pending := set(self.expected) - self.resolved:
            for name in sorted(pending):
                self._expand_json(name)

    def verify(self) -> dict[str, Any]:
        """Read every required current file exactly once with identity checks."""
        started = time.monotonic()
        self.expand()
        for name, expected in sorted(self.expected.items()):
            if name in self.verified:
                continue
            digest, size = stable_sha256(safe_leaf(name))
            self.read_bytes += size
            if expected is not None and digest != expected:
                raise ValueError(f"Current bytes differ: {name}")
            self.verified[name] = digest
        return {
            "status": "byte_closure_verified",
            "unique_leaf_count": len(self.verified),
            "logical_reference_count": len(self.edges),
            "bytes_read": self.read_bytes,
            "seconds": time.monotonic() - started,
            "coverage": dict(sorted(self.coverage.items())),
            "resolved_edges": self.edges,
            "verified_leaves": self.verified,
        }


def _require_v9_semantics() -> dict[str, Any]:
    """Check frozen completion state and actual full-profile cost evidence."""
    for directory in (V9_PROFILE, V9_FULL):
        manifest = json.loads((ROOT / directory / "queue.json").read_text())
        status = json.loads((ROOT / directory / "status.json").read_text())
        completion = json.loads((ROOT / directory / "job/priority_queue_completion.json").read_text())
        digest = ROOT_HASHES[f"{directory}/queue.json"]
        if status.get("state") != "complete" or status.get("returncode") != 0:
            raise ValueError(f"V9 queue did not complete: {directory}")
        if completion["queue_manifest_sha256"] != digest or not manifest["jobs"]:
            raise ValueError(f"V9 queue completion identity differs: {directory}")
    profile = json.loads((ROOT / "outputs/experiment016_encoder_motion_v9/profile.json").read_text())
    gate = json.loads((ROOT / "outputs/experiment016_encoder_motion_v9/cost_gate.json").read_text())
    if (
        gate.get("passed") is not True
        or gate.get("P_seconds") != 1098.4113020410005
        or gate.get("measured_preparation_plus_both_profile_pipelines_seconds") != 2693.7965717150364
        or not all(profile["correctness"].values())
    ):
        raise ValueError("V9 full-pipeline profile/cost evidence differs")
    for arm in ("M", "F"):
        complete_path = ROOT / V9_OUT / "profile" / arm / "complete.json"
        complete = json.loads(complete_path.read_text())
        if complete.get("updates") != 240 or complete.get("record_exposures") != 15359:
            raise ValueError(f"V9 profile {arm} was not a complete full-data pass")
    return {
        "profile_and_full_queue_complete": True,
        "P_seconds": gate["P_seconds"],
        "H_seconds": gate["measured_preparation_plus_both_profile_pipelines_seconds"],
    }


def _vendor_tree(graph: LeafGraph, paths: list[Path]) -> dict[str, Any]:
    """Verify the exact old vendored-tree aggregate from all current Python leaves."""
    root = ROOT / "third_party/xecg-deps/xlstm"
    digest = hashlib.sha256()
    if not paths:
        raise ValueError("Missing vendored xLSTM source tree")
    for path in paths:
        relative = str(path.relative_to(ROOT))
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(graph.verified[relative]))
    expected = "f56163a8cb1457a8f2124063aa2c7ce096d3aa2b82540ceaa9314e045c19bc5f"
    if digest.hexdigest() != expected:
        raise ValueError("Vendored xLSTM source tree differs from v9 fingerprint")
    return {"aggregate_sha256": digest.hexdigest(), "python_file_count": len(paths)}


def _verified(graph: LeafGraph, name: str) -> str:
    """Return a current-pass digest without causing another byte scan."""
    try:
        return graph.verified[name]
    except KeyError as exc:
        raise ValueError(f"Unresolved required verified leaf: {name}") from exc


def _semantic_inputs(graph: LeafGraph, vendor: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
    """Reconstruct the old v9 input fingerprint with its original assertions."""
    manifest = ROOT / "data/processed/ptbxl"
    full_dir = manifest / "seed42_fraction1"
    small_dir = manifest / "seed42_fraction0.1"
    rows = {name: read_csv(full_dir / f"{name}.csv") for name in base.SPLITS}
    small_train = read_csv(small_dir / "labeled_train.csv")
    base._check_label_budgets(rows["labeled_train"], small_train)
    for name in ("validation", "test"):
        if rows[name] != read_csv(small_dir / f"{name}.csv"):
            raise ValueError(f"Frozen {name} differs across label budgets")
    base._check_partitions(rows)
    development, calibration = partition_validation(rows["validation"])
    if len(development) != 1306 or len(calibration) != CALIBRATION_RECORDS:
        raise ValueError("Development/calibration split differs")
    if len(rows["labeled_train"]) != 15360 or {r["target"] for r in rows["labeled_train"]} != {"0", "1"}:
        raise ValueError("Full-label training manifest differs")
    manifest_hashes = {
        f"{name}.csv": _verified(graph, f"data/processed/ptbxl/seed42_fraction1/{name}.csv")
        for name in base.SPLITS
    }
    manifest_hashes["full_labeled_train.csv"] = manifest_hashes["labeled_train.csv"]
    manifest_hashes["ten_percent_labeled_train.csv"] = _verified(
        graph, "data/processed/ptbxl/seed42_fraction0.1/labeled_train.csv"
    )
    clean_dir = ROOT / "outputs/data_quality/clean_rerun_preflight_v1"
    overlay = read_csv(clean_dir / "labels_fraction1.csv")
    expected_clean = {row["record_id"]: row for row in overlay}
    train = [row for row in rows["labeled_train"] if f"ptbxl:{row['ecg_id']}" in expected_clean]
    if len(train) != 15359 or len(expected_clean) != 15359:
        raise ValueError("Clean cohort differs from frozen 15,359")
    excluded = {row["ecg_id"] for row in rows["labeled_train"]} - {row["ecg_id"] for row in train}
    if excluded != {"12722"}:
        raise ValueError("Clean exclusion differs")
    if any(
        expected_clean[f"ptbxl:{row['ecg_id']}"]["patient_id"] != f"ptbxl:{row['patient_id']}"
        or expected_clean[f"ptbxl:{row['ecg_id']}"]["target"] != row["target"]
        for row in train
    ):
        raise ValueError("Clean patient or label identity differs")
    clean_labels = "outputs/data_quality/clean_rerun_preflight_v1/labels_fraction1.csv"
    clean_receipt = "outputs/data_quality/clean_rerun_preflight_v1/receipt.json"
    clean = json.loads((ROOT / clean_receipt).read_text())
    if clean["output_sha256"]["labels_fraction1.csv"] != _verified(graph, clean_labels):
        raise ValueError("Clean label receipt differs")

    cache_dir = ROOT / "data/processed/ptbxl/xecg_views"
    metadata_name = "data/processed/ptbxl/xecg_views/metadata.json"
    views_name = "data/processed/ptbxl/xecg_views/views.npy"
    metadata = json.loads((ROOT / metadata_name).read_text())
    ids = [int(ecg_id) for ecg_id in metadata["ecg_ids"]]
    views = np.load(cache_dir / "views.npy", mmap_mode="r")
    if views.shape != (len(ids), *base.XECG_VIEW_SHAPE) or views.dtype != np.float32:
        raise ValueError("Malformed 100 Hz physical-mV cache")
    if len(ids) != base.CACHE_RECORDS or len(set(ids)) != len(ids):
        raise ValueError("Cache ECG identity/count differs")
    if metadata.get("shape") != list(views.shape) or metadata.get("dtype") != "float32":
        raise ValueError("Cache metadata shape/dtype differs")
    if metadata.get("raw_dir") != str(v5.RAW.resolve()) or metadata.get("manifest_sha256") != {
        name: manifest_hashes[name] for name in ("labeled_train.csv", "validation.csv", "test.csv")
    }:
        raise ValueError("Cache raw or full-manifest provenance differs")
    if metadata.get("views_sha256") != _verified(graph, views_name):
        raise ValueError("Cache waveform digest differs")
    old_sources = (
        "4a16e3094b6bc68bacd382607d83f8a683b225a9dd810aab00b83318cda266b9",
        "d8a56db66854262c93ffbc557eb3738c90bd29258f1bac4f6a074b809974ff6f",
    )
    source_pair = (metadata.get("preparation_sha256"), metadata.get("adapter_sha256"))
    current_pair = (
        stable_sha256(safe_leaf("scripts/data/prepare_xecg.py"))[0],
        _verified(graph, "ecg_experiment/xecg.py"),
    )
    if source_pair != current_pair:
        refactor = json.loads((ROOT / "outputs/repository_refactor/verification.json").read_text())
        pause = json.loads((ROOT / "outputs/refactor_pause/pause.json").read_text())
        archive_sha = _verified(graph, "outputs/refactor_pause/source_before_refactor.tar.gz")
        if (
            source_pair != old_sources
            or refactor.get("status") != "verified"
            or refactor.get("source_archive_sha256") != archive_sha
            or pause.get("source_archive_sha256") != archive_sha
        ):
            raise ValueError("Cache preparation source provenance differs")
    index = {ecg_id: position for position, ecg_id in enumerate(ids)}
    if any(int(row["ecg_id"]) not in index for row in train + development):
        raise ValueError("Clean ECG missing from cache")

    feature_dir = ROOT / "outputs/experiment016_xecg_probe_finetune/features"
    feature_base = "outputs/experiment016_xecg_probe_finetune/features"
    feature_receipt = json.loads((feature_dir / "receipt.json").read_text())
    feature_hashes = {
        name: _verified(graph, f"{feature_base}/{name}") for name in ("features.npy", "ecg_ids.npy")
    }
    if feature_receipt["sha256"] != feature_hashes:
        raise ValueError("Released feature cache digest differs")
    release_base = "third_party/checkpoints/xecg"
    release_hashes = {name: _verified(graph, f"{release_base}/{name}") for name in base.CHECKPOINT_FILES}
    cache_hashes = {
        "metadata_sha256": _verified(graph, metadata_name),
        "views_sha256": _verified(graph, views_name),
    }
    if (
        feature_receipt["fingerprint"]["checkpoint_sha256"] != release_hashes
        or feature_receipt["fingerprint"]["cache_sha256"] != cache_hashes
    ):
        raise ValueError("Released feature provenance differs")
    features = np.load(feature_dir / "features.npy", mmap_mode="r")
    feature_ids = np.load(feature_dir / "ecg_ids.npy")
    ordered_ids = np.array([int(row["ecg_id"]) for row in rows["labeled_train"] + development])
    if features.shape != (len(ordered_ids), 1024) or not np.array_equal(feature_ids, ordered_ids):
        raise ValueError("Released features row identity differs")
    selected = np.array([i for i, row in enumerate(rows["labeled_train"]) if row["ecg_id"] != "12722"])
    clean_features = np.asarray(features[selected], dtype=np.float32)
    development_features = np.asarray(features[len(rows["labeled_train"]) :], dtype=np.float32)
    if not np.isfinite(clean_features).all() or not np.isfinite(development_features).all():
        raise ValueError("Nonfinite released features")
    v5_fingerprint = {
        "sources": {name: _verified(graph, name) for name in v5.SOURCE_FILES},
        "manifest": manifest_hashes,
        "clean_labels_sha256": _verified(graph, clean_labels),
        "clean_receipt_sha256": _verified(graph, clean_receipt),
        "feature_receipt_sha256": _verified(graph, f"{feature_base}/receipt.json"),
        "feature_hashes": feature_hashes,
        "cache_hashes": cache_hashes,
        "release_hashes": release_hashes,
        "xlstm_source_sha256": vendor["aggregate_sha256"],
        "environment": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "xlstm": "2.0.4 (vendored source tree hashed above)",
            "uv_lock_sha256": _verified(graph, "uv.lock"),
            "pyproject_sha256": _verified(graph, "pyproject.toml"),
        },
        "train_records": len(train),
        "development_records": len(development),
        "excluded_id": "12722",
        "seed": 42,
        "epochs": 2,
    }
    v5_dir = ROOT / "outputs/experiment016_droppath_rescue_v5"
    v5_rows = {
        name: json.loads((v5_dir / name).read_text()) for name in v6.V5_RECEIPTS if name.endswith(".json")
    }
    v5_identity_receipts = ("check.json", "probe.json", "diagnostic.json", "profile.json")
    if not all(v5_rows[name]["fingerprint"] == v5_fingerprint for name in v5_identity_receipts):
        raise ValueError("V5 evidence fingerprint differs")
    v5_probe_sha = _verified(graph, "outputs/experiment016_droppath_rescue_v5/probe.npz")
    if v5_rows["probe.json"]["sha256"] != v5_probe_sha:
        raise ValueError("V5 probe receipt differs")
    if v5_rows["cost_gate.json"]["passed"]:
        raise ValueError("V5 rejected two-epoch gate changed")
    for arm in v6.ARMS:
        row = v5_rows["profile.json"]["arms"][arm]
        if (
            row["epoch"]["updates"] != v6.UPDATES
            or row["epoch"]["seen"] != 15359
            or not row["model_optimizer_scheduler_rng_roundtrip"]
            or not row["resumed_next_update_same_device"]
        ):
            raise ValueError(f"V5 full-profile correctness differs: {arm}")
    v5_status = json.loads((ROOT / "outputs/experiment_queue_016_rescue_profile_v5/status.json").read_text())
    if v5_status["state"] != "failed" or "stage profile exited 1" not in v5_status["reason"]:
        raise ValueError("V5 predecessor state differs")
    v6_fingerprint = {
        "v5": v5_fingerprint,
        "v6_sources": {name: _verified(graph, name) for name in v6.V6_SOURCES},
        "optimization_seed": v6.SEED,
        "execution_epochs": v6.EXECUTION_EPOCHS,
        "scheduler_horizon_epochs": v6.SCHEDULER_HORIZON_EPOCHS,
        "updates": v6.UPDATES,
    }
    v9_fingerprint = {
        "v6_cohort": v6_fingerprint,
        "v8_source_map_sha256": _verified(graph, str(v9.V8_MAP.relative_to(ROOT))),
        "v8_completion_sha256": _verified(graph, str(v9.V8_COMPLETION.relative_to(ROOT))),
        "v8_report_sha256": _verified(graph, "outputs/experiment016_head_mechanism_v8/report.json"),
        "v6_probe_sha256": v5_probe_sha,
        "v7_reference_logits_sha256": _verified(
            graph, "outputs/experiment016_frozen_readout_audit_v7/released_refit_logits.npy"
        ),
        "v9_sources": {name: _verified(graph, name) for name in v9.V9_SOURCES},
        "environment": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "uv_lock_sha256": _verified(graph, "uv.lock"),
        },
    }
    historical = json.loads((ROOT / V10_OUT / "check.json").read_text())["fingerprint"]["v9"]
    if v9_fingerprint != historical:
        raise ValueError("Reconstructed v9 input fingerprint differs from pinned v10 check")
    return {
        "v9_fingerprint_matches_pinned_v10": True,
        "train_records": len(train),
        "development_records": len(development),
        "development_patients": len({row["patient_id"] for row in development}),
        "calibration_test_endpoint_closed": True,
        "assertion_groups": [
            "v5:fixed_full_and_ten_percent_manifest_identities_and_patient_partitions",
            "v5:clean_overlay_15359_rows_and_excluded_12722",
            "v5:100Hz_physical_mV_cache_order_shape_preprocessing_and_source_link",
            "v5:released_weights_features_probe_and_train_only_row_alignment",
            "v6:full_profile_and_rejected_two_epoch_cost_evidence",
            "v9:cohort_size_and_nested_fingerprint_identity",
        ],
    }


def verify_historical_closure() -> dict[str, Any]:
    """Rehash predecessor leaves and record semantic assertions and coverage."""
    started = time.monotonic()
    graph = LeafGraph()
    for name, digest in ROOT_HASHES.items():
        graph.add(name, digest, "v10_pinned_root")
    vendor_paths = sorted((ROOT / "third_party/xecg-deps/xlstm").rglob("*.py"))
    for path in vendor_paths:
        graph.add_aggregate_member(str(path.relative_to(ROOT)))
    receipt = graph.verify()
    receipt["v9_semantics"] = _require_v9_semantics()
    receipt["vendor_tree"] = _vendor_tree(graph, vendor_paths)
    receipt["input_semantics"] = _semantic_inputs(graph, receipt["vendor_tree"])
    receipt["seconds"] = time.monotonic() - started
    write_json_atomic(OUT / "verification.json", receipt)
    return receipt
