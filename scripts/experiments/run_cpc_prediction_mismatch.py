#!/usr/bin/env python3
"""Experiment 009: frozen CPC prediction mismatch versus matched local features."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from ecg_experiment.cpc_prediction_mismatch import (
    ARMS, BRANCH_WIDTH, FIRST_QUERY, FIRST_TARGET, HORIZON,
    extract_branches, features_for_arm, load_bootstrap, supervised_indices,
)
from scripts.experiments import run_cpc_experiment as base


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/experiment009_cpc_prediction_mismatch"
DEFAULT_BOOTSTRAP = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl"
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
BUDGETS = {"full": ("1", 15360), "ten_percent": ("0.1", 1518)}
FEATURE_FIELDS = ("ecg_id", "patient_id", "source", "split")
ARTIFACTS = ("config.json", "linear_model.npz", "selection.json", "metrics.json",
             "test_predictions.csv", "calibration_predictions.npz")


def source_hashes():
    files = ("ecg_experiment/cpc_prediction_mismatch.py", "scripts/experiments/run_cpc_prediction_mismatch.py",
             "ecg_experiment/cpc.py", "scripts/experiments/run_cpc_experiment.py",
             "ecg_experiment/run.py", "ecg_experiment/evaluation.py", "ecg_experiment/data.py")
    return {name: base.digest_file(ROOT / name) for name in files}


def read_rows(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames


def load_manifests(pool, root):
    manifests, hashes, headers = {}, {}, {}
    for budget, (fraction, count) in BUDGETS.items():
        rows, fingerprints = base.manifest_rows(pool, root, fraction)
        if len(rows["labeled_train"]) != count or len(rows["validation"]) != 1870 or len(rows["test"]) != 1896:
            raise ValueError("Supervised manifests differ from the fixed PTB label budgets")
        if {row["target"] for row in rows["labeled_train"]} != {"0", "1"}:
            raise ValueError("Training manifests must contain both binary classes")
        all_rows = [row for name in ("labeled_train", "validation", "test") for row in rows[name]]
        if len({row["ecg_id"] for row in all_rows}) != len(all_rows):
            raise ValueError("Duplicate ECGs within/across supervised splits")
        patients = [{row["patient_id"] for row in rows[name]} for name in ("all_train_ssl", "validation", "test")]
        if any(patients[i] & patients[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Training, validation and test patients overlap")
        development, calibration = base.partition_validation(rows["validation"])
        if (len(development), len(calibration)) != (1306, 564):
            raise ValueError("Development/calibration partition differs from the fixed protocol")
        manifests[budget] = {**rows, "development": development, "calibration": calibration}
        hashes[budget] = fingerprints
        headers[budget] = {name: read_rows(Path(root) / f"seed42_fraction{fraction}" / f"{name}.csv")[1]
                           for name in ("labeled_train", "validation", "test")}
    full = {row["ecg_id"]: row for row in manifests["full"]["labeled_train"]}
    if any(full.get(row["ecg_id"]) != row for row in manifests["ten_percent"]["labeled_train"]):
        raise ValueError("The 10% labels are not the fixed subset of the full labels")
    for name in ("validation", "test", "all_train_ssl"):
        if manifests["full"][name] != manifests["ten_percent"][name]:
            raise ValueError("Label budgets use different patients or waveforms")
    features = [{"ecg_id": row["ecg_id"], "patient_id": row["patient_id"], "source": "ptbxl", "split": split}
                for name, split in (("labeled_train", "train"), ("validation", "validation"), ("test", "test"))
                for row in manifests["full"][name]]
    if len(features) != 19126:
        raise ValueError("Unexpected extraction cohort")
    return manifests, features, hashes, headers


def prepare_inputs(args):
    pool = base.Pool(args.cache_dir)
    data_hashes = base.make_source_hashes(pool, args.manifest_root)
    model, config = load_bootstrap(args.bootstrap_dir)
    if base.digest_json(config["inputs"]) != config["fingerprint"]:
        raise ValueError("Original CPC configuration fingerprint is invalid")
    for path, expected in config["inputs"]["cache"].items():
        if path in data_hashes and data_hashes[path] != expected:
            raise ValueError(f"CPC checkpoint source identity changed: {path}")
    for name, expected in config["inputs"]["code"].items():
        if base.digest_file(ROOT / name) != expected:
            raise ValueError(f"Frozen CPC implementation changed: {name}")
    normalization_path = args.bootstrap_dir.parent / "normalization.json"
    normalization = json.loads(normalization_path.read_text())
    expected_source = {"train_ids_sha256": base.digest_json([row["ecg_id"] for row in pool.train_rows]),
                       "rows_sha256": base.digest_file(pool.directory / "rows.csv"),
                       "signals_sha256": data_hashes[str((pool.directory / "signals.npy").resolve())],
                       "method": "global per-lead mean and population std, training waveforms only"}
    if normalization.get("source") != expected_source or normalization.get("count") != len(pool.train_rows) * 2500:
        raise ValueError("Frozen CPC normalization was fitted on different training waveforms")
    mean, std = (np.asarray(normalization[key], dtype=np.float32) for key in ("mean", "std"))
    if mean.shape != (12,) or std.shape != (12,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid frozen normalization")
    if any(not np.array_equal(value, np.asarray(config["inputs"]["normalization"][key], dtype=np.float32))
           for key, value in (("mean", mean), ("std", std))):
        raise ValueError("Normalization does not match the pretrained CPC input statistics")
    manifests, rows, manifest_hashes, headers = load_manifests(pool, args.manifest_root)
    checkpoint_hashes = {name: base.digest_file(args.bootstrap_dir / name)
                         for name in ("encoder.pt", "epoch_state.pt", "config.json")}
    identity = {"experiment": 9, "sources": source_hashes(), "checkpoint": checkpoint_hashes,
                "checkpoint_fingerprint": config["fingerprint"], "data": data_hashes,
                "normalization_sha256": base.digest_file(normalization_path),
                "manifests": manifest_hashes, "manifest_headers": headers,
                "feature_rows_sha256": base.digest_json(rows), "feature_header": list(FEATURE_FIELDS),
                "token_selection": {"horizon": HORIZON, "first_query": FIRST_QUERY, "first_target": FIRST_TARGET,
                                    "half_tokens": 79, "retained_tokens_per_half": 72},
                "pooling": "Per-half mean/max, then mean across halves; same retained positions in every branch",
                "residual": "normalize(z_t) - normalize(head_4(h_(t-4))); preserve difference magnitude",
                "torch_version": str(torch.__version__), "numpy_version": np.__version__}
    return pool, model, mean, std, manifests, rows, identity


@contextmanager
def gpu_lock(device):
    if device != "cuda":
        yield
        return
    with base.GPU_LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another queued experiment owns the GPU") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def normalized_batch(pool, rows, mean, std, device):
    values = np.array(pool.signals[pool.indices(rows)], dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite input in the unchanged extraction cohort")
    values -= mean[None, :, None]
    values /= std[None, :, None]
    return torch.from_numpy(values).to(device)


def extraction_fingerprint(args, identity):
    return {"inputs": identity, "device": args.device, "batch_size": args.batch_size,
            "threads": args.threads, "precision": "float32"}


def profile(args, pool, model, mean, std, rows, identity):
    fingerprint = extraction_fingerprint(args, identity)
    model = model.to(args.device)
    count = min(args.batch_size, len(rows))
    timings = []
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        started = time.monotonic()
        values = normalized_batch(pool, rows[:count], mean, std, args.device)
        features = extract_branches(model, values).cpu().numpy()
        if args.device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.monotonic() - started)
    result = {"fingerprint": fingerprint, "batch_records": count, "branch_shape": list(features.shape),
              "warmup_seconds": timings[0], "mean_measured_batch_seconds": float(np.mean(timings[1:])),
              "projected_extraction_seconds": float(np.mean(timings[1:])) * math.ceil(len(rows) / count),
              "feature_bytes": len(rows) * 3 * BRANCH_WIDTH * 4,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
              "head_parameters": sum(p.numel() for p in model.heads[0].parameters()),
              "finite": bool(np.isfinite(features).all())}
    base.atomic_json(args.output_dir / "profile.json", result)
    print(json.dumps({"stage": "mismatch_profile", **{k: v for k, v in result.items() if k != "fingerprint"}}), flush=True)
    return result


def feature_completion(directory, fingerprint):
    path = directory / "metadata.json"
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text())
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("Completed mismatch features have different input identities")
    if metadata["sha256"] != {name: base.digest_file(directory / name) for name in ("features.npy", "rows.csv")}:
        raise ValueError("Frozen feature cache checksum mismatch")
    return metadata


def write_feature_rows(path, rows):
    if path.exists():
        saved, fields = read_rows(path)
        if saved != rows or fields != list(FEATURE_FIELDS):
            raise ValueError("Existing feature row identities changed")
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def extract(args, pool, model, mean, std, rows, identity):
    directory = args.output_dir / "features"
    fingerprint = extraction_fingerprint(args, identity)
    if feature_completion(directory, fingerprint):
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_feature_rows(directory / "rows.csv", rows)
    partial, final = directory / "features.partial.npy", directory / "features.npy"
    progress_path = directory / "progress.json"
    chunks, done, previous = [], 0, 0.0
    if progress_path.exists():
        if not args.resume:
            raise FileExistsError("Interrupted feature extraction exists; use --resume")
        progress = json.loads(progress_path.read_text())
        if progress["fingerprint"] != fingerprint:
            raise ValueError("Partial feature extraction fingerprint differs")
        done, chunks, previous = progress["completed_rows"], progress["chunks"], progress["elapsed_seconds"]
        values = np.lib.format.open_memmap(partial if partial.exists() else final, mode="r+")
        expected_start = 0
        for chunk in chunks:
            if chunk["start"] != expected_start or not expected_start < chunk["stop"] <= done:
                raise ValueError("Malformed extraction progress chunks")
            actual = hashlib.sha256(np.ascontiguousarray(values[chunk["start"]:chunk["stop"]]).tobytes()).hexdigest()
            if actual != chunk["sha256"]:
                raise ValueError("Partially extracted feature bytes changed")
            expected_start = chunk["stop"]
        if expected_start != done or not 0 <= done <= len(rows):
            raise ValueError("Malformed extraction cursor")
    else:
        if partial.exists() or final.exists():
            raise FileExistsError("Feature array exists without a recovery cursor")
        values = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(rows), 3, BRANCH_WIDTH))
        base.atomic_json(progress_path, {"fingerprint": fingerprint, "completed_rows": 0,
                                        "chunks": [], "elapsed_seconds": 0.0})
    if values.shape != (len(rows), 3, BRANCH_WIDTH) or values.dtype != np.float32:
        raise ValueError("Feature cache shape/dtype differs")
    model = model.to(args.device)
    started = time.monotonic()
    for start in range(done, len(rows), args.batch_size):
        stop = min(len(rows), start + args.batch_size)
        signal = normalized_batch(pool, rows[start:stop], mean, std, args.device)
        batch = extract_branches(model, signal).cpu().numpy()
        values[start:stop] = batch
        values.flush()
        chunks.append({"start": start, "stop": stop, "sha256": hashlib.sha256(batch.tobytes()).hexdigest()})
        base.atomic_json(progress_path, {"fingerprint": fingerprint, "completed_rows": stop, "chunks": chunks,
                                        "elapsed_seconds": previous + time.monotonic() - started})
        if stop % (args.batch_size * 10) == 0 or stop == len(rows):
            print(json.dumps({"stage": "mismatch_extract", "records": stop, "total": len(rows)}), flush=True)
    del values
    if partial.exists():
        os.replace(partial, final)
    base.atomic_json(directory / "metadata.json", {"fingerprint": fingerprint,
                     "shape": [len(rows), 3, BRANCH_WIDTH], "dtype": "float32", "branch_order": list(ARMS),
                     "elapsed_seconds": previous + time.monotonic() - started,
                     "sha256": {name: base.digest_file(directory / name) for name in ("features.npy", "rows.csv")}})
    progress_path.unlink(missing_ok=True)


def load_features(args, identity, rows):
    directory = args.output_dir / "features"
    # CPU probes consume an existing GPU/CPU extraction without changing its provenance.
    raw_metadata = json.loads((directory / "metadata.json").read_text())
    fingerprint = raw_metadata["fingerprint"]
    if fingerprint.get("inputs") != identity:
        raise ValueError("Frozen features were extracted from different inputs/code")
    metadata = feature_completion(directory, fingerprint)
    saved_rows, fields = read_rows(directory / "rows.csv")
    if saved_rows != rows or fields != list(FEATURE_FIELDS):
        raise ValueError("Frozen feature order/header differs from supervised cohort")
    features = np.load(directory / "features.npy", mmap_mode="r")
    if features.shape != (len(rows), 3, BRANCH_WIDTH) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError("Invalid completed feature array")
    return features, metadata


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def linear_logits(features, saved):
    return ((features - saved["mean"]) / saved["scale"]) @ saved["coefficient"].reshape(-1) + float(saved["intercept"].reshape(-1)[0])


def fit_probe(features, feature_rows, rows, directory, fingerprint, resume=False, c_values=C_VALUES):
    """Candidate boundaries are resumable; scaler and classifier see train rows only."""
    if not c_values:
        raise ValueError("At least one regularization value is required")
    directory.mkdir(parents=True, exist_ok=True)
    train_indices = supervised_indices(feature_rows, rows["labeled_train"], "train")
    dev_indices = supervised_indices(feature_rows, rows["development"], "validation")
    train_y = np.asarray([int(row["target"]) for row in rows["labeled_train"]])
    dev_y = np.asarray([int(row["target"]) for row in rows["development"]])
    train_raw = np.asarray(features[train_indices], dtype=np.float64)
    scaler = StandardScaler().fit(train_raw)
    train_x = scaler.transform(train_raw)
    choices, best_auc, best_path = [], -1.0, None
    for index, c in enumerate(c_values):
        candidate_path, marker_path = directory / f"candidate_{index}.npz", directory / f"candidate_{index}.json"
        if marker_path.exists():
            if not resume:
                raise FileExistsError("Prior classifier candidates exist; use --resume")
            choice = json.loads(marker_path.read_text())
            if (choice["fingerprint"] != fingerprint or choice["C"] != c
                    or choice["sha256"] != base.digest_file(candidate_path)):
                raise ValueError("Classifier candidate identity/checksum mismatch")
            saved = np.load(candidate_path)
            if not np.array_equal(saved["mean"], scaler.mean_) or not np.array_equal(saved["scale"], scaler.scale_):
                raise ValueError("Candidate standardization differs from training-only statistics")
            auc = float(roc_auc_score(dev_y, linear_logits(features[dev_indices], saved)))
            if auc != choice["development_auroc"]:
                raise ValueError("Candidate development score changed")
        else:
            started = time.monotonic()
            estimator = LogisticRegression(C=c, max_iter=3000, solver="lbfgs", random_state=42)
            with warnings.catch_warnings(record=True) as messages:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(train_x, train_y)
            parameters = {"mean": scaler.mean_, "scale": scaler.scale_, "coefficient": estimator.coef_, "intercept": estimator.intercept_}
            auc = float(roc_auc_score(dev_y, linear_logits(features[dev_indices], parameters)))
            atomic_npz(candidate_path, mean=scaler.mean_, scale=scaler.scale_, coefficient=estimator.coef_, intercept=estimator.intercept_)
            choice = {"fingerprint": fingerprint, "C": c, "development_auroc": auc,
                      "iterations": estimator.n_iter_.tolist(), "seconds": time.monotonic() - started,
                      "convergence_warnings": [str(message.message) for message in messages],
                      "sha256": base.digest_file(candidate_path)}
            base.atomic_json(marker_path, choice)
            print(json.dumps({"stage": "mismatch_probe", "directory": str(directory), "C": c,
                              "development_auroc": auc}), flush=True)
        choices.append({key: value for key, value in choice.items() if key not in ("fingerprint", "sha256")})
        if auc > best_auc:
            best_auc, best_path, best_c = auc, candidate_path, c
    with np.load(best_path) as selected:
        arrays = {key: selected[key].copy() for key in selected.files}
    atomic_npz(directory / "linear_model.npz", **arrays)
    selection = {"C": best_c, "best_development_auroc": best_auc, "candidates": choices,
                 "tie_rule": "First C in ascending grid wins ties", "selection_split": "development only"}
    base.atomic_json(directory / "selection.json", selection)
    return arrays, selection


def train_probe(args, branches, feature_rows, rows, feature_metadata, identity, arm, budget):
    directory = args.output_dir / f"{arm}_{budget}_seed42"
    fingerprint = {"inputs": identity, "features_sha256": feature_metadata["sha256"],
                   "arm": arm, "budget": budget, "seed": 42, "C_values": list(C_VALUES),
                   "solver": "lbfgs", "max_iter": 3000, "scaler": "StandardScaler fit on labeled training rows only",
                   "bootstrap": args.bootstrap, "sklearn_version": sklearn.__version__, "threads": args.threads}
    completion = directory / "complete.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["fingerprint"] != fingerprint or saved["sha256"] != {name: base.digest_file(directory / name) for name in ARTIFACTS}:
            raise ValueError("Completed classifier artifacts or configuration differ")
        return
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        raise FileExistsError(f"Incomplete classifier exists; use --resume: {directory}")
    if (directory / "config.json").exists() and json.loads((directory / "config.json").read_text())["fingerprint"] != fingerprint:
        raise ValueError("Partial classifier configuration differs")
    directory.mkdir(parents=True, exist_ok=True)
    features = features_for_arm(branches, arm)
    config = {"fingerprint": fingerprint, "arm": arm, "feature_dimension": features.shape[1],
              "records": {name: len(rows[name]) for name in ("labeled_train", "development", "calibration", "test")},
              "encoder_frozen": True, "normalization": "Frozen Experiment 004 training-pool mean/std",
              "retained_target_positions": "7..78 in each 79-token half", "prediction_query_positions": "3..74",
              "task": "PTB-XL diagnostic abnormality proxy; exploratory previously inspected test cohort"}
    base.atomic_json(directory / "config.json", config)
    started = time.monotonic()
    fitted, selection = fit_probe(features, feature_rows, rows, directory, fingerprint, resume=args.resume)
    calibration_indices = supervised_indices(feature_rows, rows["calibration"], "validation")
    test_indices = supervised_indices(feature_rows, rows["test"], "test")
    base.evaluate_predictions(f"cpc_mismatch_{arm}_{budget}", linear_logits(features[calibration_indices], fitted),
                               linear_logits(features[test_indices], fitted), rows["calibration"], rows["test"],
                               directory, 42, args.bootstrap)
    config.update({"C": selection["C"], "best_development_auroc": selection["best_development_auroc"],
                   "seconds_this_invocation": time.monotonic() - started})
    base.atomic_json(directory / "config.json", config)
    base.atomic_json(completion, {"fingerprint": fingerprint, "sha256": {name: base.digest_file(directory / name) for name in ARTIFACTS}})


def report(args):
    comparisons = {}
    lines = ["# Experiment 009: frozen CPC prediction mismatch", "",
             "All arms reuse the same completed 20-epoch local CPC encoder, frozen input normalization, "
             "and temporal positions. No encoder updates or additional SSL training were performed.", "",
             "The primary comparison is residual minus ordinary local features: both classifiers receive 1024 dimensions. "
             "The 512-dimensional context-only reference tests whether either extra branch adds useful information.", ""]
    for budget in BUDGETS:
        paths = {arm: args.output_dir / f"{arm}_{budget}_seed42" for arm in ARMS}
        if not all((path / "complete.json").is_file() for path in paths.values()):
            raise RuntimeError("All six frozen probes must complete before the study report")
        lines += [f"## {budget}", "", "| Arm | Dimensions | AUROC | AP | Sensitivity | Specificity | Brier |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for arm, path in paths.items():
            metrics = json.loads((path / "metrics.json").read_text())["test"]
            config = json.loads((path / "config.json").read_text())
            lines.append(f"| {arm} | {config['feature_dimension']} | {metrics['auroc']:.4f} | {metrics['average_precision']:.4f} | "
                         f"{metrics['sensitivity']:.4f} | {metrics['specificity']:.4f} | {metrics['brier']:.4f} |")
        lines += ["", "| Paired comparison | AUROC difference | Patient bootstrap 95% interval |", "| --- | ---: | --- |"]
        for left, right in (("residual", "ordinary"), ("ordinary", "context"), ("residual", "context")):
            comparison = base.paired_comparison(paths[left], paths[right], args.bootstrap)
            comparisons[f"{budget}_{left}_minus_{right}"] = comparison
            auc = comparison["auroc"]
            lines.append(f"| {left} minus {right} | {auc['difference']:+.4f} | [{auc['ci95'][0]:+.4f}, {auc['ci95'][1]:+.4f}] |")
        lines.append("")
    lines += ["Regularization is selected on development patients. Platt calibration and the >=95% sensitivity threshold "
              "use separate calibration patients. Each threshold is frozen before test evaluation.", "",
              "The residual subtracts normalized predicted tokens from normalized observed tokens; its norm is retained. "
              "A large mismatch can reflect noise or artifacts, so this feature is not a clinical abnormality score. "
              "One seed and a previously used test cohort support exploratory conclusions only. Paired intervals describe "
              "patient sampling uncertainty, not retraining variation.", ""]
    base.atomic_json(args.output_dir / "paired_comparisons.json", comparisons)
    (args.output_dir / "report.md").write_text("\n".join(lines))
    base.atomic_json(args.output_dir / "complete.json", {
        "sha256": {name: base.digest_file(args.output_dir / name) for name in ("report.md", "paired_comparisons.json")},
        "probe_completions": {f"{arm}_{budget}": base.digest_file(args.output_dir / f"{arm}_{budget}_seed42/complete.json")
                              for arm in ARMS for budget in BUDGETS}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "extract", "train", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument("--manifest-root", type=Path, default=base.DEFAULT_MANIFEST)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads, args.bootstrap) < 1:
        parser.error("Batch size, threads and bootstrap must be positive")
    if args.stage in ("profile", "extract", "all") and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA extraction requested but unavailable")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runner.lock").open("a+") as lock, threadpool_limits(limits=args.threads):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool, model, mean, std, manifests, rows, identity = prepare_inputs(args)
        base.atomic_json(args.output_dir / "preflight.json", {"identity": identity, "feature_records": len(rows),
                         "budgets": {name: len(value["labeled_train"]) for name, value in manifests.items()},
                         "encoder_frozen": not any(p.requires_grad for p in model.parameters())})
        print(json.dumps({"stage": "mismatch_check", "records": len(rows), "arms": list(ARMS)}), flush=True)
        if args.stage == "check":
            return
        if args.stage in ("profile", "extract", "all"):
            with gpu_lock(args.device):
                profile_path = args.output_dir / "profile.json"
                if (args.stage == "profile" or not profile_path.is_file()
                        or json.loads(profile_path.read_text()).get("fingerprint") != extraction_fingerprint(args, identity)):
                    profile(args, pool, model, mean, std, rows, identity)
                if args.stage in ("extract", "all"):
                    extract(args, pool, model, mean, std, rows, identity)
            model.to("cpu")
            if args.device == "cuda":
                torch.cuda.empty_cache()
        if args.stage in ("train", "all"):
            branches, metadata = load_features(args, identity, rows)
            for budget in BUDGETS:
                for arm in ARMS:
                    train_probe(args, branches, rows, manifests[budget], metadata, identity, arm, budget)
            report(args)


if __name__ == "__main__":
    main()
