"""Run the frozen native encoder and NTT reference-affinity clustering comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.clustering.native_cache_v2 import SOURCES as CACHE_SOURCES
from ecg_experiment.clustering.native_cache_v2 import frozen_sources as cache_sources
from ecg_experiment.clustering.native_cache_v2 import runtime_fingerprint, verified_cohort
from ecg_experiment.clustering.reference_geometry import (
    COVARIANCES,
    SEED,
    VIEWS,
    affinity_views,
    candidate,
    check_budget,
    density_views,
    primary_decision,
    representation_cache,
    select,
    single_reference_fit,
    support_prototypes,
)
from ecg_experiment.clustering01 import (
    associations,
    correct_enrichments,
    enrichments,
    partition,
    partition_summary,
    representation,
)
from ecg_experiment.clustering02 import anchor_patient, permitted_metadata
from ecg_experiment.eda.ptbxl import diagnostic_classes, superclass_flags
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.localization.jepa_native import CHECKPOINT_SHA256, EMBEDDING_SPACE
from ecg_experiment.localization.native_study import (
    native_predictions,
    native_summary,
    predecessor_inputs,
)
from ecg_experiment.paths import to_stored

NTT06_SHA = "3deb6cc481ebd085ff30225e7954ad14d965880542484e162d43b2b5b9aea1b3"
SOURCES = tuple(
    dict.fromkeys(
        (
            *CACHE_SOURCES,
            "ecg_experiment/clustering/reference_geometry.py",
            "scripts/experiments/clustering/run_reference_geometry.py",
            "tests/clustering/test_reference_geometry.py",
            "ecg_experiment/clustering01.py",
            "ecg_experiment/clustering02.py",
            "ecg_experiment/eda/ptbxl.py",
            "ecg_experiment/intervals.py",
        )
    )
)


def source_binding() -> dict:
    """Require committed byte identity for all cache and clustering dependencies."""
    hashes = cache_sources()
    for name in SOURCES:
        data = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Uncommitted source: {name}")
        hashes[name] = digest
    return hashes


def native_integrity() -> tuple[dict, dict, dict, dict]:
    """Rebuild all frozen native predecessor methods before consuming the broad-cohort cache."""
    base = ROOT / "outputs/ntt06_native_encoder"
    if sha256_file(base / "result.json") != NTT06_SHA:
        raise ValueError("Frozen native predecessor changed")
    result = json.loads((base / "result.json").read_text())
    inputs = {to_stored(base / "result.json"): NTT06_SHA}
    for name in ("selection_local.json", "feature_vectors_local.json", "predictions_local.json"):
        path = base / name
        digest = sha256_file(path)
        if digest != result["artifact_sha256"][to_stored(path)]:
            raise ValueError(f"Native predecessor artifact differs: {name}")
        inputs[to_stored(path)] = digest
    selected, _, previous, _, extra = predecessor_inputs()
    inputs.update(extra)
    features = json.loads((base / "feature_vectors_local.json").read_text())
    converted = {int(k): {m: np.asarray(v) for m, v in row.items()} for k, row in features.items()}
    replayed, _ = native_predictions(converted, selected, previous, result["embedding_space"])
    saved = json.loads((base / "predictions_local.json").read_text())
    maximum = 0.0
    if len(replayed) != 180 or len(saved) != 180:
        raise ValueError("Native predecessor denominator differs")
    for actual, expected in zip(replayed, saved, strict=True):
        if actual["chosen"] != expected["chosen"] or actual["patient_id"] != expected["patient_id"]:
            raise ValueError("Native predecessor choice or identity differs")
        for m in actual["scores"]:
            maximum = max(
                maximum, float(np.max(np.abs(np.array(actual["scores"][m]) - expected["scores"][m])))
            )
    if maximum > 1e-12 or native_summary(replayed) != result["summary"]:
        raise ValueError("Native predecessor score or bootstrap does not reproduce")
    return (
        selected,
        features,
        {
            "native_scores_choices_and_primary_interval_exact": True,
            "predictions": 180,
            "maximum_score_error": maximum,
        },
        inputs,
    )


def physical_integrity() -> dict:
    """Reproduce the original physical partitions on their unchanged 1,999-patient cohort."""
    base = ROOT / "outputs/cluster01_physical_morphology"
    old = json.loads((base / "result.json").read_text())
    identities = pd.read_csv(base / "analyzed_training_patients.csv")
    features = pd.read_csv(base / "morphology_features.csv")
    measures = (
        pd.read_csv(base / "physical_measurements.csv")
        .set_index("ecg_id")
        .loc[identities.ecg_id]
        .reset_index(drop=True)
    )
    meta = permitted_metadata(ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv", identities.ecg_id)
    nuisance = pd.concat([measures, meta[["device", "site", "quality"]]], axis=1)
    original = pd.read_csv(base / "assignments.csv")
    checks = {}
    for view in ("raw", "residual"):
        key = f"{view}:kmeans"
        values, _ = representation(features, nuisance, view == "residual")
        labels = partition(values, "kmeans", old["results"][key]["parameter"])
        ari = float(adjusted_rand_score(original[key], labels))
        error = abs(partition_summary(values, labels)["silhouette"] - old["results"][key]["silhouette"])
        if ari != 1 or error > 1e-10:
            raise ValueError("Physical predecessor partition failed exact replay")
        checks[key] = {"assignment_ari": ari, "silhouette_absolute_error": error}
    return checks


def verified_cache(path: Path, expected_runtime: dict, inputs: dict) -> tuple[dict, dict]:
    """Admit only a completed, source-bound native cache with adequate cohort retention."""
    result = json.loads(path.read_text())
    status_path = path.with_name("status.json")
    status = json.loads(status_path.read_text())
    if status.get("status") != "completed" or status.get("result_sha256") != sha256_file(path):
        raise ValueError("Native cache lacks a completed result binding")
    if result["source_sha256"] != cache_sources() or result["runtime"] != expected_runtime:
        raise ValueError("Native cache source/runtime differs")
    if (
        result.get("stage") != "native_cache_extraction_v2"
        or result.get("status") != "completed"
        or result.get("retention_gate_passed") is not True
        or result.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or result.get("embedding_space") != EMBEDDING_SPACE
        or result.get("candidate_patients") != 1993
    ):
        raise ValueError("Native cache stage, retention or embedding identity differs")
    for name, digest in result["input_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Native cache input changed: {name}")
        inputs[name] = digest
    for name, digest in result["artifact_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Native cache artifact differs: {name}")
        inputs[name] = digest
    inputs[to_stored(path)] = sha256_file(path)
    inputs[to_stored(status_path)] = sha256_file(status_path)
    with np.load(path.with_name("native_features_local.npz"), allow_pickle=False) as archive:
        cache = {k: archive[k] for k in archive.files}
    if (
        len(cache["ecg_ids"]) != result["accepted_patients"]
        or len(cache["ecg_ids"]) < 0.95 * 1993
        or not np.isfinite(cache["region_vectors"]).all()
    ):
        raise ValueError("Insufficient native cache retention or nonfinite values")
    return cache, result


def interpret(
    selected: dict,
    assignments: dict,
    ids: pd.DataFrame,
    nuisance: pd.DataFrame,
    anchors: np.ndarray,
    permutation_labels: list[np.ndarray],
    output: Path,
    all_candidates: pd.DataFrame,
) -> dict:
    """Join existing target annotations only after the unsupervised decision is saved."""
    meta_path = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
    classes = diagnostic_classes(pd.read_csv(meta_path.with_name("scp_statements.csv"), index_col=0))
    meta = permitted_metadata(meta_path, ids.ecg_id, True)
    flags = superclass_flags(meta, classes).drop(columns="abnormal")
    anchor_flags = flags.iloc[anchors].reset_index(drop=True)
    anchor_nuisance = nuisance.iloc[anchors].reset_index(drop=True)
    summaries, rows = {}, []
    for view, chosen in selected.items():
        labels = assignments[chosen["key"]]
        summaries[view] = {
            "eligible": chosen["eligible"],
            "anchor_associations": associations(labels["anchor"], anchor_flags, anchor_nuisance),
            "full_cohort_associations_descriptive": associations(labels["all"], flags, nuisance),
        }
        rows.extend(enrichments(labels["anchor"], anchor_flags, view))
    correct_enrichments(rows)
    pd.DataFrame(rows).to_csv(output / "anchor_enrichment_local.csv", index=False)
    combinations = anchor_flags.astype(int).astype(str).agg("".join, axis=1).to_numpy()
    true_ami = float(
        adjusted_mutual_info_score(combinations, assignments[selected["ntt_regions"]["key"]]["anchor"])
    )
    permutation_ami = [
        float(adjusted_mutual_info_score(combinations, labels)) for labels in permutation_labels
    ]
    all_meta = permitted_metadata(meta_path, all_candidates.ecg_id, True)
    all_flags = superclass_flags(all_meta, classes).drop(columns="abnormal")
    retained = all_candidates.ecg_id.isin(ids.ecg_id).to_numpy()
    retention = {
        label: {
            "candidates": int(all_flags[label].sum()),
            "retained": int((all_flags[label].to_numpy(bool) & retained).sum()),
            "excluded": int((all_flags[label].to_numpy(bool) & ~retained).sum()),
        }
        for label in all_flags
    }
    quality = all_meta.quality.to_numpy()
    return {
        "selected": summaries,
        "anchor_annotation_counts": anchor_flags.sum().astype(int).to_dict(),
        "anchor_enrichments": rows,
        "bh_family_tests": len(rows),
        "ntt_region_annotation_ami": true_ami,
        "random_support_annotation_ami": permutation_ami,
        "random_support_sensitivity_not_confirmatory_p_value": True,
        "retention_by_annotation": retention,
        "retention_by_quality": {
            str(int(q)): {
                "candidates": int((quality == q).sum()),
                "retained": int(((quality == q) & retained).sum()),
            }
            for q in np.unique(quality)
        },
    }


def run_grid(
    transforms: dict, samples: list[np.ndarray], anchors: np.ndarray, output: Path, deadline: float
) -> tuple[list, dict]:
    """Execute every declared configuration and preserve incremental label-free progress."""
    candidates, assignments = [], {}
    for view in VIEWS:
        for method, covariances in (("kmeans", ("none",)), ("gmm", COVARIANCES)):
            for covariance in covariances:
                for clusters in (2, 3, 4, 8):
                    row, labels = candidate(
                        view, method, covariance, clusters, transforms[view], samples, anchors, deadline
                    )
                    candidates.append(row)
                    assignments[row["key"]] = labels
                    write_json_atomic(output / "candidate_progress.json", candidates)
                    print(
                        f"{row['key']}: eligible={row['eligible']} "
                        f"medianARI={row['stability']['median']:.3f}",
                        flush=True,
                    )
    return candidates, assignments


def run(output: Path, cache_path: Path) -> dict:
    """Execute label-blind candidate selection, then the fixed interpretation stage."""
    started = time.perf_counter()
    deadline = started + 900
    sources = source_binding()
    all_ids, physical, nuisance, inputs = verified_cohort()
    selection, support_features, native_check, native_inputs = native_integrity()
    inputs.update(native_inputs)
    physical_check = physical_integrity()
    cache, cache_result = verified_cache(cache_path, runtime_fingerprint(), inputs)
    positions = all_ids.set_index("ecg_id").index.get_indexer(cache["ecg_ids"])
    if (positions < 0).any() or len(set(cache["patient_ids"])) != len(positions):
        raise ValueError("Native cache identities are outside curated candidates")
    ids = all_ids.iloc[positions].reset_index(drop=True)
    if not np.array_equal(ids.patient_id.to_numpy(), cache["patient_ids"]):
        raise ValueError("Native cache patient/ECG row alignment differs")
    selected_physical = physical.iloc[positions].reset_index(drop=True)
    selected_nuisance = nuisance.iloc[positions].reset_index(drop=True)
    prototype_banks = support_prototypes(selection, support_features)
    affinity, regions, bank_regions = affinity_views(cache, prototype_banks)
    views = {
        "native_qrs": cache["qrs"],
        "native_global": cache["global_pair"],
        "ntt_affinity": affinity,
        "ntt_regions": regions,
        "physical": selected_physical.to_numpy(),
    }
    anchors = np.flatnonzero(ids.patient_id.map(anchor_patient).to_numpy())
    fitting = np.flatnonzero(~ids.patient_id.map(anchor_patient).to_numpy())
    if len(anchors) < 100 or len(fitting) < 500:
        raise ValueError("Too few fitting or validation-anchor patients")
    samples = [fitting] + [
        np.sort(np.random.default_rng(SEED + 1 + i).choice(fitting, int(0.8 * len(fitting)), replace=False))
        for i in range(8)
    ]
    transforms = representation_cache(views, samples, deadline)
    candidates, assignments = run_grid(transforms, samples, anchors, output, deadline)
    selected = select(candidates)
    checks, bank_labels = [], []
    for values in bank_regions:
        status, labels = single_reference_fit(values, fitting, anchors, selected["ntt_regions"], deadline)
        checks.append(status)
        bank_labels.append(labels)
    decision = primary_decision(selected, assignments, bank_labels, checks)
    permutation_checks, permutation_labels = [], []
    reference = assignments[selected["ntt_regions"]["key"]]["anchor"]
    for i in range(20):
        check_budget(deadline)
        _, values, _ = affinity_views(cache, support_prototypes(selection, support_features, SEED + 1 + i))
        status, labels = single_reference_fit(values, fitting, anchors, selected["ntt_regions"], deadline)
        status.update(seed=SEED + 1 + i, anchor_ari_with_true=float(adjusted_rand_score(reference, labels)))
        permutation_checks.append(status)
        permutation_labels.append(labels)
    density, density_labels = density_views(transforms, deadline)
    unsupervised = {
        "selected": selected,
        "decision": decision,
        "individual_bank_checks": checks,
        "support_label_permutations": permutation_checks,
        "density_descriptive": density,
    }
    unsupervised_path = output / "unsupervised_selection.json"
    write_json_atomic(unsupervised_path, unsupervised)
    selection_hash = sha256_file(unsupervised_path)
    np.savez_compressed(
        output / "assignments_local.npz",
        **{f"{key}|{kind}": v for key, entry in assignments.items() for kind, v in entry.items()},
        individual_banks=np.stack(bank_labels),
        permuted_banks=np.stack(permutation_labels),
        **{f"density|{key}": v for key, v in density_labels.items()},
    )
    ids.to_csv(output / "retained_identities_local.csv", index=False)
    np.savez_compressed(output / "representations_local.npz", **views)
    write_json_atomic(
        output / "patient_splits_local.json",
        {
            "ecg_ids": cache["ecg_ids"].tolist(),
            "anchors": anchors.tolist(),
            "samples": [s.tolist() for s in samples],
        },
    )
    interpretation = interpret(
        selected, assignments, ids, selected_nuisance, anchors, permutation_labels, output, all_ids
    )
    if sha256_file(unsupervised_path) != selection_hash:
        raise ValueError("Unsupervised decision changed during annotation interpretation")
    check_budget(deadline)
    result = {
        "study": "CLUSTER-03",
        "status": "completed",
        "source_sha256": sources,
        "input_sha256": inputs,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": runtime_fingerprint(),
        "counts": {
            "candidates": 1993,
            "retained": len(ids),
            "fitting": len(fitting),
            "validation_anchors": len(anchors),
            "excluded_support_overlaps": 6,
        },
        "predecessor_reproduction": {"ntt06": native_check, "cluster01": physical_check},
        "candidates": candidates,
        **unsupervised,
        "interpretation": interpretation,
        "pca": {
            view: [{k: v for k, v in state.items() if k != "coordinates"} for state in states]
            for view, states in transforms.items()
        },
        "patient_sample_hashes": [
            hashlib.sha256(ids.patient_id.to_numpy()[s].astype("<i8").tobytes()).hexdigest() for s in samples
        ],
        "selection_frozen_before_target_annotation_parse": True,
        "unsupervised_selection_sha256": selection_hash,
        "native_extraction_receipt_sha256": sha256_file(cache_path),
        "embedding_space": cache_result["embedding_space"],
        "seconds": time.perf_counter() - started,
        "waveform_reads": 0,
        "new_encoder_inference": False,
        "gpu_used": False,
        "artifact_sha256": {
            to_stored(p): sha256_file(p)
            for p in output.iterdir()
            if p.name not in ("status.json", "result.json", "failure.json")
        },
    }
    write_json_atomic(output / "result.json", result)
    return result


def main() -> None:
    """Preserve one bounded execution and typed failures without changing its scientific settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("outputs/cluster03_native_cache_v2/result.json"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    cache_path = args.cache if args.cache.is_absolute() else ROOT / args.cache
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "status.json", {"status": "running"})
    try:
        import torch

        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            result = run(output, cache_path)
        write_json_atomic(
            output / "status.json",
            {"status": "completed", "result_sha256": sha256_file(output / "result.json")},
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "primary_gate": result["decision"]["primary_persistence_gate_passed"],
                    "seconds": result["seconds"],
                }
            ),
            flush=True,
        )
    except Exception as error:
        write_json_atomic(output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        write_json_atomic(output / "status.json", {"status": "failed"})
        raise


if __name__ == "__main__":
    main()
