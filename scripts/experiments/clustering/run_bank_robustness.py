"""Execute the prospective cached NTT support-bank robustness experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.clustering.bank_inputs import cached_inputs, committed_sources, replay_predecessor
from ecg_experiment.clustering.bank_robustness import (
    bank_affinities,
    bank_fit,
    bank_prototypes,
    distribution,
    primary_decision,
    sensitivity,
    support_draws,
)
from ecg_experiment.clustering.native_cache import runtime_fingerprint
from ecg_experiment.clustering02 import permitted_metadata
from ecg_experiment.eda.ptbxl import diagnostic_classes, superclass_flags
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored
from ecg_experiment.same_wave_ntt04 import FINDINGS

SOURCES = (
    "ecg_experiment/clustering/bank_inputs.py",
    "ecg_experiment/clustering/bank_robustness.py",
    "scripts/experiments/clustering/run_bank_robustness.py",
    "tests/clustering/test_bank_robustness.py",
    "docs/research/clustering/cluster04-bank-robustness/protocol.md",
)


def classifier_sensitivity(data: dict, prototypes: np.ndarray) -> tuple[dict, dict]:
    """Evaluate fixed previously explored target vectors under every sampled bank."""
    targets = data["selection"]["targets"]
    vectors = np.array([data["support_features"][str(row["ecg_id"])]["native_qrs"] for row in targets])
    truth = np.array([FINDINGS.index(row["finding"]) for row in targets])
    if len(targets) != 60 or not np.array_equal(np.bincount(truth, minlength=3), [20, 20, 20]):
        raise ValueError("Frozen classification target denominator changed")
    scores = np.clip(np.einsum("nd,bfd->bnf", vectors, prototypes), -1, 1)
    predicted = np.argmax(scores, axis=2)
    correct = predicted == truth
    balanced = np.stack([correct[:, truth == k].mean(axis=1) for k in range(3)], axis=1).mean(axis=1)
    physical = np.mean([row["correct"]["raw_full_context"] for row in data["ntt_predictions"]])
    if abs(physical - 167 / 180) > 1e-12:
        raise ValueError("Physical predecessor baseline changed")
    return {
        "targets": 60,
        "targets_per_finding": 20,
        "balanced_accuracy": distribution(balanced),
        "bank_accuracy_values": balanced.tolist(),
        "per_finding_recall": {
            finding: distribution(correct[:, truth == k].mean(axis=1)) for k, finding in enumerate(FINDINGS)
        },
        "patient_fraction_correct": distribution(correct.mean(axis=0)),
        "patients_correct_in_every_bank": int(np.all(correct, axis=0).sum()),
        "patients_correct_in_no_bank": int(np.all(~correct, axis=0).sum()),
        "physical_original_three_bank_mean_balanced_accuracy": float(physical),
        "physical_bank_uncertainty_not_resampled": True,
        "new_independent_accuracy_evidence": False,
    }, {
        "classification_predictions": predicted,
        "classification_correct": correct,
        "classification_truth": truth,
    }


def annotation_sensitivity(data: dict, assignments: dict) -> dict:
    """Describe permitted existing source annotations after freezing the geometric decision."""
    metadata = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
    classes = diagnostic_classes(pd.read_csv(metadata.with_name("scp_statements.csv"), index_col=0))
    selected = permitted_metadata(metadata, data["ids"].ecg_id, True)
    flags = superclass_flags(selected, classes).drop(columns="abnormal").iloc[data["anchors"]]
    combinations = flags.astype(int).astype(str).agg("".join, axis=1).to_numpy()
    return {
        view: distribution(
            np.array([adjusted_mutual_info_score(combinations, row[data["anchors"]]) for row in labels])
        )
        for view, labels in assignments.items()
    }


def run(output: Path) -> dict:
    """Run all declared banks and ensembles with immutable admission and bounded CPU work."""
    started = time.perf_counter()
    deadline = started + 1200
    data = cached_inputs()
    sources = committed_sources(SOURCES, data["cluster"]["source_sha256"])
    reproduction = replay_predecessor(data, deadline)
    draws = support_draws(data["selection"])
    manifest = output / "bank_manifest_local.json"
    write_json_atomic(
        manifest,
        {
            "seed": 20261020,
            "banks": draws,
            "ensemble_blocks": [list(range(i, i + 5)) for i in range(0, 50, 5)],
        },
    )
    manifest_sha = sha256_file(manifest)
    prototypes = np.stack([bank_prototypes(draw, data["support_features"]) for draw in draws])
    assignments, checks = {"regional": [], "record": []}, {"regional": [], "record": []}
    regional_views = []
    configs = {
        "regional": data["cluster"]["selected"]["ntt_regions"],
        "record": data["cluster"]["selected"]["ntt_affinity"],
    }
    for i, bank in enumerate(prototypes):
        record, regional = bank_affinities(data["cache"], bank)
        regional_views.append(regional)
        for view, values in (("regional", regional), ("record", record)):
            state, labels = bank_fit(values, data["fitting"], data["anchors"], configs[view], deadline)
            checks[view].append(state)
            assignments[view].append(labels)
        write_json_atomic(output / "progress.json", {"completed_banks": i + 1, "checks": checks})
        print(
            f"Bank {i + 1}/50: regional admissible={checks['regional'][-1]['admissible']}, "
            f"record admissible={checks['record'][-1]['admissible']}",
            flush=True,
        )
    arrays = {view: np.stack(labels) for view, labels in assignments.items()}
    summaries, private = {}, {}
    for view in ("regional", "record"):
        config = configs[view]
        reference = data["assignments"][config["key"] + "|all"]
        summaries[view], details = sensitivity(
            arrays[view], checks[view], reference, data["fitting"], data["anchors"], config["clusters"]
        )
        private.update({view + "_" + key: value for key, value in details.items()})
    ensemble_checks, ensemble_labels = [], []
    for i in range(0, 50, 5):
        values = np.mean(regional_views[i : i + 5], axis=0)
        state, labels = bank_fit(values, data["fitting"], data["anchors"], configs["regional"], deadline)
        ensemble_checks.append(state)
        ensemble_labels.append(labels)
    reference = data["assignments"][configs["regional"]["key"] + "|all"]
    summaries["five_bank_ensembles"], details = sensitivity(
        np.stack(ensemble_labels), ensemble_checks, reference, data["fitting"], data["anchors"], 2
    )
    private.update({"ensemble_" + key: value for key, value in details.items()})
    decision = primary_decision(summaries["regional"])
    geometric_path = output / "frozen_geometric_decision.json"
    write_json_atomic(
        geometric_path,
        {
            "summaries": summaries,
            "decision": decision,
            "checks": checks,
            "ensemble_checks": ensemble_checks,
            "bank_manifest_sha256": manifest_sha,
        },
    )
    frozen_sha = sha256_file(geometric_path)
    classification, target_arrays = classifier_sensitivity(data, prototypes)
    annotation = annotation_sensitivity(data, arrays)
    if sha256_file(geometric_path) != frozen_sha or sha256_file(manifest) != manifest_sha:
        raise ValueError("Frozen decision or support manifest changed")
    if time.perf_counter() >= deadline:
        raise TimeoutError("Prospective bank study budget exceeded")
    np.savez_compressed(
        output / "assignments_local.npz",
        **arrays,
        ensembles=np.stack(ensemble_labels),
        **private,
        **target_arrays,
    )
    data["ids"].to_csv(output / "identities_local.csv", index=False)
    result = {
        "study": "CLUSTER-04",
        "status": "completed",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": sources,
        "input_sha256": data["inputs"],
        "runtime": runtime_fingerprint(),
        "counts": {
            "patients": 1993,
            "fitting": 1582,
            "validation_anchors": 411,
            "banks": 50,
            "single_bank_fits": 100,
            "ensemble_fits": 10,
            "new_encoder_inference": 0,
            "waveform_reads": 0,
        },
        "predecessor_reproduction": reproduction,
        "summaries": summaries,
        "decision": decision,
        "checks": checks,
        "ensemble_checks": ensemble_checks,
        "classification": classification,
        "annotation_ami": annotation,
        "geometric_decision_frozen_before_annotation_parse": True,
        "frozen_decision_sha256": frozen_sha,
        "bank_manifest_sha256": manifest_sha,
        "seconds": time.perf_counter() - started,
        "artifact_sha256": {
            to_stored(path): sha256_file(path)
            for path in output.iterdir()
            if path.name not in ("result.json", "status.json", "failure.json")
        },
    }
    write_json_atomic(output / "result.json", result)
    return result


def main() -> None:
    """Preserve a single bounded execution and its failure receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "status.json", {"status": "running"})
    try:
        with threadpool_limits(limits=2):
            result = run(output)
        write_json_atomic(
            output / "status.json",
            {"status": "completed", "result_sha256": sha256_file(output / "result.json")},
        )
        print(json.dumps({"decision": result["decision"], "seconds": result["seconds"]}), flush=True)
    except Exception as error:
        write_json_atomic(
            output / "failure.json", {"error_type": type(error).__name__, "message": str(error)}
        )
        write_json_atomic(output / "status.json", {"status": "failed"})
        raise


if __name__ == "__main__":
    main()
