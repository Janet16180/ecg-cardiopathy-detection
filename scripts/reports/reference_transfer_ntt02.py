"""Run the frozen CPU-only NTT-02 reference-window selection feasibility study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import wfdb

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.reference_label_audit import audit_local_labels, audit_reasoning, identifier_table
from ecg_experiment.reference_transfer_ntt02 import (
    SEEDS,
    attach_features,
    class_groups,
    eligibility_table,
    inner_fold,
    read_questions,
    score_groups,
    summarize,
    support_patients,
)

matplotlib.use("Agg")

SOURCE_FILES = (
    "ecg_experiment/reference_transfer_ntt02.py",
    "scripts/reports/reference_transfer_ntt02.py",
    "tests/test_reference_transfer_ntt02.py",
    "docs/ntt-02-native-reference-transfer.md",
    "ecg_experiment/reference_label_audit.py",
    "ecg_experiment/eda/ptbxl.py",
    "ecg_experiment/eda/signals.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/files.py",
    "ecg_experiment/__init__.py",
    "pyproject.toml",
    "uv.lock",
    "reports/ntt01_reference_label_audit.json",
)


def write_json(path: Path, value: Any) -> None:
    """Write strict JSON without converting nonfinite numbers silently."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes() -> dict[str, str]:
    """Require the exact executed source bytes to be committed before scoring."""
    hashes = {}
    for name in SOURCE_FILES:
        committed = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, check=True, capture_output=True)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(committed.stdout).hexdigest() != digest:
            raise ValueError(f"Uncommitted source: {name}")
        hashes[name] = digest
    return hashes


def replay_predecessor(annotations: Path, metadata: Path) -> dict[str, Any]:
    """Exactly reproduce both NTT-01 aggregate blocks before any new scored analysis."""
    previous = json.loads((ROOT / "reports/ntt01_reference_label_audit.json").read_text())
    for name, digest in previous["input_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Predecessor input bytes changed")
    reasoning = audit_reasoning(annotations, identifier_table(metadata))
    labels = audit_local_labels(metadata, metadata.with_name("scp_statements.csv"))
    normalized = json.loads(json.dumps(reasoning))
    if normalized != previous["reasoning_annotations"] or labels != previous["local_labels"]:
        raise ValueError("Predecessor aggregate reproduction failed")
    return {
        "all_reasoning_aggregates_exact": True,
        "all_local_label_aggregates_exact": True,
        "safe_cases": reasoning["safe_cases"],
        "safe_ecgs": reasoning["safe_unique_ecgs"],
        "safe_patients": reasoning["safe_unique_patients"],
        "wave_answers": reasoning["safe_grounding"]["wave_grounding"]["questions"],
        "timestamp_answers": reasoning["safe_grounding"]["wave_grounding"]["selected_time_ranges"],
        "non_timestamp_answers": reasoning["safe_grounding"]["wave_grounding"][
            "invalid_selected_time_ranges"
        ],
    }


def illustrate(groups: dict, predictions: list[dict], base: Path, output: Path) -> dict:
    """Plot a prospectively fixed first example, including its exact coordinate mapping."""
    import matplotlib.pyplot as plt

    key = sorted(groups)[0]
    target = min(groups[key], key=lambda q: (q["patient_id"], q["qid"]))
    fold = inner_fold(target["patient_id"])
    patients = support_patients(groups[key], key, fold, SEEDS[0], 3)
    support = min((q for q in groups[key] if q["patient_id"] == patients[0]), key=lambda q: q["qid"])
    selected = next(
        row
        for row in predictions
        if row["qid"] == target["qid"] and row["budget"] == 3 and row["seed"] == SEEDS[0]
    )
    figure, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    for axis, question, title in zip(
        axes, (support, target), ("Reference patient", "Disjoint target patient"), strict=True
    ):
        signal, fields = wfdb.rdsamp(str(base / question["waveform"]))
        lead = fields["sig_name"].index("II")
        axis.plot(np.arange(len(signal)) / 100, signal[:, lead], color="black", linewidth=0.65)
        for index, (start, stop) in enumerate(question["intervals"]):
            color = "tab:green" if index == question["answer"] else "tab:gray"
            axis.axvspan(start, stop, color=color, alpha=0.22)
            axis.text(
                (start + stop) / 2,
                0.98,
                str(index),
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
            )
        if question is target:
            start, stop = question["intervals"][selected["chosen"]["morphology"]]
            axis.axvspan(start, stop, color="tab:blue", alpha=0.25, hatch="//")
        axis.set(title=title, xlabel="Time (s)", ylabel="Lead II (mV)", xlim=(0, 10))
    figure.suptitle(
        "Fixed first example: green = released silver answer; blue hatch = morphology choice\n"
        "Lead II shown for readability; matching uses all 12 leads. " + support["criterion"],
        fontsize=10,
    )
    figure.savefig(output / "example.svg")
    figure.savefig(output / "example.png", dpi=160)
    plt.close(figure)
    coordinates = {
        "target_qid": target["qid"],
        "support_qid": support["qid"],
        "class_key": key,
        "seed": SEEDS[0],
        "budget": 3,
        "target_intervals_seconds": target["intervals"],
        "target_answer": target["answer"],
        "morphology_choice": selected["chosen"]["morphology"],
        "support_selected_seconds": support["intervals"][support["answer"]],
        "display_lead": "II",
        "matching_leads": "all12",
        "sampling_hz": 100,
    }
    write_json(output / "example_coordinates.json", coordinates)
    return coordinates


def verify_partition(table: Any, metadata: Path, partition: Path, output: Path) -> dict:
    """Require exact agreement with PR67's shared patient roles and retain its input."""
    allocation = json.loads(partition.read_text())
    if allocation["metadata_sha256"] != sha256_file(metadata):
        raise ValueError("Shared partition metadata differs")
    train = {int(row["ecg_id"]) for row in allocation["records"] if row["role"] == "train"}
    independent = set(table.index[table.eligible])
    if train != independent:
        raise ValueError("Independent eligibility disagrees with PR67 allocation")
    retained = output / "partition_local.json"
    retained.write_bytes(partition.read_bytes())
    return {
        "path": to_stored(retained),
        "sha256": sha256_file(retained),
        "all_record_train_roles_exact": True,
        "train_records": len(train),
        "train_patients": int(table.loc[table.eligible, "patient_id"].nunique()),
        "partition_id": allocation["partition_id"],
        "policy": allocation["policy"],
    }


def candidate_descriptions(groups: dict) -> dict:
    """Describe option construction on exactly the selected scoring denominator."""
    rows = [q for questions in groups.values() for q in questions]
    selected_widths = [q["intervals"][q["answer"]][1] - q["intervals"][q["answer"]][0] for q in rows]
    other_widths = [
        stop - start for q in rows for i, (start, stop) in enumerate(q["intervals"]) if i != q["answer"]
    ]
    return {
        "ecgs": len({q["ecg_id"] for q in rows}),
        "questions_with_step_lead_answers": sum(bool(q["step_lead_answers_unlinked"]) for q in rows),
        "selected_width_seconds_percentiles": np.percentile(selected_widths, [0, 25, 50, 75, 100]).tolist(),
        "other_width_seconds_percentiles": np.percentile(other_widths, [0, 25, 50, 75, 100]).tolist(),
        "answer_positions": {
            str(i): sum(q["answer"] == i for q in rows) for i in sorted({q["answer"] for q in rows})
        },
    }


def run(output: Path, partition: Path) -> dict[str, Any]:
    """Bind committed sources, replay the predecessor and run one bounded analysis."""
    started = time.perf_counter()
    hashes = source_hashes()
    annotations = ROOT / "data/raw/ecg-reasoning-benchmark/90a53f1/ptbxl.jsonl"
    metadata = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
    replay = replay_predecessor(annotations, metadata)
    table = eligibility_table(metadata)
    partition_receipt = verify_partition(table, metadata, partition, output)
    questions, counts = read_questions(annotations, table)
    groups, inventory = class_groups(questions)
    wave_hashes = attach_features(groups, metadata.parent)
    feature_seconds = time.perf_counter() - started
    predictions, manifests = score_groups(groups)
    result = summarize(predictions, inventory)
    result.update(
        {
            "status": "complete_candidate_selection_feasibility",
            "predecessor_reproduction": replay,
            "cohort_counts": counts,
            "partition": partition_receipt,
            "candidate_descriptions": candidate_descriptions(groups),
            "source_sha256": hashes,
            "input_sha256": {
                to_stored(p): sha256_file(p)
                for p in (annotations, metadata, metadata.with_name("scp_statements.csv"))
            },
            "waveform_sha256": wave_hashes,
            "waveforms_read": len(wave_hashes) // 2,
            "feature_seconds": feature_seconds,
            "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "wfdb", "matplotlib")},
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip(),
            "interpretation": (
                "Generated silver candidate-choice agreement; no free localization or clinical accuracy"
            ),
            "reserved_patients_scored": False,
            "gpu_used": False,
            "training_performed": False,
        }
    )
    result["example"] = illustrate(groups, predictions, metadata.parent, output)
    serializable = [
        {key: value for key, value in q.items() if key not in {"features", "scalars"}}
        for rows in groups.values()
        for q in rows
    ]
    write_json(output / "questions_local.json", serializable)
    write_json(output / "predictions_local.json", predictions)
    write_json(output / "supports_local.json", manifests)
    result["seconds"] = time.perf_counter() - started
    if result["seconds"] > 600:
        raise TimeoutError("Initial ten-minute CPU budget exceeded; preserve this run")
    return result


def main() -> None:
    """Create a fresh output directory and preserve success or failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ntt02_native_reference_transfer")
    parser.add_argument("--partition", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "running",
        "command": [
            "python",
            "-m",
            "scripts.reports.reference_transfer_ntt02",
            "--output",
            to_stored(args.output),
            "--partition",
            to_stored(args.output / "partition_local.json"),
        ],
        "thread_environment": {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json(args.output / "status.json", status)
    try:
        result = run(args.output, args.partition)
        result.update(command=status["command"], thread_environment=status["thread_environment"])
        write_json(args.output / "result.json", result)
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__)
        write_json(args.output / "status.json", status)
        raise
    status.update(status="complete", result_sha256=sha256_file(args.output / "result.json"))
    write_json(args.output / "status.json", status)
    print(
        json.dumps(
            {
                "seconds": result["seconds"],
                "patients": result["patients"],
                "questions": result["unique_questions"],
                "primary": result["primary"],
                "advance": result["advance_gate_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
