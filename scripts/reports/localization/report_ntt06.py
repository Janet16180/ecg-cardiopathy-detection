"""Export only aggregate NTT-06 results and a comparison figure from local receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored

AGGREGATE_FIELDS = (
    "study",
    "stage",
    "status",
    "created_at_utc",
    "git_commit",
    "source_sha256",
    "checkpoint_sha256",
    "predecessor_reproduction",
    "profile_gate_passed",
    "targets",
    "support_patients",
    "unique_waveforms",
    "support_draws",
    "summary",
    "target_vector_cosines",
    "regional_candidates",
    "automatic_qrs_candidates",
    "candidate_assignments_are_calibrated",
    "runtime",
    "embedding_space",
    "context_scope",
    "native_leads",
    "token_anchor_seconds",
    "clinical_localization_accuracy",
    "pretraining_eligibility",
    "coexisting_annotations",
    "inference_seconds",
    "seconds",
    "peak_rss_bytes",
    "gpu_used",
    "weight_training",
    "image_model_input",
)


def main() -> None:
    """Read completed local receipts and write a strict allowlisted aggregate export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=ROOT / "outputs/ntt06_native_encoder/result.json")
    parser.add_argument(
        "--profile", type=Path, default=ROOT / "outputs/ntt06_native_encoder_profile/result.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/localization/ntt06-native-encoder")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    profile = json.loads(args.profile.read_text())
    status = json.loads(args.result.with_name("status.json").read_text())
    if status["status"] != "completed" or status["result_sha256"] != sha256_file(args.result):
        raise ValueError("Final status does not bind the result receipt")
    if result["source_sha256"] != profile["source_sha256"]:
        raise ValueError("Profile and evaluation sources differ")
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = {key: result[key] for key in AGGREGATE_FIELDS}
    aggregate["local_result"] = {"path": to_stored(args.result), "sha256": sha256_file(args.result)}
    aggregate["profile"] = profile
    aggregate["profile_receipt_sha256"] = sha256_file(args.profile)
    write_json_atomic(args.output / "aggregate.json", aggregate)

    import matplotlib.pyplot as plt

    methods = (
        "native_qrs",
        "native_late",
        "native_global_pair",
        "native_global_eight",
        "native_wrong_reference",
        "native_shuffled_labels",
        "raw_full_context",
        "dino",
    )
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.bar(methods, [result["summary"]["balanced_accuracy"][m] for m in methods])
    axis.set(
        ylabel="Balanced record-finding accuracy",
        ylim=(0, 1),
        title="Same 60 development targets; three fixed reference draws",
    )
    axis.tick_params(axis="x", rotation=25)
    axis.axhline(1 / 3, color="black", linestyle="--", linewidth=0.8)
    figure.savefig(args.output / "comparison.png", dpi=150)
    plt.close(figure)
    print(json.dumps({"export": to_stored(args.output), "result_sha256": sha256_file(args.result)}))


if __name__ == "__main__":
    main()
