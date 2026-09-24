"""Fit a regularized linear probe on frozen ECG embeddings using the fixed label budget."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecg_experiment.data import read_manifest
from ecg_experiment.evaluation import evaluate_predictions, partition_validation
from ecg_experiment.files import write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/processed/ptbxl/seed42_fraction0.1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment001"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    directory = args.output_dir / f"{args.name}_linear_seed{args.seed}"
    if (directory / "metrics.json").exists():
        raise FileExistsError(f"Completed results already exist: {directory}")
    features = np.load(args.embeddings_dir / "features.npy")
    ids = np.load(args.embeddings_dir / "ecg_ids.npy")
    if len(features) != len(ids) or not np.isfinite(features).all():
        raise ValueError("Invalid embeddings")
    index = {int(ecg_id): i for i, ecg_id in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("Duplicate embedding ECG identifiers")
    train = read_manifest(args.manifest_dir / "labeled_train.csv")
    development, calibration = partition_validation(read_manifest(args.manifest_dir / "validation.csv"))
    test = read_manifest(args.manifest_dir / "test.csv")

    def xy(rows):
        return features[[index[int(r["ecg_id"])] for r in rows]], np.array([int(r["target"]) for r in rows])

    train_x, train_y = xy(train)
    dev_x, dev_y = xy(development)
    choices, best_auc, best, best_c = [], -1, None, None
    for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000,
                                                                       solver="lbfgs", random_state=args.seed))
        pipeline.fit(train_x, train_y)
        auc = float(roc_auc_score(dev_y, pipeline.decision_function(dev_x)))
        choices.append({"C": c, "development_auroc": auc})
        if auc > best_auc:
            best_auc, best, best_c = auc, pipeline, c
    result = evaluate_predictions(args.name + "_linear", best.decision_function(xy(calibration)[0]),
                                   best.decision_function(xy(test)[0]), calibration, test, directory, args.seed)
    scaler, classifier = best.steps[0][1], best.steps[1][1]
    np.savez(directory / "linear_model.npz", mean=scaler.mean_, scale=scaler.scale_,
             coefficient=classifier.coef_, intercept=classifier.intercept_)
    write_json_atomic(directory / "config.json", {"model": args.name, "seed": args.seed, "C": best_c,
        "best_development_auroc": best_auc, "regularization_search": choices,
        "labeled_training_records": len(train), "development_records": len(development),
        "calibration_records": len(calibration), "test_records": len(test),
        "embedding_directory": str(args.embeddings_dir),
        "pretraining_exposure": "Released checkpoint may have seen PTB-XL waveforms; see extraction metadata"})


if __name__ == "__main__":
    main()
