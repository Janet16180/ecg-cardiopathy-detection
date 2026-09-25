"""Fit a regularized linear probe on frozen ECG embeddings using the fixed label budget."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ecg_experiment.evaluation import PROBE_C_GRID, evaluate_predictions, partition_validation
from ecg_experiment.files import read_csv, write_json_atomic, write_npz_atomic
from ecg_experiment.receipts import artifacts_exist

ROOT = Path(__file__).resolve().parents[2]
MAX_ITERATIONS = 3000
ARTIFACTS = ("metrics.json", "test_predictions.csv", "calibration_predictions.npz",
             "linear_model.npz", "config.json")

Rows = list[dict[str, str]]


def load_embeddings(directory: Path) -> tuple[np.ndarray, dict[int, int]]:
    """
    Load frozen features and index them by ECG identifier.

    Parameters
    ----------
    directory : Path
        Directory with ``features.npy`` and ``ecg_ids.npy``.

    Returns
    -------
    tuple[np.ndarray, dict[int, int]]
        Feature matrix and feature row per integer ECG identifier.

    Raises
    ------
    ValueError
        If the features are misaligned, nonfinite, or have duplicate identifiers.
    """
    features = np.load(directory / "features.npy")
    ids = np.load(directory / "ecg_ids.npy")
    if len(features) != len(ids) or not np.isfinite(features).all():
        raise ValueError("Invalid embeddings")
    index = {int(ecg_id): i for i, ecg_id in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("Duplicate embedding ECG identifiers")
    return features, index


def features_and_targets(features: np.ndarray, index: dict[int, int],
                         rows: Rows) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the features and binary targets of manifest rows.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    index : dict[int, int]
        Feature row per integer ECG identifier.
    rows : list[dict[str, str]]
        Manifest rows with ``ecg_id`` and ``target``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Features and integer targets in row order.
    """
    return features[[index[int(r["ecg_id"])] for r in rows]], np.array([int(r["target"]) for r in rows])


def select_probe(train_x: np.ndarray, train_y: np.ndarray, dev_x: np.ndarray, dev_y: np.ndarray,
                 seed: int) -> tuple[Pipeline, float, float, list[dict[str, float]]]:
    """
    Choose the logistic-regression strength by development AUROC; the first best wins ties.

    Parameters
    ----------
    train_x : np.ndarray
        Training features.
    train_y : np.ndarray
        Training targets.
    dev_x : np.ndarray
        Development features.
    dev_y : np.ndarray
        Development targets.
    seed : int
        Solver random state.

    Returns
    -------
    tuple[Pipeline, float, float, list[dict[str, float]]]
        Best fitted pipeline, its ``C``, its development AUROC, and the search record.

    Raises
    ------
    RuntimeError
        If no candidate produced a comparable AUROC.
    """
    choices, best_auc, best, best_c = [], -1, None, None
    for c in PROBE_C_GRID:
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=MAX_ITERATIONS,
                                                                       solver="lbfgs", random_state=seed))
        pipeline.fit(train_x, train_y)
        auc = float(roc_auc_score(dev_y, pipeline.decision_function(dev_x)))
        choices.append({"C": c, "development_auroc": auc})
        if auc > best_auc:
            best_auc, best, best_c = auc, pipeline, c
    if best is None:
        raise RuntimeError("No regularization candidate produced a development AUROC")
    return best, best_c, best_auc, choices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--manifest-dir", type=Path,
                        default=ROOT / "data/processed/ptbxl/seed42_fraction0.1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiment001")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Fit, select, and evaluate the linear probe, then save its weights and configuration.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    directory = args.output_dir / f"{args.name}_linear_seed{args.seed}"
    if artifacts_exist(directory, ARTIFACTS):
        raise FileExistsError(f"Completed results already exist: {directory}")
    features, index = load_embeddings(args.embeddings_dir)
    train = read_csv(args.manifest_dir / "labeled_train.csv")
    development, calibration = partition_validation(read_csv(args.manifest_dir / "validation.csv"))
    test = read_csv(args.manifest_dir / "test.csv")
    train_x, train_y = features_and_targets(features, index, train)
    dev_x, dev_y = features_and_targets(features, index, development)
    best, best_c, best_auc, choices = select_probe(train_x, train_y, dev_x, dev_y, args.seed)
    evaluate_predictions(args.name + "_linear",
                         best.decision_function(features_and_targets(features, index, calibration)[0]),
                         best.decision_function(features_and_targets(features, index, test)[0]),
                         calibration, test, directory, args.seed)
    scaler, classifier = best.steps[0][1], best.steps[1][1]
    write_npz_atomic(directory / "linear_model.npz", mean=scaler.mean_, scale=scaler.scale_,
                     coefficient=classifier.coef_, intercept=classifier.intercept_)
    write_json_atomic(directory / "config.json", {
        "model": args.name, "seed": args.seed, "C": best_c,
        "best_development_auroc": best_auc, "regularization_search": choices,
        "labeled_training_records": len(train), "development_records": len(development),
        "calibration_records": len(calibration), "test_records": len(test),
        "embedding_directory": str(args.embeddings_dir),
        "pretraining_exposure": ("Released checkpoint may have seen PTB-XL waveforms; "
                                 "see extraction metadata")})


if __name__ == "__main__":
    main()
