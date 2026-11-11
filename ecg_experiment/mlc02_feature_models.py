"""
Feature-based candidate models of MLC-02, their saved artefacts, and ``predict``.

Recipes of the protocol's Stage 2 table:

- P0, L-J, L-C: one L2 logistic regression per condition on standardized, row-L2-normalized pooled features
  (KEY-04's ``fit_model``: lbfgs, tol 1e-8, max_iter 5000, no class weights, random_state 42; failure to
  converge raises), C from ``LOGISTIC_C`` per condition.
- G-M: one histogram gradient-boosting classifier per condition on the 130 measurements (missing values
  native), learning rate and leaves from the grid, and the number of iterations chosen on validation.
- H-X, H-J, H-C, H-All: standardized input, one hidden layer of 512 ReLU units, dropout 0.3, 14 logits, masked
  binary cross-entropy, AdamW at a constant rate, batch 256, up to 100 epochs.

Every model is trained with undefined labels masked: a per-condition model is fitted on the records where that
condition is defined, and the network's loss averages over the defined entries of a batch.

A fitted candidate lives in ``<directory>/<id>/`` with ``model.json`` (kind, inputs, settings and the SHA-256
of every artefact) and its arrays (``.npz``) or boosting models (``.joblib``). ``predict`` loads them, checks
their hashes, and returns ``[records, 14]`` probabilities for raw feature rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from ecg_experiment import ROOT
from ecg_experiment.condition_keys import unit
from ecg_experiment.files import read_json, sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.key_classifier_models import fit_model
from ecg_experiment.mlc02_data import CONDITIONS, check_probabilities
from ecg_experiment.mlc02_evaluation import selection_metric
from ecg_experiment.mlc02_feature_data import FEATURE_WIDTHS, impute

CANDIDATES = ROOT / "outputs/mlc02_condition_models_20261110/candidates"
LOGISTIC_C = (0.01, 0.1, 1.0, 10.0, 100.0)
LOGISTIC_SEED = 42
BOOSTING_RATES = (0.05, 0.1)
BOOSTING_LEAVES = (15, 31)
BOOSTING_MAX_ITER = 1000
BOOSTING_SEED = 20261110
NETWORK_RATES = (1e-3, 3e-4)
NETWORK_DECAYS = (1e-4, 1e-2)
HIDDEN = 512
DROPOUT = 0.3
BATCH = 256
EPOCHS = 100
SEEDS = (20261110, 20261111, 20261112)
INPUTS = {
    "P0": ("xecg",),
    "L-J": ("jepa",),
    "L-C": ("cpc",),
    "G-M": ("measurements",),
    "H-X": ("xecg",),
    "H-J": ("jepa",),
    "H-C": ("cpc",),
    "H-All": ("xecg", "jepa", "cpc", "measurements"),
}
KINDS = {
    "P0": "logistic",
    "L-J": "logistic",
    "L-C": "logistic",
    "G-M": "boosting",
    "H-X": "network",
    "H-J": "network",
    "H-C": "network",
    "H-All": "network",
}


def fit_scaler(x: np.ndarray) -> dict[str, np.ndarray]:
    """
    Standardizer fitted on training records.

    Parameters
    ----------
    x : np.ndarray
        ``[training records, features]``.

    Returns
    -------
    dict[str, np.ndarray]
        ``mean`` and ``scale`` of ``sklearn.preprocessing.StandardScaler``.
    """
    scaler = StandardScaler().fit(np.asarray(x, dtype=np.float64))
    return {"mean": scaler.mean_, "scale": scaler.scale_}


def standardize(x: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    """
    Apply a fitted standardizer, as ``StandardScaler.transform`` does.

    Parameters
    ----------
    x : np.ndarray
        ``[records, features]``.
    scaler : dict[str, np.ndarray]
        Output of ``fit_scaler``.

    Returns
    -------
    np.ndarray
        Float64 ``(x - mean) / scale``.
    """
    return (np.asarray(x, dtype=np.float64) - scaler["mean"]) / scaler["scale"]


def logistic_inputs(x: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    """
    Standardized, row-L2-normalized pooled features, the input of P0, L-J and L-C.

    Parameters
    ----------
    x : np.ndarray
        Raw pooled features.
    scaler : dict[str, np.ndarray]
        Training standardizer.

    Returns
    -------
    np.ndarray
        Unit rows.
    """
    return unit(standardize(x, scaler))


def network_inputs(
    candidate: str, features: dict[str, np.ndarray], medians: np.ndarray | None = None
) -> np.ndarray:
    """
    Concatenated raw inputs of a network candidate, before standardization.

    Parameters
    ----------
    candidate : str
        H-X, H-J, H-C or H-All.
    features : dict[str, np.ndarray]
        Raw feature matrices of the same records.
    medians : np.ndarray | None
        Training medians of the measurements; required when the candidate uses them.

    Returns
    -------
    np.ndarray
        Float64 ``[records, inputs]``; measurements are imputed and followed by their missing flags.
    """
    parts = [
        impute(features[name], medians) if name == "measurements" else np.asarray(features[name], float)
        for name in INPUTS[candidate]
    ]
    return np.hstack(parts)


def defined_rows(y: np.ndarray) -> np.ndarray:
    """
    Mark the records where a condition is defined, checked to hold both classes.

    Parameters
    ----------
    y : np.ndarray
        States 1.0, 0.0 or NaN of one condition.

    Returns
    -------
    np.ndarray
        Boolean mask of defined records.

    Raises
    ------
    ValueError
        If the defined records do not hold both classes.
    """
    defined = ~np.isnan(y)
    positives = int(y[defined].sum())
    if positives == 0 or positives == int(defined.sum()):
        raise ValueError("a condition needs defined positives and negatives to be fitted")
    return defined


def fit_logistic(z: np.ndarray, y: np.ndarray, c: float) -> dict[str, Any]:
    """
    KEY-04's logistic regression for one condition, fitted on the records where it is defined.

    Parameters
    ----------
    z : np.ndarray
        Unit training rows.
    y : np.ndarray
        States of one condition; NaN entries are left out.
    c : float
        Inverse L2 regularization strength.

    Returns
    -------
    dict[str, Any]
        ``coef`` ``[features]``, ``intercept``, ``iterations`` and ``records``.

    Raises
    ------
    RuntimeError
        If lbfgs does not converge (from ``key_classifier_models.fit_model``).
    """
    defined = defined_rows(y)
    model = fit_model(z[defined], y[defined], c)
    return {
        "coef": model.coef_[0].copy(),
        "intercept": float(model.intercept_[0]),
        "iterations": int(np.max(model.n_iter_)),
        "records": int(defined.sum()),
    }


def logistic_probability(z: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """
    Probabilities of independent logistic regressions.

    Parameters
    ----------
    z : np.ndarray
        Unit rows.
    coef : np.ndarray
        ``[conditions, features]`` coefficients.
    intercept : np.ndarray
        ``[conditions]`` intercepts.

    Returns
    -------
    np.ndarray
        ``[records, conditions]`` probabilities.
    """
    return expit(z @ np.asarray(coef).T + np.asarray(intercept))


def choose_per_condition(
    labels: np.ndarray,
    sources: np.ndarray,
    probabilities: list[np.ndarray],
    conditions: tuple[str, ...] = CONDITIONS,
) -> dict[str, Any]:
    """
    Choose a grid value per condition on that condition's validation metric.

    The metric is the condition's within-source AUROC averaged over the validation sources where it enters
    (``selection_metric(...)["per_condition"]``). Exact ties go to the earliest grid value.

    Parameters
    ----------
    labels : np.ndarray
        ``[validation records, conditions]`` states.
    sources : np.ndarray
        Validation source of each record.
    probabilities : list[np.ndarray]
        One ``[records, conditions]`` probability matrix per grid value, in grid order.
    conditions : tuple[str, ...]
        Condition names in column order.

    Returns
    -------
    dict[str, Any]
        ``scores`` (``[values, conditions]``), ``chosen`` (grid index per condition), ``ties`` (conditions
        whose best value is shared) and ``probability`` (each condition's column from its chosen value).

    Raises
    ------
    ValueError
        If a condition enters no validation source.
    """
    tables = [selection_metric(labels, p, sources, conditions)["per_condition"] for p in probabilities]
    scores = np.array([[table[name] for name in conditions] for table in tables])
    if not np.isfinite(scores).all():
        raise ValueError("every condition needs a validation source where it enters")
    chosen = np.argmax(scores, axis=0)
    ties = [
        name
        for column, name in enumerate(conditions)
        if (scores[:, column] == scores[chosen[column], column]).sum() > 1
    ]
    probability = np.column_stack([probabilities[chosen[c]][:, c] for c in range(len(conditions))])
    return {"scores": scores, "chosen": chosen, "ties": ties, "probability": probability}


def fit_boosting(
    x: np.ndarray, y: np.ndarray, rate: float, leaves: int, iterations: int, seed: int = BOOSTING_SEED
) -> HistGradientBoostingClassifier:
    """
    Histogram gradient boosting for one condition, fitted on the records where it is defined.

    Internal early stopping is off: the number of iterations is chosen on the validation role instead
    (``staged_condition_scores``).

    Parameters
    ----------
    x : np.ndarray
        ``[training records, 130]`` measurements, NaN where missing.
    y : np.ndarray
        States of one condition; NaN entries are left out.
    rate : float
        Learning rate.
    leaves : int
        Maximum leaves per tree.
    iterations : int
        Boosting iterations.
    seed : int
        ``random_state``.

    Returns
    -------
    HistGradientBoostingClassifier
        The fitted model.
    """
    defined = defined_rows(y)
    model = HistGradientBoostingClassifier(
        learning_rate=rate,
        max_leaf_nodes=leaves,
        max_iter=iterations,
        early_stopping=False,
        random_state=seed,
    )
    return model.fit(x[defined], y[defined].astype(int))


def staged_condition_scores(
    model: HistGradientBoostingClassifier, x: np.ndarray, y: np.ndarray, sources: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score one condition on validation after every boosting iteration.

    Each iteration's probabilities become one column scored against the same labels, so the library's
    per-condition selection variant gives each iteration's within-source AUROC averaged over the sources where
    the condition enters.

    Parameters
    ----------
    model : HistGradientBoostingClassifier
        Fitted model.
    x : np.ndarray
        Validation measurements.
    y : np.ndarray
        Validation states of the condition.
    sources : np.ndarray
        Validation source of each record.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Metric per iteration (``[iterations]``) and staged probabilities (``[records, iterations]``).
    """
    staged = np.column_stack([proba[:, 1] for proba in model.staged_predict_proba(x)])
    names = tuple(f"iteration_{i + 1}" for i in range(staged.shape[1]))
    labels = np.repeat(np.asarray(y, dtype=np.float64)[:, None], staged.shape[1], axis=1)
    table = selection_metric(labels, staged, sources, names)["per_condition"]
    return np.array([table[name] for name in names]), staged


def build_network(inputs: int) -> Any:
    """
    One hidden layer of ``HIDDEN`` ReLU units with dropout, and one logit per condition.

    Parameters
    ----------
    inputs : int
        Input width.

    Returns
    -------
    torch.nn.Sequential
        Linear, ReLU, Dropout, Linear.
    """
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(inputs, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Dropout(DROPOUT),
        torch.nn.Linear(HIDDEN, len(CONDITIONS)),
    )


def masked_bce(logits: Any, labels: Any) -> Any:
    """
    Binary cross-entropy averaged over the defined entries of a batch.

    Parameters
    ----------
    logits : torch.Tensor
        ``[batch, conditions]`` logits.
    labels : torch.Tensor
        ``[batch, conditions]`` states 1.0, 0.0 or NaN.

    Returns
    -------
    torch.Tensor
        Scalar loss; undefined entries contribute nothing, not even to the gradient.
    """
    import torch

    defined = ~torch.isnan(labels)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(labels, nan=0.0), reduction="none"
    )
    return (losses * defined).sum() / defined.sum().clamp_min(1)


def network_weights(model: Any) -> dict[str, np.ndarray]:
    """
    Parameters of ``build_network`` as float32 arrays.

    Parameters
    ----------
    model : torch.nn.Sequential
        Network of ``build_network``.

    Returns
    -------
    dict[str, np.ndarray]
        ``w1``, ``b1``, ``w2`` and ``b2``.
    """
    state = model.state_dict()
    names = {"w1": "0.weight", "b1": "0.bias", "w2": "3.weight", "b2": "3.bias"}
    return {short: state[long].detach().cpu().numpy().copy() for short, long in names.items()}


def network_probability(z: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    """
    Evaluation-mode probabilities of one network, in float64 with NumPy.

    Parameters
    ----------
    z : np.ndarray
        Standardized inputs.
    weights : dict[str, np.ndarray]
        Output of ``network_weights``.

    Returns
    -------
    np.ndarray
        ``[records, conditions]`` probabilities.
    """
    w = {name: np.asarray(value, dtype=np.float64) for name, value in weights.items()}
    hidden = np.maximum(np.asarray(z, dtype=np.float64) @ w["w1"].T + w["b1"], 0.0)
    return expit(hidden @ w["w2"].T + w["b2"])


def epoch_order(records: int, generator: Any, device: str) -> Any:
    """
    Shuffled record order of one epoch.

    Parameters
    ----------
    records : int
        Training records.
    generator : torch.Generator
        Seeded CPU generator.
    device : str
        Torch device of the training tensors.

    Returns
    -------
    torch.Tensor
        A permutation on ``device``.
    """
    import torch

    return torch.randperm(records, generator=generator).to(device)


def train_epoch(model: Any, optimizer: Any, x: Any, y: Any, order: Any) -> float:
    """
    One pass over the training records in mini-batches of ``BATCH``.

    Parameters
    ----------
    model : torch.nn.Module
        Network in training mode.
    optimizer : torch.optim.Optimizer
        AdamW over the network's parameters.
    x, y : torch.Tensor
        Training inputs and states on the device.
    order : torch.Tensor
        Record order of this epoch.

    Returns
    -------
    float
        Mean batch loss.
    """
    import torch

    model.train()
    total = torch.zeros((), device=x.device)
    batches = 0
    for start in range(0, len(order), BATCH):
        index = order[start : start + BATCH]
        loss = masked_bce(model(x[index]), y[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.detach()
        batches += 1
    return float(total) / batches


def evaluate_network(model: Any, x: Any) -> np.ndarray:
    """
    Evaluation-mode probabilities on the training device.

    Parameters
    ----------
    model : torch.nn.Module
        Network.
    x : torch.Tensor
        Inputs on the device.

    Returns
    -------
    np.ndarray
        Float64 ``[records, conditions]`` probabilities.
    """
    import torch

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x)).double().cpu().numpy()


def train_network(
    x: np.ndarray,
    y: np.ndarray,
    rate: float,
    decay: float,
    seed: int,
    epochs: int,
    device: str,
    x_val: np.ndarray | None = None,
    score: Any = None,
) -> dict[str, Any]:
    """
    Train one network; with validation inputs, keep the weights of the best epoch.

    Parameters
    ----------
    x : np.ndarray
        Standardized training inputs.
    y : np.ndarray
        Training states with NaN where undefined.
    rate, decay : float
        AdamW learning rate and weight decay.
    seed : int
        Seed of the initialization, dropout and shuffling.
    epochs : int
        Epochs to run.
    device : str
        Torch device.
    x_val : np.ndarray | None
        Standardized validation inputs, scored after every epoch; None keeps the last epoch.
    score : callable | None
        Validation probabilities to the selection metric; required with ``x_val``.

    Returns
    -------
    dict[str, Any]
        ``weights`` and ``epoch`` kept, ``metric`` and ``probability`` (validation, None without it), and
        ``history`` (epoch, training loss, validation metric, seconds).
    """
    import time

    import torch

    torch.manual_seed(seed)
    model = build_network(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=decay)
    generator = torch.Generator().manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    xv = None if x_val is None else torch.as_tensor(x_val, dtype=torch.float32, device=device)
    best = {"metric": -np.inf, "epoch": epochs, "probability": None, "weights": None}
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        loss = train_epoch(model, optimizer, xt, yt, epoch_order(len(x), generator, device))
        metric = None
        if xv is not None:
            probability = evaluate_network(model, xv)
            metric = float(score(probability))
            if metric > best["metric"]:
                best = {
                    "metric": metric,
                    "epoch": epoch,
                    "probability": probability,
                    "weights": network_weights(model),
                }
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "validation_metric": metric,
                "seconds": time.perf_counter() - started,
            }
        )
    if xv is None:
        best = {"metric": None, "epoch": epochs, "probability": None, "weights": network_weights(model)}
    return {**best, "history": history}


def file_hashes(directory: Path, names: list[str]) -> dict[str, str]:
    """
    SHA-256 of artefact files.

    Parameters
    ----------
    directory : Path
        Candidate directory.
    names : list[str]
        File names inside it.

    Returns
    -------
    dict[str, str]
        Name to digest.
    """
    return {name: sha256_file(Path(directory) / name) for name in names}


def write_model(
    directory: Path, candidate: str, files: list[str], settings: dict[str, Any], seeds: tuple[int, ...]
) -> dict[str, Any]:
    """
    Write ``model.json``, the index of a fitted candidate.

    Parameters
    ----------
    directory : Path
        Candidate directory.
    candidate : str
        Candidate id.
    files : list[str]
        Artefact file names written before.
    settings : dict[str, Any]
        Chosen settings.
    seeds : tuple[int, ...]
        Seeds whose mean probability ``predict`` returns.

    Returns
    -------
    dict[str, Any]
        The written index.
    """
    model = {
        "candidate": candidate,
        "kind": KINDS[candidate],
        "inputs": list(INPUTS[candidate]),
        "conditions": list(CONDITIONS),
        "seeds": list(seeds),
        "settings": settings,
        "files": file_hashes(directory, files),
    }
    write_json_atomic(Path(directory) / "model.json", model)
    return model


def save_logistic(
    directory: Path,
    candidate: str,
    scaler: dict[str, np.ndarray],
    fits: dict[str, np.ndarray],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """
    Save a fitted logistic candidate.

    Parameters
    ----------
    directory : Path
        Candidate directory.
    candidate : str
        P0, L-J or L-C.
    scaler : dict[str, np.ndarray]
        Training standardizer.
    fits : dict[str, np.ndarray]
        ``coef`` ``[14, features]``, ``intercept``, ``c`` and ``iterations`` of the chosen fits.
    settings : dict[str, Any]
        Chosen settings.

    Returns
    -------
    dict[str, Any]
        ``model.json`` content.
    """
    directory = Path(directory)
    write_npz_atomic(directory / "scaler.npz", **scaler)
    write_npz_atomic(directory / "logistic.npz", **fits)
    return write_model(directory, candidate, ["scaler.npz", "logistic.npz"], settings, (LOGISTIC_SEED,))


def save_boosting(
    directory: Path, models: list[HistGradientBoostingClassifier], settings: dict[str, Any]
) -> dict[str, Any]:
    """
    Save the 14 fitted boosting models of G-M.

    Parameters
    ----------
    directory : Path
        Candidate directory.
    models : list[HistGradientBoostingClassifier]
        One model per condition, in condition order.
    settings : dict[str, Any]
        Chosen settings.

    Returns
    -------
    dict[str, Any]
        ``model.json`` content.
    """
    import joblib

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name, model in zip(CONDITIONS, models, strict=True):
        files.append(f"boosting__{name}.joblib")
        joblib.dump(model, directory / files[-1])
    return write_model(directory, "G-M", files, settings, (BOOSTING_SEED,))


def save_network(
    directory: Path,
    candidate: str,
    scaler: dict[str, np.ndarray],
    medians: np.ndarray | None,
    weights: dict[int, dict[str, np.ndarray]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """
    Save a fitted network candidate: standardizer, imputation medians and the weights of every seed.

    Parameters
    ----------
    directory : Path
        Candidate directory.
    candidate : str
        H-X, H-J, H-C or H-All.
    scaler : dict[str, np.ndarray]
        Training standardizer of the concatenated inputs.
    medians : np.ndarray | None
        Training medians of the measurements (H-All).
    weights : dict[int, dict[str, np.ndarray]]
        Seed to ``network_weights``.
    settings : dict[str, Any]
        Chosen settings.

    Returns
    -------
    dict[str, Any]
        ``model.json`` content.
    """
    directory = Path(directory)
    write_npz_atomic(directory / "scaler.npz", **scaler)
    files = ["scaler.npz"]
    if medians is not None:
        write_npz_atomic(directory / "imputation.npz", medians=medians)
        files.append("imputation.npz")
    for seed, values in weights.items():
        files.append(f"weights__seed-{seed}.npz")
        write_npz_atomic(directory / files[-1], **values)
    return write_model(directory, candidate, files, settings, tuple(weights))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """
    Read an ``.npz`` archive without pickle.

    Parameters
    ----------
    path : Path
        Archive.

    Returns
    -------
    dict[str, np.ndarray]
        Its arrays.
    """
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_candidate(candidate: str, directory: Path = CANDIDATES) -> dict[str, Any]:
    """
    Read a fitted candidate after checking the SHA-256 of every artefact.

    Parameters
    ----------
    candidate : str
        Candidate id.
    directory : Path
        Directory holding ``<id>/model.json``.

    Returns
    -------
    dict[str, Any]
        ``model.json`` content and ``artefacts`` (file name to loaded content).

    Raises
    ------
    ValueError
        If the index names another candidate or an artefact's hash differs.
    """
    import joblib

    folder = Path(directory) / candidate
    model = read_json(folder / "model.json")
    if model["candidate"] != candidate or tuple(model["conditions"]) != CONDITIONS:
        raise ValueError(f"{folder} does not hold candidate {candidate}")
    if file_hashes(folder, list(model["files"])) != model["files"]:
        raise ValueError(f"artefacts of {candidate} differ from model.json")
    artefacts = {
        name: joblib.load(folder / name) if name.endswith(".joblib") else load_npz(folder / name)
        for name in model["files"]
    }
    return {**model, "artefacts": artefacts}


def check_features(model: dict[str, Any], features: dict[str, np.ndarray]) -> int:
    """
    Check that the feature rows given to ``predict`` cover the candidate's inputs.

    Parameters
    ----------
    model : dict[str, Any]
        Output of ``load_candidate``.
    features : dict[str, np.ndarray]
        Raw feature matrices.

    Returns
    -------
    int
        Number of records.

    Raises
    ------
    ValueError
        If an input is missing, has another width, or the matrices differ in length.
    """
    lengths = set()
    for name in model["inputs"]:
        if name not in features or np.shape(features[name])[1:] != (FEATURE_WIDTHS[name],):
            raise ValueError(f"predict needs {name} features of width {FEATURE_WIDTHS[name]}")
        lengths.add(len(features[name]))
    if len(lengths) != 1:
        raise ValueError("feature matrices must describe the same records")
    return lengths.pop()


def predict_loaded(
    model: dict[str, Any], features: dict[str, np.ndarray], seeds: tuple[int, ...] | None = None
) -> np.ndarray:
    """
    Probabilities of a loaded candidate.

    Parameters
    ----------
    model : dict[str, Any]
        Output of ``load_candidate``.
    features : dict[str, np.ndarray]
        Raw feature matrices of the records to score.
    seeds : tuple[int, ...] | None
        Network seeds to average; None uses all of the candidate's seeds.

    Returns
    -------
    np.ndarray
        Float64 ``[records, 14]``.
    """
    records = check_features(model, features)
    files = model["artefacts"]
    if model["kind"] == "logistic":
        z = logistic_inputs(features[model["inputs"][0]], files["scaler.npz"])
        probability = logistic_probability(
            z, files["logistic.npz"]["coef"], files["logistic.npz"]["intercept"]
        )
    elif model["kind"] == "boosting":
        x = np.asarray(features["measurements"], dtype=np.float64)
        probability = np.column_stack(
            [files[f"boosting__{name}.joblib"].predict_proba(x)[:, 1] for name in CONDITIONS]
        )
    else:
        medians = files["imputation.npz"]["medians"] if "imputation.npz" in files else None
        z = standardize(network_inputs(model["candidate"], features, medians), files["scaler.npz"])
        chosen = tuple(model["seeds"]) if seeds is None else tuple(seeds)
        if not set(chosen) <= set(model["seeds"]) or not chosen:
            raise ValueError(f"seeds {chosen} are not among {model['seeds']}")
        probability = np.mean(
            [network_probability(z, files[f"weights__seed-{seed}.npz"]) for seed in chosen], axis=0
        )
    return check_probabilities(probability, records, len(CONDITIONS))


def predict(
    candidate_id: str,
    features: dict[str, np.ndarray],
    directory: Path = CANDIDATES,
    seeds: tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Probabilities of a saved candidate for raw feature rows.

    Parameters
    ----------
    candidate_id : str
        P0, L-J, L-C, G-M, H-X, H-J, H-C or H-All.
    features : dict[str, np.ndarray]
        Raw Stage 1 features of the records to score, aligned row by row: ``xecg`` ``[n, 1024]``, ``jepa``
        ``[n, 768]``, ``cpc`` ``[n, 512]`` and ``measurements`` ``[n, 130]`` (NaN where unavailable), as far
        as the candidate uses them.
    directory : Path
        Directory holding the candidates.
    seeds : tuple[int, ...] | None
        Network seeds to average; None gives the candidate's prediction (all of its seeds).

    Returns
    -------
    np.ndarray
        Float64 ``[n, 14]`` probabilities in ``CONDITIONS`` order.
    """
    return predict_loaded(load_candidate(candidate_id, directory), features, seeds)
