"""One-class scores compared with the Mahalanobis normal reference in Experiment 034.

Every score is fitted on normal ECG embeddings only and is larger for ECGs farther from normal. The kNN
scores use unit-length embeddings; the others use the Mahalanobis reference's own scaler and PCA, whitened.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import OneClassSVM
from torch import nn

SEED = 38038
KNN_K = 10
HOLDOUT_FRACTION = 0.1
GMM_COMPONENTS = (1, 2, 4, 8, 16)
GMM_INITS = 3
MAX_EPOCHS = 300
PATIENCE = 30
BATCH = 256
LEARNING_RATE = 1e-3
CLIP_NORM = 5.0
SVDD_WEIGHT_DECAY = 1e-6
SVDD_MIN_CENTRE = 0.1
FLOW_LAYERS = 8
FLOW_HIDDEN = 128
CHUNK = 2048
NETWORKS = ("deep_svdd", "autoencoder", "flow")

RowLoss = Callable[[torch.Tensor], torch.Tensor]


class WallTimeExceededError(RuntimeError):
    """Raised when network training passes its wall-time deadline."""


def knn_distances(reference: np.ndarray, query: np.ndarray, k: int = KNN_K) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean distances from unit-length queries to their nearest unit-length fit normals.

    Parameters
    ----------
    reference : np.ndarray
        Raw embeddings of the fit set.
    query : np.ndarray
        Raw embeddings to score; they must not belong to the fit set, since no self-match is removed.
    k : int
        Number of neighbours.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Distance to the k-th nearest neighbour and mean distance to the k nearest, one per query row.
    """
    reference = normalize(np.asarray(reference, dtype=np.float64))
    query = normalize(np.asarray(query, dtype=np.float64))
    kth, mean = [], []
    for start in range(0, len(query), CHUNK):
        squared = np.maximum(2.0 - 2.0 * query[start:start + CHUNK] @ reference.T, 0.0)
        nearest = np.sort(np.partition(squared, k - 1, axis=1)[:, :k], axis=1)
        distances = np.sqrt(nearest)
        kth.append(distances[:, -1])
        mean.append(distances.mean(axis=1))
    return np.concatenate(kth), np.concatenate(mean)


def whiten(model: tuple[StandardScaler, PCA, LedoitWolf], x: np.ndarray) -> np.ndarray:
    """
    Project embeddings on the reference's principal components, each scaled to unit fit-set variance.

    Parameters
    ----------
    model : tuple[StandardScaler, PCA, LedoitWolf]
        Output of ``normal_manifold.fit_mahalanobis``.
    x : np.ndarray
        Embeddings to project.

    Returns
    -------
    np.ndarray
        Whitened component scores, float64.
    """
    scaler, pca, _ = model
    components = pca.transform(scaler.transform(np.asarray(x, dtype=np.float64)))
    return components / np.sqrt(pca.explained_variance_)


def holdout_mask(units: np.ndarray, fraction: float = HOLDOUT_FRACTION, seed: int = SEED) -> np.ndarray:
    """
    Hold out a share of whole units (patients or records) of the fit set for tuning.

    Parameters
    ----------
    units : np.ndarray
        Unit ID of each fit-set row.
    fraction : float
        Share of units held out, rounded up.
    seed : int
        Seed of the ``numpy.random.default_rng`` permutation of the sorted unique units.

    Returns
    -------
    np.ndarray
        True for the rows of held-out units.
    """
    unique = np.unique(units)
    permuted = np.random.default_rng(seed).permutation(len(unique))
    held = unique[permuted[:math.ceil(fraction * len(unique))]]
    return np.isin(units, held)


def ocsvm_scores(fit: np.ndarray, queries: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    One-class SVM with the library defaults.

    Parameters
    ----------
    fit : np.ndarray
        Whitened fit-set rows.
    queries : dict[str, np.ndarray]
        Whitened rows to score, by set.

    Returns
    -------
    dict[str, np.ndarray]
        Minus the decision function, per set.
    """
    model = OneClassSVM(kernel="rbf", nu=0.5, gamma="scale").fit(fit)
    return {name: -model.decision_function(x) for name, x in queries.items()}


def iforest_scores(fit: np.ndarray, queries: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Isolation forest with the library defaults.

    Parameters
    ----------
    fit : np.ndarray
        Whitened fit-set rows.
    queries : dict[str, np.ndarray]
        Whitened rows to score, by set.

    Returns
    -------
    dict[str, np.ndarray]
        Minus the forest's ``score_samples``, per set.
    """
    model = IsolationForest(n_estimators=100, max_samples=256, random_state=SEED).fit(fit)
    return {name: -model.score_samples(x) for name, x in queries.items()}


def gmm(components: int) -> GaussianMixture:
    """
    Unfitted full-covariance Gaussian mixture of the protocol.

    Parameters
    ----------
    components : int
        Number of mixture components.

    Returns
    -------
    GaussianMixture
        The estimator.
    """
    return GaussianMixture(n_components=components, covariance_type="full", n_init=GMM_INITS,
                           random_state=SEED)


def choose_gmm_components(train: np.ndarray, held: np.ndarray) -> tuple[int, dict[int, float]]:
    """
    Choose the number of mixture components by the mean held-out log-likelihood.

    Parameters
    ----------
    train : np.ndarray
        Whitened tuning-training rows.
    held : np.ndarray
        Whitened held-out normals.

    Returns
    -------
    tuple[int, dict[int, float]]
        The chosen number (the smaller on ties) and the held-out mean log-likelihood per candidate.
    """
    likelihood = {k: float(gmm(k).fit(train).score(held)) for k in GMM_COMPONENTS}
    best = max(likelihood.values())
    return min(k for k, value in likelihood.items() if value == best), likelihood


def gmm_scores(fit: np.ndarray, queries: dict[str, np.ndarray], components: int) -> dict[str, np.ndarray]:
    """
    Minus the log density of a Gaussian mixture fitted on the whole fit set.

    Parameters
    ----------
    fit : np.ndarray
        Whitened fit-set rows.
    queries : dict[str, np.ndarray]
        Whitened rows to score, by set.
    components : int
        Number of mixture components.

    Returns
    -------
    dict[str, np.ndarray]
        Minus the log density, per set.
    """
    model = gmm(components).fit(fit)
    return {name: -model.score_samples(x) for name, x in queries.items()}


class Coupling(nn.Module):
    """RealNVP affine coupling: half of the coordinates are shifted and scaled given the other half."""

    def __init__(self, permutation: np.ndarray, hidden: int = FLOW_HIDDEN) -> None:
        """
        Build the coupling.

        Parameters
        ----------
        permutation : np.ndarray
            Order of the coordinates; the first half conditions the second half.
        hidden : int
            Width of the conditioner's hidden layers.
        """
        super().__init__()
        half = len(permutation) // 2
        self.register_buffer("kept", torch.as_tensor(permutation[:half], dtype=torch.long))
        self.register_buffer("changed", torch.as_tensor(permutation[half:], dtype=torch.long))
        self.net = nn.Sequential(nn.Linear(half, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * (len(permutation) - half)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Map towards the base distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of rows.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Transformed rows and the log-determinant of the Jacobian per row.
        """
        log_scale, shift = self.net(x[:, self.kept]).chunk(2, dim=1)
        log_scale = torch.tanh(log_scale)
        y = x.clone()
        y[:, self.changed] = x[:, self.changed] * torch.exp(log_scale) + shift
        return y, log_scale.sum(dim=1)


class RealNVP(nn.Module):
    """Stack of affine couplings on a standard normal base."""

    def __init__(self, dimensions: int, layers: int = FLOW_LAYERS, seed: int = SEED) -> None:
        """
        Build the flow with fixed random coordinate splits.

        Parameters
        ----------
        dimensions : int
            Number of input coordinates.
        layers : int
            Number of couplings.
        seed : int
            Seed of the ``numpy.random.default_rng`` that draws each layer's permutation.
        """
        super().__init__()
        rng = np.random.default_rng(seed)
        self.couplings = nn.ModuleList(Coupling(rng.permutation(dimensions)) for _ in range(layers))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log density of each row.

        Parameters
        ----------
        x : torch.Tensor
            Batch of rows.

        Returns
        -------
        torch.Tensor
            One log density per row.
        """
        total = torch.zeros(len(x), dtype=x.dtype)
        for coupling in self.couplings:
            x, log_det = coupling(x)
            total = total + log_det
        base = -0.5 * (x ** 2).sum(dim=1) - 0.5 * x.shape[1] * math.log(2 * math.pi)
        return base + total


def build_network(kind: str, train: torch.Tensor) -> tuple[nn.Module, RowLoss, float]:
    """
    Build one of the protocol's networks from the fixed seed, with its per-row loss.

    Parameters
    ----------
    kind : str
        ``deep_svdd``, ``autoencoder`` or ``flow``.
    train : torch.Tensor
        Training rows; Deep SVDD takes its centre from the untrained network's mean output on them.

    Returns
    -------
    tuple[nn.Module, RowLoss, float]
        The network, the per-row loss (also the anomaly score) and the weight decay.

    Raises
    ------
    ValueError
        If ``kind`` is unknown.
    """
    torch.manual_seed(SEED)
    dimensions = train.shape[1]
    if kind == "deep_svdd":
        network = nn.Sequential(nn.Linear(dimensions, 128, bias=False), nn.LeakyReLU(0.1),
                                nn.Linear(128, 64, bias=False), nn.LeakyReLU(0.1),
                                nn.Linear(64, 32, bias=False))
        with torch.no_grad():
            centre = network(train).mean(dim=0)
        small = centre.abs() < SVDD_MIN_CENTRE
        centre[small] = torch.where(centre[small] < 0, -SVDD_MIN_CENTRE, SVDD_MIN_CENTRE)
        return network, lambda x: ((network(x) - centre) ** 2).sum(dim=1), SVDD_WEIGHT_DECAY
    if kind == "autoencoder":
        network = nn.Sequential(nn.Linear(dimensions, 128), nn.ReLU(), nn.Linear(128, 16), nn.ReLU(),
                                nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, dimensions))
        return network, lambda x: ((network(x) - x) ** 2).mean(dim=1), 0.0
    if kind == "flow":
        network = RealNVP(dimensions)
        return network, lambda x: -network.log_prob(x), 0.0
    raise ValueError(f"Unknown network: {kind}")


def mean_loss(row_loss: RowLoss, x: torch.Tensor) -> float:
    """
    Mean per-row loss without gradients.

    Parameters
    ----------
    row_loss : RowLoss
        Per-row loss of a network.
    x : torch.Tensor
        Rows.

    Returns
    -------
    float
        Mean loss.
    """
    with torch.no_grad():
        return float(row_loss(x).mean())


def train_network(kind: str, train: np.ndarray, epochs: int, held: np.ndarray | None = None,
                  deadline: float = math.inf) -> tuple[RowLoss, list[float]]:
    """
    Train a network with Adam, measuring the held-out loss after each epoch when held-out rows are given.

    With held-out rows, training stops once ``PATIENCE`` epochs pass without a new lowest held-out loss.

    Parameters
    ----------
    kind : str
        ``deep_svdd``, ``autoencoder`` or ``flow``.
    train : np.ndarray
        Training rows.
    epochs : int
        Maximum number of epochs.
    held : np.ndarray | None
        Held-out normals, or None to train exactly ``epochs`` epochs.
    deadline : float
        ``time.monotonic()`` value after which training stops with an error.

    Returns
    -------
    tuple[RowLoss, list[float]]
        The trained per-row loss (the anomaly score) and the held-out loss after each epoch.

    Raises
    ------
    WallTimeExceededError
        If an epoch ends after ``deadline``.
    """
    rows = torch.as_tensor(train, dtype=torch.float32)
    held_rows = None if held is None else torch.as_tensor(held, dtype=torch.float32)
    network, row_loss, weight_decay = build_network(kind, rows)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(SEED)
    history: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(len(rows), generator=generator)
        for start in range(0, len(rows), BATCH):
            loss = row_loss(rows[order[start:start + BATCH]]).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), CLIP_NORM)
            optimizer.step()
        if time.monotonic() > deadline:
            raise WallTimeExceededError(f"{kind} passed its wall-time cap")
        if held_rows is None:
            continue
        history.append(mean_loss(row_loss, held_rows))
        if len(history) - 1 - int(np.argmin(history)) >= PATIENCE:
            break
    return row_loss, history


def network_scores(kind: str, fit: np.ndarray, held_out: np.ndarray, queries: dict[str, np.ndarray],
                   deadline: float) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """
    Choose the epochs on held-out normals, retrain on the whole fit set, and score every set.

    Parameters
    ----------
    kind : str
        ``deep_svdd``, ``autoencoder`` or ``flow``.
    fit : np.ndarray
        Whitened fit-set rows.
    held_out : np.ndarray
        True for the fit-set rows of the tuning slice.
    queries : dict[str, np.ndarray]
        Whitened rows to score, by set.
    deadline : float
        ``time.monotonic()`` wall-time deadline.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, object]]
        Per-row loss per set, and the chosen epochs with the held-out loss curve.
    """
    _, history = train_network(kind, fit[~held_out], MAX_EPOCHS, fit[held_out], deadline)
    chosen = int(np.argmin(history)) + 1
    row_loss, _ = train_network(kind, fit, chosen, deadline=deadline)
    scores = {}
    with torch.no_grad():
        for name, x in queries.items():
            scores[name] = row_loss(torch.as_tensor(x, dtype=torch.float32)).double().numpy()
    return scores, {"chosen_epochs": chosen, "held_out_loss": history}
