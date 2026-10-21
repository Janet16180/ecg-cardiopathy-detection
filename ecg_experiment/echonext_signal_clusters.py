"""Experiment 042: CNN + Transformer autoencoder on every open EchoNext ECG, then unsupervised clusters.

The autoencoder learns from waveforms alone: it receives 12 leads x 2,500 samples and reconstructs them.
Labels from the release CSV are read only by ``clinical_profile``, after the clusters are fixed.

Only the ``train`` and ``val`` splits are opened. ``test`` stays closed, and ``no_split`` is left out
because its patients also appear in ``val`` and ``test``.

The release stores float64 arrays of shape ``(N, 1, 2500, 12)``; ``train`` alone is 17.4 GB, more than the
memory of a 16 GB machine. ``build_signal_cache`` writes them once as float16 ``(N, 12, 2500)``, 4.3 GB
for ``train``, which the operating system keeps in memory so that shuffled batches do not wait for disk.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch import nn

RELEASE = "echonext-a-dataset-for-detecting-echocardiogram-confirmed-structural-heart-disease-from-ecgs-1.1.0"
RAW_DIR = Path("data/raw") / RELEASE
CACHE_DIR = Path("data/processed/echonext_signal_float16_v1")
METADATA_FILE = "echonext_metadata_100k.csv"
OPEN_SPLITS = ("train", "val")
SAMPLING_RATE = 250
LEADS = 12
SAMPLES = 2500
CACHE_CHUNK = 2048
COMPOSITE = "shd_moderate_or_greater_flag"
VALVE_FLAGS = (
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
)
PROFILE_FLAGS = (COMPOSITE, *VALVE_FLAGS)
CLUSTER_CANDIDATES = (2, 3, 4, 5, 6, 7, 8)
SILHOUETTE_SAMPLE = 5000
PCA_VARIANCE = 0.95


def best_device() -> torch.device:
    """
    Pick the fastest available device: Apple GPU (MPS), then CUDA, then CPU.

    Returns
    -------
    torch.device
        The device to train on.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_signal_cache(root: Path, split: str) -> Path:
    """
    Write one split of the release as a float16 ``(N, 12, 2500)`` array, once.

    Parameters
    ----------
    root : Path
        Repository root.
    split : str
        ``"train"`` or ``"val"``.

    Returns
    -------
    Path
        The cached array. An existing cache is reused when its receipt matches the source file.

    Raises
    ------
    ValueError
        If the split is not open, the source has an unexpected shape, or a value does not survive float16.
    """
    if split not in OPEN_SPLITS:
        raise ValueError(f"Only {OPEN_SPLITS} may be opened, not {split!r}")
    source_path = root / RAW_DIR / f"EchoNext_{split}_waveforms.npy"
    target = root / CACHE_DIR / f"{split}.npy"
    receipt_path = target.with_suffix(".json")
    source_bytes = source_path.stat().st_size
    if target.exists() and receipt_path.exists() and \
            json.loads(receipt_path.read_text())["source_bytes"] == source_bytes:
        return target
    source = np.load(source_path, mmap_mode="r")
    if source.shape[1:] != (1, SAMPLES, LEADS):
        raise ValueError(f"Expected (N, 1, {SAMPLES}, {LEADS}), got {source.shape}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.npy")
    cache = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16,
                                      shape=(len(source), LEADS, SAMPLES))
    nonfinite, largest, started = 0, 0.0, time.perf_counter()
    for start in range(0, len(source), CACHE_CHUNK):
        chunk = np.asarray(source[start:start + CACHE_CHUNK, 0]).transpose(0, 2, 1)
        finite = np.isfinite(chunk)
        nonfinite += int((~finite.all(axis=(1, 2))).sum())
        largest = max(largest, float(np.abs(chunk[finite]).max()))
        cache[start:start + CACHE_CHUNK] = chunk
        done = min(start + CACHE_CHUNK, len(source))
        print(f"\rCaché {split}: {done:,}/{len(source):,} ECG", end="", flush=True)
    print(f" en {time.perf_counter() - started:.0f} s")
    if largest >= float(np.finfo(np.float16).max):
        raise ValueError(f"Value {largest} of {split} does not fit float16")
    cache.flush()
    del cache
    os.replace(partial, target)
    receipt = {"split": split, "records": len(source), "source_bytes": source_bytes, "dtype": "float16",
               "records_with_nonfinite_samples": nonfinite, "largest_absolute_value": largest}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return target


def load_open_splits(root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Read the metadata rows and cached waveforms of ``train`` and ``val``, never those of ``test``.

    The release CSV lists ``train``, then ``val``, then the other splits, so reading stops after ``val``.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray]]
        Metadata with a ``row`` column (position inside the split's array), and one read-only memory-mapped
        float16 ``(N, 12, 2500)`` array per split.

    Raises
    ------
    ValueError
        If the open rows are not first in the CSV, counts differ from the arrays, or ``train`` and ``val``
        share a patient.
    """
    csv_path = root / RAW_DIR / METADATA_FILE
    split = pd.read_csv(csv_path, usecols=["split"])["split"]
    is_open = split.isin(OPEN_SPLITS).to_numpy()
    count = int(is_open.sum())
    if not is_open[:count].all():
        raise ValueError("The open splits are not the first rows of the metadata")
    metadata = pd.read_csv(csv_path, nrows=count).drop(columns="Unnamed: 0", errors="ignore")
    metadata["row"] = metadata.groupby("split").cumcount()
    waveforms = {name: np.load(build_signal_cache(root, name), mmap_mode="r") for name in OPEN_SPLITS}
    for name in OPEN_SPLITS:
        if int(metadata["split"].eq(name).sum()) != len(waveforms[name]):
            raise ValueError(f"Metadata and waveform counts differ for {name}")
    patients = {name: set(metadata.loc[metadata["split"].eq(name), "patient_key"]) for name in OPEN_SPLITS}
    if patients["train"] & patients["val"]:
        raise ValueError("train and val share patients")
    return metadata, waveforms


class SignalDataset(torch.utils.data.Dataset):
    """
    Waveforms of one split, served as float32 tensors without labels.

    Parameters
    ----------
    waveforms : np.ndarray
        Array of shape ``(N, 12, 2500)``, usually a float16 memory map.
    """

    def __init__(self, waveforms: np.ndarray) -> None:
        self.waveforms = waveforms

    def __len__(self) -> int:
        """Return the number of ECGs."""
        return len(self.waveforms)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return one ECG as a float32 tensor of shape ``(12, 2500)``."""
        return torch.from_numpy(np.asarray(self.waveforms[index], dtype=np.float32))

    def batch(self, indices: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Read several ECGs at once and move them to the device.

        Parameters
        ----------
        indices : np.ndarray
            Positions in the split; they are read in ascending order, which is faster on a memory map.
        device : torch.device
            Where the batch is placed.

        Returns
        -------
        torch.Tensor
            Float32 tensor of shape ``(len(indices), 12, 2500)``.
        """
        block = np.ascontiguousarray(self.waveforms[np.sort(indices)])
        return torch.from_numpy(block).to(device).float()


def full_split_datasets(metadata: pd.DataFrame, waveforms: dict[str, np.ndarray]
                        ) -> tuple[pd.DataFrame, SignalDataset, pd.DataFrame, SignalDataset]:
    """
    Build the training and validation datasets over every ECG of each split.

    Parameters
    ----------
    metadata : pd.DataFrame
        Output of ``load_open_splits``.
    waveforms : dict[str, np.ndarray]
        Output of ``load_open_splits``.

    Returns
    -------
    tuple[pd.DataFrame, SignalDataset, pd.DataFrame, SignalDataset]
        Training rows and dataset, then validation rows and dataset; row ``i`` describes ECG ``i``.
    """
    rows = {name: metadata[metadata["split"].eq(name)].sort_values("row").reset_index(drop=True)
            for name in OPEN_SPLITS}
    return rows["train"], SignalDataset(waveforms["train"]), rows["val"], SignalDataset(waveforms["val"])


def conv_block(module: nn.Module, channels: int) -> nn.Sequential:
    """
    Follow a convolution with group normalization and a GELU.

    Parameters
    ----------
    module : nn.Module
        The convolution or transposed convolution.
    channels : int
        Its output channels.

    Returns
    -------
    nn.Sequential
        The three layers.
    """
    return nn.Sequential(module, nn.GroupNorm(8, channels), nn.GELU())


class SignalAutoencoder(nn.Module):
    """
    1D CNN, Transformer encoder and transposed-convolution decoder for 12-lead ECGs.

    The CNN shortens time by 2 x 5 x 5 = 50, so 2,500 samples become 50 tokens of 0.2 s each. A learned
    position is added and the Transformer lets every token attend to the others. The mean of the tokens is
    the ECG embedding; the decoder rebuilds the waveform from the tokens.

    Parameters
    ----------
    width : int
        Features per token and size of the embedding.
    layers : int
        Transformer layers.
    heads : int
        Attention heads.
    dropout : float
        Dropout inside the Transformer.
    """

    def __init__(self, width: int = 96, layers: int = 4, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        half = width // 2
        self.config = {"width": width, "layers": layers, "heads": heads, "dropout": dropout}
        self.cnn = nn.Sequential(
            conv_block(nn.Conv1d(LEADS, half, kernel_size=7, stride=2, padding=3), half),
            conv_block(nn.Conv1d(half, width, kernel_size=9, stride=5, padding=4), width),
            conv_block(nn.Conv1d(width, width, kernel_size=9, stride=5, padding=4), width),
        )
        self.position = nn.Parameter(torch.zeros(1, SAMPLES // 50, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width),
                                                 enable_nested_tensor=False)
        self.decoder = nn.Sequential(
            conv_block(nn.ConvTranspose1d(width, width, kernel_size=9, stride=5, padding=2), width),
            conv_block(nn.ConvTranspose1d(width, half, kernel_size=9, stride=5, padding=2), half),
            conv_block(nn.ConvTranspose1d(half, half, kernel_size=8, stride=2, padding=3), half),
            nn.Conv1d(half, LEADS, kernel_size=7, padding=3),
        )

    def encode(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Turn waveforms into contextual tokens and one embedding per ECG.

        Parameters
        ----------
        signals : torch.Tensor
            Shape ``(batch, 12, 2500)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tokens ``(batch, 50, width)`` and embeddings ``(batch, width)``.
        """
        tokens = self.transformer(self.cnn(signals).transpose(1, 2) + self.position)
        return tokens, tokens.mean(dim=1)

    def forward(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct the waveforms.

        Parameters
        ----------
        signals : torch.Tensor
            Shape ``(batch, 12, 2500)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Reconstruction ``(batch, 12, 2500)``, tokens ``(batch, 50, width)``, embeddings
            ``(batch, width)``.
        """
        tokens, embedding = self.encode(signals)
        return self.decoder(tokens.transpose(1, 2)), tokens, embedding


def epoch_order(records: int, seed: int, epoch: int) -> np.ndarray:
    """
    Shuffle the records of one epoch.

    The order depends only on the seed and the epoch, so a resumed run sees the same batches.

    Parameters
    ----------
    records : int
        ECGs in the split.
    seed : int
        Run seed.
    epoch : int
        Epoch number, from 1.

    Returns
    -------
    np.ndarray
        A permutation of ``range(records)``.
    """
    return np.random.default_rng([seed, epoch]).permutation(records)


@torch.no_grad()
def mean_squared_error(model: SignalAutoencoder, dataset: SignalDataset, batch_size: int,
                       device: torch.device) -> float:
    """
    Reconstruction error over every ECG of a dataset.

    Parameters
    ----------
    model : SignalAutoencoder
        The autoencoder.
    dataset : SignalDataset
        ECGs to reconstruct.
    batch_size : int
        ECGs per batch.
    device : torch.device
        Where the model is.

    Returns
    -------
    float
        Mean squared error per sample.
    """
    model.eval()
    total = torch.zeros((), device=device)
    for start in range(0, len(dataset), batch_size):
        signals = dataset.batch(np.arange(start, min(start + batch_size, len(dataset))), device)
        total += nn.functional.mse_loss(model(signals)[0], signals, reduction="sum")
    return float(total) / (len(dataset) * LEADS * SAMPLES)


def train_epoch(model: SignalAutoencoder, dataset: SignalDataset, order: np.ndarray, batch_size: int,
                optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                device: torch.device, label: str) -> float:
    """
    Run one pass over the training ECGs.

    Parameters
    ----------
    model : SignalAutoencoder
        The autoencoder.
    dataset : SignalDataset
        Training ECGs.
    order : np.ndarray
        Shuffled record positions; the last incomplete batch is dropped.
    batch_size : int
        ECGs per batch.
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Learning-rate schedule, stepped once per batch.
    device : torch.device
        Where the model is.
    label : str
        Prefix of the progress line.

    Returns
    -------
    float
        Mean training loss of the epoch.

    Raises
    ------
    FloatingPointError
        If the loss stops being finite.
    """
    model.train()
    steps = len(order) // batch_size
    total, started = torch.zeros((), device=device), time.perf_counter()
    for step in range(steps):
        signals = dataset.batch(order[step * batch_size:(step + 1) * batch_size], device)
        loss = nn.functional.mse_loss(model(signals)[0], signals)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.detach()
        if (step + 1) % max(1, steps // 10) == 0 or step + 1 == steps:
            mean = float(total) / (step + 1)
            if not np.isfinite(mean):
                raise FloatingPointError(f"Training loss is not finite at step {step + 1}")
            rate = (step + 1) * batch_size / (time.perf_counter() - started)
            print(f"\r{label} paso {step + 1}/{steps} mse={mean:.4f} {rate:,.0f} ECG/s", end="", flush=True)
    return float(total) / steps


def train_autoencoder(model: SignalAutoencoder, train_dataset: SignalDataset, val_dataset: SignalDataset, *,
                      epochs: int, batch_size: int, seed: int, device: torch.device,
                      checkpoint_path: Path, resume: bool = True, learning_rate: float = 1e-3,
                      eval_batch_size: int = 512) -> pd.DataFrame:
    """
    Train by reconstruction and checkpoint after every epoch.

    AdamW follows a one-cycle schedule (5% warm-up, cosine decay) over ``epochs``. Gradients are clipped
    at norm 1.

    Parameters
    ----------
    model : SignalAutoencoder
        The autoencoder, already on ``device``.
    train_dataset, val_dataset : SignalDataset
        Training and validation ECGs.
    epochs : int
        Passes over the training ECGs.
    batch_size : int
        ECGs per training batch.
    seed : int
        Seed of the batch order.
    device : torch.device
        Where the model is.
    checkpoint_path : Path
        File rewritten after every epoch with the model, optimizer, schedule and history.
    resume : bool
        Continue from ``checkpoint_path`` when it exists.
    learning_rate : float
        Peak learning rate.
    eval_batch_size : int
        ECGs per validation batch.

    Returns
    -------
    pd.DataFrame
        One row per epoch: ``epoch``, ``train_mse``, ``val_mse``, ``seconds``.

    Raises
    ------
    ValueError
        If the checkpoint was written with another configuration.
    """
    steps = len(train_dataset) // batch_size
    settings = {"epochs": epochs, "batch_size": batch_size, "seed": seed, "learning_rate": learning_rate,
                "train_records": len(train_dataset), "model": model.config}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                    total_steps=epochs * steps, pct_start=0.05)
    history: list[dict[str, float]] = []
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if saved["settings"] != settings:
            raise ValueError(f"Checkpoint settings {saved['settings']} differ from {settings}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = saved["history"]
        print(f"Reanudando después de la época {len(history)}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(len(history) + 1, epochs + 1):
        started = time.perf_counter()
        train_mse = train_epoch(model, train_dataset, epoch_order(len(train_dataset), seed, epoch),
                                batch_size, optimizer, scheduler, device, f"Época {epoch}/{epochs}")
        val_mse = mean_squared_error(model, val_dataset, eval_batch_size, device)
        seconds = time.perf_counter() - started
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "seconds": seconds})
        print(f"\rÉpoca {epoch}/{epochs}: train_mse={train_mse:.4f} val_mse={val_mse:.4f} {seconds:.0f} s"
              + " " * 20)
        partial = checkpoint_path.with_suffix(".partial")
        torch.save({"settings": settings, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history}, partial)
        os.replace(partial, checkpoint_path)
    return pd.DataFrame(history)


@torch.no_grad()
def embeddings(model: SignalAutoencoder, dataset: SignalDataset, *, batch_size: int,
               device: torch.device) -> np.ndarray:
    """
    Encode every ECG of a dataset, in dataset order, without the decoder.

    Parameters
    ----------
    model : SignalAutoencoder
        The trained autoencoder.
    dataset : SignalDataset
        ECGs to encode.
    batch_size : int
        ECGs per batch.
    device : torch.device
        Where the model is.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(len(dataset), width)``.
    """
    model.eval()
    parts = []
    for start in range(0, len(dataset), batch_size):
        signals = dataset.batch(np.arange(start, min(start + batch_size, len(dataset))), device)
        parts.append(model.encode(signals)[1].cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def cluster_embeddings(train_embeddings: np.ndarray, val_embeddings: np.ndarray, *, seed: int,
                       candidates: tuple[int, ...] = CLUSTER_CANDIDATES
                       ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    """
    Cluster the training embeddings with k-means and assign the validation embeddings; no labels are used.

    The scaler, the PCA (components for 95% of the variance) and the centers are fitted on training
    embeddings only. The number of clusters is the candidate with the highest silhouette on one fixed
    sample of up to 5,000 training embeddings.

    Parameters
    ----------
    train_embeddings, val_embeddings : np.ndarray
        Arrays of shape ``(records, width)``.
    seed : int
        Seed of k-means and of the silhouette sample.
    candidates : tuple[int, ...]
        Numbers of clusters to compare.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]
        Training and validation cluster numbers, the comparison table, and the fitted ``scaler``, ``pca``
        and ``kmeans``.
    """
    scaler = StandardScaler().fit(train_embeddings)
    pca = PCA(n_components=PCA_VARIANCE, random_state=seed).fit(scaler.transform(train_embeddings))
    train_points = pca.transform(scaler.transform(train_embeddings))
    sample = np.random.default_rng(seed).choice(len(train_points), min(SILHOUETTE_SAMPLE, len(train_points)),
                                                replace=False)
    fits, table = {}, []
    for clusters in candidates:
        fits[clusters] = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(train_points)
        labels = fits[clusters].labels_
        table.append({"clusters": clusters,
                      "silhouette": float(silhouette_score(train_points[sample], labels[sample])),
                      "inertia": float(fits[clusters].inertia_),
                      "smallest_cluster": int(np.bincount(labels, minlength=clusters).min())})
    table = pd.DataFrame(table)
    kmeans = fits[int(table.loc[table["silhouette"].idxmax(), "clusters"])]
    val_cluster = kmeans.predict(pca.transform(scaler.transform(val_embeddings)))
    return kmeans.labels_.copy(), val_cluster, table, {"scaler": scaler, "pca": pca, "kmeans": kmeans}


def clinical_profile(rows: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    """
    Describe fixed clusters with the echocardiographic flags of the release.

    Parameters
    ----------
    rows : pd.DataFrame
        Metadata rows in the order of ``clusters``.
    clusters : np.ndarray
        Cluster number of each row.

    Returns
    -------
    pd.DataFrame
        One row per cluster: ``cluster``, ``ecgs`` and, for each flag, ``<flag>_known`` (ECGs with a
        recorded value), ``<flag>_positive`` and ``<flag>_pct`` (positives among known values).

    Raises
    ------
    ValueError
        If the rows and cluster numbers differ in length.
    """
    if len(rows) != len(clusters):
        raise ValueError(f"{len(rows)} rows but {len(clusters)} cluster numbers")
    profile = []
    for cluster in np.unique(clusters):
        part = rows[clusters == cluster]
        entry: dict[str, float] = {"cluster": int(cluster), "ecgs": len(part)}
        for flag in PROFILE_FLAGS:
            values = pd.to_numeric(part[flag], errors="coerce").dropna()
            entry[f"{flag}_known"] = len(values)
            entry[f"{flag}_positive"] = int(values.sum())
            entry[f"{flag}_pct"] = 100 * float(values.mean()) if len(values) else float("nan")
        profile.append(entry)
    return pd.DataFrame(profile)


def save_artifacts(output_dir: Path, model: SignalAutoencoder, history: pd.DataFrame,
                   train_embeddings: np.ndarray, val_embeddings: np.ndarray, train_cluster: np.ndarray,
                   val_cluster: np.ndarray, cluster_fit: dict[str, object]) -> None:
    """
    Save weights, history, embeddings and the cluster transformation; no patient keys or labels.

    Arrays keep the row order of each split, so row ``i`` belongs to ECG ``i`` of that split.

    Parameters
    ----------
    output_dir : Path
        Folder to write.
    model : SignalAutoencoder
        The trained autoencoder.
    history : pd.DataFrame
        Output of ``train_autoencoder``.
    train_embeddings, val_embeddings : np.ndarray
        Outputs of ``embeddings``.
    train_cluster, val_cluster : np.ndarray
        Cluster numbers from ``cluster_embeddings``.
    cluster_fit : dict[str, object]
        Fitted ``scaler``, ``pca`` and ``kmeans`` from ``cluster_embeddings``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "model": model.state_dict()}, output_dir / "autoencoder.pt")
    history.to_csv(output_dir / "history.csv", index=False)
    np.save(output_dir / "train_embeddings.npy", train_embeddings)
    np.save(output_dir / "val_embeddings.npy", val_embeddings)
    np.save(output_dir / "train_clusters.npy", train_cluster)
    np.save(output_dir / "val_clusters.npy", val_cluster)
    scaler, pca, kmeans = cluster_fit["scaler"], cluster_fit["pca"], cluster_fit["kmeans"]
    np.savez(output_dir / "cluster_transform.npz", scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
             pca_mean=pca.mean_, pca_components=pca.components_, kmeans_centers=kmeans.cluster_centers_)
