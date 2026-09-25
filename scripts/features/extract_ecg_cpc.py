#!/usr/bin/env python3
"""Extract frozen features from the released HEEDB S4 ECG-CPC checkpoint.

Run in .venv-pretrained. Uses the author's encoder and S4 implementation,
strictly loads all 76 backbone tensors, and drops only the SSL prediction head.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.provenance import git_head
from ecg_experiment.waveforms import manifest_rows, read_record

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "bc253edc6ac279ce2caec86d3c3a21b740e43735ca03be1dc7c50e630b0ad4cf"
CHECKPOINT = ROOT / "third_party/checkpoints/ecg-cpc/ECG-CPC Checkpoint/last_11597276.ckpt"
REPOSITORY = ROOT / "third_party/ecg-fm-benchmarking"
EXPECTED_PREFIXES = ("ts_encoder.encoder.", "ts_encoder.predictor.")
SSL_HEAD_TENSORS = {"ts_encoder.head_ssl.proj.0.weight"}
OMEGA_LENGTH = 601
KERNEL_LENGTH_TOKENS = 1200
FEATURE_DIMENSION = 512
CROPS_PER_RECORD = 4
CROP_SAMPLES = 1250
TOKENS_PER_CROP = 300
VIEW_SHAPE = (12, 600)
PRINT_EVERY_BATCHES = 20
OUTPUT_FILES = ("features.npy", "ecg_ids.npy")

# Placeholder classes for the training framework's config metadata, one per name.
_CONFIG_TYPES: dict[tuple[str, str], type] = {}


class MetadataUnpickler(pickle.Unpickler):
    """
    Avoid importing the training framework for unused config type metadata.

    Tensor unpickling is unchanged. This is NOT a security sandbox: the exact
    verified author checkpoint is required before this loader is used.
    """

    def find_class(self, module: str, name: str) -> Any:
        """
        Resolve a pickled global, replacing framework config classes by placeholders.

        Parameters
        ----------
        module : str
            Module of the pickled global.
        name : str
            Name of the pickled global.

        Returns
        -------
        Any
            The resolved global or a placeholder class.

        Raises
        ------
        ValueError
            If a framework object other than a config class is pickled.
        """
        if not module.startswith("clinical_ts."):
            return super().find_class(module, name)
        if "Config" not in name:
            raise ValueError(f"Unexpected checkpoint object: {module}.{name}")
        return _CONFIG_TYPES.setdefault((module, name), type(name, (), {}))


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """
    Load the verified released checkpoint.

    Parameters
    ----------
    path : Path
        Released Lightning checkpoint.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, int]]
        State dict and the training epoch and global step.

    Raises
    ------
    ValueError
        If the checkpoint differs from the verified release.
    """
    if sha256_file(path) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA256 differs from verified Figshare release")
    payload = torch.load(path, map_location="cpu", weights_only=False,
                         pickle_module=types.SimpleNamespace(
                             __name__="ecg_cpc_metadata_pickle", Unpickler=MetadataUnpickler))
    return payload["state_dict"], {"epoch": payload["epoch"], "global_step": payload["global_step"]}


def load_exact(module: nn.Module, state: dict[str, torch.Tensor], prefix: str) -> int:
    """
    Strictly load the tensors under a prefix into a module.

    Parameters
    ----------
    module : nn.Module
        Module to load.
    state : dict[str, torch.Tensor]
        Checkpoint state dict.
    prefix : str
        Key prefix of the module's tensors.

    Returns
    -------
    int
        Number of loaded tensors.

    Raises
    ------
    RuntimeError
        If tensors are missing, unexpected, or of another shape.
    """
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    # Some official S4 initial tensors are expanded views. Assignment preserves
    # exact checkpoint tensors without copying into overlapping storage.
    module.load_state_dict(selected, strict=True, assign=True)
    return len(selected)


def torch_cauchy_conj(v: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Exact conjugate-pair Cauchy sum, using native complex PyTorch operations.

    Replaces only the optional compiled reduction backend, not S4 mathematics.
    State dimension eight makes this small enough for ordinary broadcasting.

    Parameters
    ----------
    v : torch.Tensor
        Complex numerators.
    z : torch.Tensor
        Complex evaluation nodes.
    w : torch.Tensor
        Complex poles, one per conjugate pair.

    Returns
    -------
    torch.Tensor
        Sum over poles of ``v / (z - w) + conj(v) / (z - conj(w))``.
    """
    dims = max(v.ndim, z.ndim, w.ndim)
    v, z, w = (x.reshape((1,) * (dims - x.ndim) + tuple(x.shape)) for x in (v, z, w))
    z, v, w = z.unsqueeze(-2), v.unsqueeze(-1), w.unsqueeze(-1)
    return (v / (z - w) + v.conj() / (z - w.conj())).sum(dim=-2)


def add_repository_path(repository: Path) -> None:
    """
    Make the author's ``clinical_ts`` package importable.

    Parameters
    ----------
    repository : Path
        Checkout of the benchmarking repository.
    """
    code = str(Path(repository) / "code")
    if code not in sys.path:
        sys.path.insert(0, code)


def patch_s4_backend() -> None:
    """
    Route the official S4 Cauchy reduction through ``torch_cauchy_conj``.

    The machine lacks Python development headers required by PyKeOps's
    binder, so the mathematically identical PyTorch sum replaces it.
    """
    from clinical_ts.ts.s4_modules import s42

    s42.has_cauchy_extension = False
    s42.cauchy_conj = torch_cauchy_conj


class ReleasedCPC(nn.Module):
    """
    Released ECG-CPC S4 encoder and predictor without the SSL head, frozen.

    Parameters
    ----------
    repository : Path
        Checkout of the benchmarking repository.
    checkpoint : Path
        Verified released checkpoint.

    Raises
    ------
    ValueError
        If the checkpoint holds unexpected tensors or S4 buffer lengths.
    """

    def __init__(self, repository: Path, checkpoint: Path) -> None:
        super().__init__()
        add_repository_path(repository)
        from clinical_ts.template_modules import ShapeConfig, StaticStats
        from clinical_ts.ts.encoder import RNNEncoder, RNNEncoderConfig
        from clinical_ts.ts.s4 import S4Predictor, S4PredictorConfig

        # Patch after the official modules import, in the released loader's order.
        patch_s4_backend()

        state, training = checkpoint_state(checkpoint)
        omitted = set(state) - {key for key in state if key.startswith(EXPECTED_PREFIXES)}
        if omitted != SSL_HEAD_TENSORS:
            raise ValueError(f"Unexpected non-backbone checkpoint tensors: {omitted}")
        # Saved C parameters include a finite-length correction. Keep the saved
        # 1200-token S4 kernel length rather than reinterpreting C at length 300.
        omega_lengths = {value.shape[0] for key, value in state.items() if key.endswith(".omega")}
        if omega_lengths != {OMEGA_LENGTH}:
            raise ValueError(f"Unexpected S4 Fourier buffer lengths: {omega_lengths}")
        shape = ShapeConfig(channels=12, length=2400, sequence_last=True)
        self.encoder = RNNEncoder(RNNEncoderConfig(features=[FEATURE_DIMENSION] * 4, kss=[3, 1, 1, 1],
                                                   strides=[2, 1, 1, 1]), shape, StaticStats())
        self.predictor = S4Predictor(S4PredictorConfig(causal=True, state_dim=8, model_dim=FEATURE_DIMENSION,
                                                       backbone="s42"), self.encoder.get_output_shape())
        loaded = load_exact(self.encoder, state, EXPECTED_PREFIXES[0])
        loaded += load_exact(self.predictor, state, EXPECTED_PREFIXES[1])
        self.loading_info = {"loaded_backbone_tensors": loaded, "omitted_ssl_head_tensors": sorted(omitted),
                             "kernel_length_tokens": KERNEL_LENGTH_TOKENS, "checkpoint_training": training,
                             "cauchy_backend": "native PyTorch exact conjugate-pair Cauchy sum"}
        self.eval()
        self.requires_grad_(False)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """
        Mean-pool the S4 tokens of each 2.5-second view.

        Parameters
        ----------
        views : torch.Tensor
            ``[batch, 12, 600]`` views at 240 Hz.

        Returns
        -------
        torch.Tensor
            ``[batch, 512]`` features.

        Raises
        ------
        ValueError
            If the input or output has an unexpected shape or nonfinite values.
        """
        if views.ndim != 3 or views.shape[1:] != VIEW_SHAPE:
            raise ValueError("Expected 2.5-second views: [batch,12,600] at 240 Hz")
        tokens = self.predictor(**self.encoder(seq=views))["seq"]
        expected = (len(views), TOKENS_PER_CROP, FEATURE_DIMENSION)
        if tokens.shape != expected or not torch.isfinite(tokens).all():
            raise ValueError(f"Invalid ECG-CPC output {tuple(tokens.shape)}")
        return tokens.mean(dim=1)


def preprocess(signal: np.ndarray) -> np.ndarray:
    """
    Official no-normalization mV input and resampy default resampling.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, 5000]`` 500 Hz physical-mV waveform.

    Returns
    -------
    np.ndarray
        ``[4, 12, 600]`` float32 crops at 240 Hz.

    Raises
    ------
    ValueError
        If the waveform is malformed or nonfinite.
    """
    import resampy

    if signal.shape != (12, 5000) or not np.isfinite(signal).all():
        raise ValueError("Expected finite [12,5000] 500-Hz physical mV waveform")
    # The benchmark crops before its Resample transform. Match that ordering.
    views = [resampy.resample(signal[:, start:start + CROP_SAMPLES], 500, 240, axis=-1)
             for start in range(0, 5000, CROP_SAMPLES)]
    return np.stack(views).astype(np.float32)


def extraction_identity(args: argparse.Namespace, rows: list[dict[str, str]],
                        manifests: dict[str, str]) -> dict[str, Any]:
    """
    Describe every input that determines the extracted features.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    rows : list[dict[str, str]]
        Records in output order.
    manifests : dict[str, str]
        Manifest digests.

    Returns
    -------
    dict[str, Any]
        Identity that partial and completed outputs must match.
    """
    return {"checkpoint_sha256": sha256_file(args.checkpoint), "repository_commit": git_head(args.repository),
            "manifests": manifests, "ecg_ids": [row["ecg_id"] for row in rows],
            "extractor_sha256": sha256_file(Path(__file__)),
            "preprocessing": "four 2.5-second 500Hz mV crops, resampy 500->240Hz, no normalization",
            "pooling": "mean over 300 S4 tokens per crop, then mean across four crops"}


def verified_metadata(output: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    """
    Return the metadata of a completed extraction after verifying it.

    Parameters
    ----------
    output : Path
        Output directory.
    identity : dict[str, Any]
        Identity of the requested extraction.

    Returns
    -------
    dict[str, Any] | None
        Completed metadata, or None when no extraction has completed.

    Raises
    ------
    ValueError
        If the completed extraction used other inputs or an output changed.
    """
    if not (output / "metadata.json").exists():
        return None
    old = json.loads((output / "metadata.json").read_text())
    if old["identity"] != identity:
        raise ValueError("Completed feature extraction has different inputs")
    for name, digest in old["output_sha256"].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"Completed feature checksum mismatch: {name}")
    return old


def open_partial_features(output: Path, identity: dict[str, Any], records: int) -> tuple[np.memmap, int]:
    """
    Open the partial feature array, resuming after the last completed batch.

    Parameters
    ----------
    output : Path
        Output directory.
    identity : dict[str, Any]
        Identity of the requested extraction.
    records : int
        Number of records to extract.

    Returns
    -------
    tuple[np.memmap, int]
        Writable ``[records, 512]`` array and the number of completed records.

    Raises
    ------
    ValueError
        If partial outputs belong to other inputs or are malformed.
    """
    progress_path = output / "progress.json"
    partial_path = output / "features.partial.npy"
    if not progress_path.exists():
        if partial_path.exists() or (output / "features.npy").exists():
            raise ValueError("Feature array exists without progress metadata")
        features = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32,
                                             shape=(records, FEATURE_DIMENSION))
        write_json_atomic(progress_path, {"identity": identity, "completed": 0})
        return features, 0
    progress = json.loads(progress_path.read_text())
    if progress["identity"] != identity:
        raise ValueError("Partial feature extraction has different inputs")
    done = progress["completed"]
    # Recover interruption after the last atomic rename.
    if not partial_path.exists() and (output / "features.npy").exists() and done == records:
        os.replace(output / "features.npy", partial_path)
    features = np.lib.format.open_memmap(partial_path, mode="r+")
    if features.shape != (records, FEATURE_DIMENSION) or features.dtype != np.float32:
        raise ValueError("Invalid partial features")
    return features, done


def embed_records(model: ReleasedCPC, features: np.memmap, rows: list[dict[str, str]], done: int,
                  args: argparse.Namespace, identity: dict[str, Any]) -> None:
    """
    Embed the remaining records batch by batch, recording progress after each.

    Parameters
    ----------
    model : ReleasedCPC
        Frozen encoder.
    features : np.memmap
        Partial feature array.
    rows : list[dict[str, str]]
        Records in output order.
    done : int
        Records already embedded.
    args : argparse.Namespace
        Parsed command-line arguments.
    identity : dict[str, Any]
        Identity written with each progress update.
    """
    progress_path = args.output_dir / "progress.json"
    started = time.monotonic()
    for start in range(done, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        views = np.concatenate([preprocess(read_record(args.raw_dir, row["filename_hr"])) for row in batch])
        with torch.inference_mode():
            embedded = model(torch.from_numpy(views).to(args.device))
            embedded = embedded.reshape(len(batch), CROPS_PER_RECORD, FEATURE_DIMENSION).mean(dim=1)
        features[start:start + len(batch)] = embedded.cpu().numpy()
        features.flush()
        write_json_atomic(progress_path, {"identity": identity, "completed": start + len(batch)})
        first, periodic = start == done, (start // args.batch_size) % PRINT_EVERY_BATCHES == 0
        if first or periodic or start + len(batch) == len(rows):
            print(json.dumps({"completed": start + len(batch), "records": len(rows),
                              "elapsed_seconds_this_invocation": time.monotonic() - started}), flush=True)


def extract(args: argparse.Namespace) -> dict[str, Any]:
    """
    Extract, or reuse verified, frozen ECG-CPC features for the manifests.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict[str, Any]
        Metadata of the completed extraction.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ecg-cpc-mpl")
    os.environ.setdefault("KEOPS_CACHE_FOLDER", "/tmp/ecg-cpc-keops")
    torch.set_num_threads(args.threads)
    rows, manifests = manifest_rows(args.manifest_dir, args.limit)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    identity = extraction_identity(args, rows, manifests)
    completed = verified_metadata(output, identity)
    if completed is not None:
        return completed
    model = ReleasedCPC(args.repository, args.checkpoint).to(args.device)
    features, done = open_partial_features(output, identity, len(rows))
    embed_records(model, features, rows, done, args, identity)
    del features
    os.replace(output / "features.partial.npy", output / "features.npy")
    np.save(output / "ecg_ids.npy", np.asarray([int(row["ecg_id"]) for row in rows]))
    metadata = {"identity": identity, "model": "Released ECG-CPC S4, frozen mean-pool linear-probe features",
                "checkpoint_source": "https://figshare.com/articles/dataset/ECG-CPC_Checkpoint/30192604",
                "license": "CC BY 4.0", "reported_pretraining": "HEEDB, 10.7 million ECGs",
                "pretraining_overlap": "No known PTB-XL source in the reported HEEDB corpus; "
                                       "cross-source identity overlap is not audited here",
                "comparison_limit": "Project linear probe; not a reproduction of the publication's full "
                                    "benchmark or query-attention frozen evaluation",
                "parameters": sum(p.numel() for p in model.parameters()), **model.loading_info,
                "records": len(rows), "device": args.device,
                "output_sha256": {name: sha256_file(output / name) for name in OUTPUT_FILES}}
    write_json_atomic(output / "metadata.json", metadata)
    (output / "progress.json").unlink()
    return metadata


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
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/processed/ptbxl/seed42_fraction1")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/ptb-xl/1.0.3")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/experiment004_cpc_40k/released_features")
    parser.add_argument("--batch-size", type=int, default=16, help="Records, each producing four views")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch, threads, and optional record limit must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments and extract features while holding the shared GPU lock.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    with gpu_lock(args.device, blocking=True):
        result = extract(args)
    summary = {key: value for key, value in result.items() if key != "identity"}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
