#!/usr/bin/env python3
"""Extract frozen features from the released HEEDB S4 ECG-CPC checkpoint.

Run in .venv-pretrained. Uses the author's encoder and S4 implementation,
strictly loads all 76 backbone tensors, and drops only the SSL prediction head.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.provenance import git_head
from ecg_experiment.waveforms import manifest_rows, read_record

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "bc253edc6ac279ce2caec86d3c3a21b740e43735ca03be1dc7c50e630b0ad4cf"
CHECKPOINT = ROOT / "third_party/checkpoints/ecg-cpc/ECG-CPC Checkpoint/last_11597276.ckpt"
REPOSITORY = ROOT / "third_party/ecg-fm-benchmarking"


class MetadataUnpickler(pickle.Unpickler):
    """Avoid importing the training framework for unused config type metadata.

    Tensor unpickling is unchanged. This is NOT a security sandbox: the exact
    verified author checkpoint is required before this loader is used.
    """
    config_types = {}

    def find_class(self, module, name):
        if module.startswith("clinical_ts."):
            if "Config" not in name:
                raise ValueError(f"Unexpected checkpoint object: {module}.{name}")
            return self.config_types.setdefault((module, name), type(name, (), {}))
        return super().find_class(module, name)


def checkpoint_state(path):
    if sha256_file(path) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA256 differs from verified Figshare release")
    payload = torch.load(path, map_location="cpu", weights_only=False,
                         pickle_module=types.SimpleNamespace(
                             __name__="ecg_cpc_metadata_pickle", Unpickler=MetadataUnpickler))
    return payload["state_dict"], {"epoch": payload["epoch"], "global_step": payload["global_step"]}


def load_exact(module, state, prefix):
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    # Some official S4 initial tensors are expanded views. Assignment preserves
    # exact checkpoint tensors without copying into overlapping storage.
    module.load_state_dict(selected, strict=True, assign=True)
    return len(selected)


def torch_cauchy_conj(v, z, w):
    """Exact conjugate-pair Cauchy sum, using native complex PyTorch operations.

    Replaces only the optional compiled reduction backend, not S4 mathematics.
    State dimension eight makes this small enough for ordinary broadcasting.
    """
    dims = max(v.ndim, z.ndim, w.ndim)
    v, z, w = (x.reshape((1,) * (dims - x.ndim) + tuple(x.shape)) for x in (v, z, w))
    z, v, w = z.unsqueeze(-2), v.unsqueeze(-1), w.unsqueeze(-1)
    return (v / (z - w) + v.conj() / (z - w.conj())).sum(dim=-2)


class ReleasedCPC(nn.Module):
    def __init__(self, repository, checkpoint):
        super().__init__()
        sys.path.insert(0, str(Path(repository) / "code"))
        from clinical_ts.template_modules import ShapeConfig, StaticStats
        from clinical_ts.ts.encoder import RNNEncoder, RNNEncoderConfig
        from clinical_ts.ts.s4 import S4Predictor, S4PredictorConfig
        from clinical_ts.ts.s4_modules import s42
        # The machine lacks Python development headers required by PyKeOps's
        # binder. Use its mathematically identical Cauchy sum in PyTorch.
        s42.has_cauchy_extension = False
        s42.cauchy_conj = torch_cauchy_conj

        state, training = checkpoint_state(checkpoint)
        expected_prefixes = ("ts_encoder.encoder.", "ts_encoder.predictor.")
        omitted = set(state) - {key for key in state if key.startswith(expected_prefixes)}
        if omitted != {"ts_encoder.head_ssl.proj.0.weight"}:
            raise ValueError(f"Unexpected non-backbone checkpoint tensors: {omitted}")
        # Saved C parameters include a finite-length correction. Keep the saved
        # 1200-token S4 kernel length rather than reinterpreting C at length 300.
        omega_lengths = {value.shape[0] for key, value in state.items() if key.endswith(".omega")}
        if omega_lengths != {601}:
            raise ValueError(f"Unexpected S4 Fourier buffer lengths: {omega_lengths}")
        shape = ShapeConfig(channels=12, length=2400, sequence_last=True)
        self.encoder = RNNEncoder(RNNEncoderConfig(features=[512] * 4, kss=[3, 1, 1, 1],
                                    strides=[2, 1, 1, 1]), shape, StaticStats())
        self.predictor = S4Predictor(S4PredictorConfig(causal=True, state_dim=8,
                                    model_dim=512, backbone="s42"), self.encoder.get_output_shape())
        loaded = load_exact(self.encoder, state, expected_prefixes[0])
        loaded += load_exact(self.predictor, state, expected_prefixes[1])
        self.loading_info = {"loaded_backbone_tensors": loaded, "omitted_ssl_head_tensors": sorted(omitted),
                             "kernel_length_tokens": 1200, "checkpoint_training": training,
                             "cauchy_backend": "native PyTorch exact conjugate-pair Cauchy sum"}
        self.eval()
        self.requires_grad_(False)

    def forward(self, views):
        if views.ndim != 3 or views.shape[1:] != (12, 600):
            raise ValueError("Expected 2.5-second views: [batch,12,600] at 240 Hz")
        tokens = self.predictor(**self.encoder(seq=views))["seq"]
        if tokens.shape != (len(views), 300, 512) or not torch.isfinite(tokens).all():
            raise ValueError(f"Invalid ECG-CPC output {tuple(tokens.shape)}")
        return tokens.mean(dim=1)


def preprocess(signal):
    """Official no-normalization mV input and resampy default resampling."""
    import resampy
    if signal.shape != (12, 5000) or not np.isfinite(signal).all():
        raise ValueError("Expected finite [12,5000] 500-Hz physical mV waveform")
    # The benchmark crops before its Resample transform. Match that ordering.
    views = [resampy.resample(signal[:, start:start + 1250], 500, 240, axis=-1)
             for start in range(0, 5000, 1250)]
    return np.stack(views).astype(np.float32)


def extract(args):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ecg-cpc-mpl")
    os.environ.setdefault("KEOPS_CACHE_FOLDER", "/tmp/ecg-cpc-keops")
    torch.set_num_threads(args.threads)
    rows, manifests = manifest_rows(args.manifest_dir, args.limit)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint_sha256": sha256_file(args.checkpoint), "repository_commit": git_head(args.repository),
                "manifests": manifests, "ecg_ids": [row["ecg_id"] for row in rows],
                "extractor_sha256": sha256_file(Path(__file__)), "preprocessing": "four 2.5-second 500Hz mV crops, resampy 500->240Hz, no normalization",
                "pooling": "mean over 300 S4 tokens per crop, then mean across four crops"}
    if (output / "metadata.json").exists():
        old = json.loads((output / "metadata.json").read_text())
        if old["identity"] != identity:
            raise ValueError("Completed feature extraction has different inputs")
        for name, digest in old["output_sha256"].items():
            if sha256_file(output / name) != digest:
                raise ValueError(f"Completed feature checksum mismatch: {name}")
        return old
    model = ReleasedCPC(args.repository, args.checkpoint).to(args.device)
    progress_path = output / "progress.json"
    partial_path = output / "features.partial.npy"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["identity"] != identity:
            raise ValueError("Partial feature extraction has different inputs")
        done = progress["completed"]
        # Recover interruption after the last atomic rename.
        if not partial_path.exists() and (output / "features.npy").exists() and done == len(rows):
            os.replace(output / "features.npy", partial_path)
        features = np.lib.format.open_memmap(partial_path, mode="r+")
        if features.shape != (len(rows), 512) or features.dtype != np.float32:
            raise ValueError("Invalid partial features")
    else:
        if partial_path.exists() or (output / "features.npy").exists():
            raise ValueError("Feature array exists without progress metadata")
        done = 0
        features = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=(len(rows), 512))
        write_json_atomic(progress_path, {"identity": identity, "completed": 0})
    started = time.monotonic()
    for start in range(done, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        views = np.concatenate([preprocess(read_record(args.raw_dir, row["filename_hr"])) for row in batch])
        with torch.inference_mode():
            embedded = model(torch.from_numpy(views).to(args.device)).reshape(len(batch), 4, 512).mean(dim=1)
        features[start:start + len(batch)] = embedded.cpu().numpy()
        features.flush()
        write_json_atomic(progress_path, {"identity": identity, "completed": start + len(batch)})
        if start == done or (start // args.batch_size) % 20 == 0 or start + len(batch) == len(rows):
            print(json.dumps({"completed": start + len(batch), "records": len(rows),
                              "elapsed_seconds_this_invocation": time.monotonic() - started}), flush=True)
    del features
    os.replace(partial_path, output / "features.npy")
    np.save(output / "ecg_ids.npy", np.asarray([int(row["ecg_id"]) for row in rows]))
    metadata = {"identity": identity, "model": "Released ECG-CPC S4, frozen mean-pool linear-probe features",
                "checkpoint_source": "https://figshare.com/articles/dataset/ECG-CPC_Checkpoint/30192604",
                "license": "CC BY 4.0", "reported_pretraining": "HEEDB, 10.7 million ECGs",
                "pretraining_overlap": "No known PTB-XL source in the reported HEEDB corpus; cross-source identity overlap is not audited here",
                "comparison_limit": "Project linear probe; not a reproduction of the publication's full benchmark or query-attention frozen evaluation",
                "parameters": sum(p.numel() for p in model.parameters()), **model.loading_info,
                "records": len(rows), "device": args.device,
                "output_sha256": {name: sha256_file(output / name) for name in ("features.npy", "ecg_ids.npy")}}
    write_json_atomic(output / "metadata.json", metadata)
    progress_path.unlink()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/processed/ptbxl/seed42_fraction1")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/ptb-xl/1.0.3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiment004_cpc_40k/released_features")
    parser.add_argument("--batch-size", type=int, default=16, help="Records, each producing four views")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch, threads, and optional record limit must be positive")
    with Path("/tmp/ecg_project_gpu.lock").open("a+") as lock:
        if args.device == "cuda":
            fcntl.flock(lock, fcntl.LOCK_EX)
        result = extract(args)
    print(json.dumps({key: value for key, value in result.items() if key != "identity"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
