#!/usr/bin/env python3
"""Sequential frozen released ECG-CPC extraction and the existing label probes."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from ecg_experiment.files import write_json_atomic


ROOT = Path(__file__).resolve().parents[2]


def main():
    output = ROOT / "outputs/experiment004_cpc_40k"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "released_status.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MPLCONFIGDIR="/tmp/ecg-cpc-mpl", KEOPS_CACHE_FOLDER="/tmp/ecg-cpc-keops",
               CUDA_PATH="/usr/local/cuda")
    with (output / "released_runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json_atomic(state_path, {"stage": "extraction", "pid": os.getpid()})
            subprocess.run([sys.executable, "-m", "scripts.features.extract_ecg_cpc"], cwd=ROOT, env=env, check=True)
            features = output / "released_features"
            provenance = json.loads((features / "metadata.json").read_text())
            for fraction, name in (("1", "released_full"), ("0.1", "released_10pct")):
                directory = output / name / "ecg-cpc_released_linear_seed42"
                artifacts = ("metrics.json", "test_predictions.csv", "calibration_predictions.npz",
                             "linear_model.npz", "config.json")
                if directory.exists() and not all((directory / x).is_file() for x in artifacts):
                    suffix = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    directory.rename(directory.with_name(directory.name + "_interrupted_" + suffix))
                if not directory.exists():
                    write_json_atomic(state_path, {"stage": "linear_probe", "label_fraction": fraction, "pid": os.getpid()})
                    subprocess.run([sys.executable, "-m", "scripts.experiments.probe_pretrained", "--embeddings-dir", str(features),
                                    "--name", "ecg-cpc_released", "--manifest-dir",
                                    str(ROOT / "data/processed/ptbxl" / f"seed42_fraction{fraction}"),
                                    "--output-dir", str(output / name)], cwd=ROOT, env=env, check=True)
                config_path = directory / "config.json"
                config = json.loads(config_path.read_text())
                config["pretraining_exposure"] = provenance["pretraining_overlap"]
                config["released_model"] = {key: provenance[key] for key in (
                    "checkpoint_source", "license", "reported_pretraining", "comparison_limit", "cauchy_backend")}
                config["released_model"]["checkpoint_sha256"] = provenance["identity"]["checkpoint_sha256"]
                write_json_atomic(config_path, config)
            write_json_atomic(state_path, {"stage": "complete", "pid": os.getpid(), "label_fractions": ["1", "0.1"]})
        except Exception as exc:
            write_json_atomic(state_path, {"stage": "failed", "pid": os.getpid(), "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
