"""Evaluate frozen features with three separate 10%-label training samples."""

import subprocess
import sys
from pathlib import Path


MODELS = {
    "hubert-small": "data/processed/pretrained/hubert-small-union",
    "ecg-fm": "data/processed/pretrained/ecg-fm",
    "ecg-jepa": "data/processed/ptbxl/features_jepa_multiblock_union_seeds42_43_44",
}


def main():
    output = Path("outputs/experiment001")
    for model, feature_dir in MODELS.items():
        feature_dir = Path(feature_dir)
        if not (feature_dir / "metadata.json").exists():
            raise FileNotFoundError(f"Complete feature extraction first: {feature_dir}")
        for seed in (42, 43, 44):
            if (output / f"{model}_linear_seed{seed}" / "metrics.json").exists():
                continue
            command = [sys.executable, "-u", "-m", "scripts.experiments.probe_pretrained",
                       "--embeddings-dir", str(feature_dir), "--name", model,
                       "--manifest-dir", f"data/processed/ptbxl/seed{seed}_fraction0.1",
                       "--seed", str(seed), "--output-dir", str(output)]
            with (output / f"{model}_linear_seed{seed}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            print(f"Completed frozen {model}, label seed {seed}", flush=True)


if __name__ == "__main__":
    main()
