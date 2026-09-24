"""Evaluate frozen features with three separate 10%-label training samples.

Paths are relative to the repository root, where the probes run.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = "outputs/experiment001"
LABEL_SEEDS = (42, 43, 44)
MODELS = {
    "hubert-small": "data/processed/pretrained/hubert-small-union",
    "ecg-fm": "data/processed/pretrained/ecg-fm",
    "ecg-jepa": "data/processed/ptbxl/features_jepa_multiblock_union_seeds42_43_44",
}


def run_logged(command: list[str], log_path: Path) -> None:
    """
    Run a command from the repository root, writing its output to a log.

    Parameters
    ----------
    command : list[str]
        Command to run.
    log_path : Path
        Log file, overwritten.

    Raises
    ------
    subprocess.CalledProcessError
        If the command fails.
    """
    with log_path.open("w") as log:
        subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    """
    Run every missing frozen-feature probe.

    Raises
    ------
    FileNotFoundError
        If a model's feature extraction has not completed.
    """
    output = ROOT / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    for model, feature_dir in MODELS.items():
        if not (ROOT / feature_dir / "metadata.json").exists():
            raise FileNotFoundError(f"Complete feature extraction first: {feature_dir}")
        for seed in LABEL_SEEDS:
            if (output / f"{model}_linear_seed{seed}" / "metrics.json").exists():
                continue
            # Relative paths keep the recorded embedding directory identical to earlier runs.
            command = [sys.executable, "-u", "-m", "scripts.experiments.probe_pretrained",
                       "--embeddings-dir", feature_dir, "--name", model,
                       "--manifest-dir", f"data/processed/ptbxl/seed{seed}_fraction0.1",
                       "--seed", str(seed), "--output-dir", OUTPUT]
            run_logged(command, output / f"{model}_linear_seed{seed}.log")
            print(f"Completed frozen {model}, label seed {seed}", flush=True)


if __name__ == "__main__":
    main()
