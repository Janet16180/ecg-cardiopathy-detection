"""Run the prespecified compact-model comparison for three label-sampling seeds.

Relative paths are resolved against the repository root, where every stage runs.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from ecg_experiment.processes import run_logged

ROOT = Path(__file__).resolve().parents[2]
MODELS = (("cnn", "cnn_supervised"), ("transformer", "transformer_supervised"),
          ("mae", "mae_finetuned"), ("jepa", "jepa_finetuned"))
SSL_MODELS = {"mae", "jepa"}
SHARED_SSL_SEED = 42


def model_command(args: argparse.Namespace, model: str, seed: int, manifest: str) -> list[str]:
    """
    Build the training command of one compact model and label seed.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    model : str
        Model family.
    seed : int
        Label-sampling seed.
    manifest : str
        Manifest directory relative to the repository root.

    Returns
    -------
    list[str]
        Command to run.

    Raises
    ------
    FileNotFoundError
        If a later SSL seed needs the shared seed-42 checkpoint that is absent.
    """
    command = [sys.executable, "-u", "-m", "ecg_experiment.run", "--model", model,
               "--manifest-dir", manifest, "--output-dir", str(args.output_dir),
               "--seed", str(seed), "--device", args.device]
    if model in SSL_MODELS and seed != SHARED_SSL_SEED:
        checkpoint = args.output_dir / f"{model}_ssl" / "encoder.pt"
        if not (ROOT / checkpoint).exists():
            raise FileNotFoundError("Run seed 42 first to create the shared SSL checkpoint")
        command.extend(["--ssl-checkpoint", str(checkpoint)])
    return command


def main() -> None:
    """Prepare missing label samples and train every missing model and seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment001"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        manifest = f"data/processed/ptbxl/seed{seed}_fraction0.1"
        if not (ROOT / manifest).exists():
            subprocess.run([sys.executable, "-m", "scripts.data.prepare_ptbxl", "--seed", str(seed)],
                           check=True, cwd=ROOT)
        for model, name in MODELS:
            if (output / f"{name}_seed{seed}" / "metrics.json").exists():
                print(f"Already completed {name}, seed {seed}", flush=True)
                continue
            command = model_command(args, model, seed, manifest)
            print("Running " + " ".join(command), flush=True)
            run_logged(command, output / f"{name}_seed{seed}.log", ROOT)
            print(f"Completed {name}, seed {seed}", flush=True)


if __name__ == "__main__":
    main()
