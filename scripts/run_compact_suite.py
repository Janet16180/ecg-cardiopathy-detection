"""Run the prespecified compact-model comparison for three label-sampling seeds."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment001"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        manifest = Path(f"data/processed/ptbxl/seed{seed}_fraction0.1")
        if not manifest.exists():
            subprocess.run([sys.executable, "scripts/prepare_ptbxl.py", "--seed", str(seed)], check=True)
        for model, name in [("cnn", "cnn_supervised"), ("transformer", "transformer_supervised"),
                            ("mae", "mae_finetuned"), ("jepa", "jepa_finetuned")]:
            if (args.output_dir / f"{name}_seed{seed}" / "metrics.json").exists():
                print(f"Already completed {name}, seed {seed}", flush=True)
                continue
            command = [sys.executable, "-u", "-m", "ecg_experiment.run", "--model", model,
                       "--manifest-dir", str(manifest), "--output-dir", str(args.output_dir),
                       "--seed", str(seed), "--device", args.device]
            if model in {"mae", "jepa"} and seed != 42:
                checkpoint = args.output_dir / f"{model}_ssl" / "encoder.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError("Run seed 42 first to create the shared SSL checkpoint")
                command.extend(["--ssl-checkpoint", str(checkpoint)])
            print("Running " + " ".join(command), flush=True)
            with (args.output_dir / f"{name}_seed{seed}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            print(f"Completed {name}, seed {seed}", flush=True)


if __name__ == "__main__":
    main()
