"""Generate the local interactive explanation from existing real ECG experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from ecg_experiment.localization_visual import build_visual_explanation


def main() -> None:
    """Render the explanation without training or evaluating a new model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("notebooks/localization-explained.executed.html"))
    args = parser.parse_args()
    print(build_visual_explanation(args.root, args.output))


if __name__ == "__main__":
    main()
