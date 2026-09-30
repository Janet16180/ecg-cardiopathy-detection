"""Measure sustained runtime in separate outputs without starting the original study."""

from __future__ import annotations

import json

from ecg_experiment.simdino_runtime_probe040 import run


def main() -> None:
    """Run only the fixed prospective runtime diagnostic."""
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
