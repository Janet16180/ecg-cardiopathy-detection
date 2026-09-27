"""Build every feature cache the notebooks read, so they open quickly.

Run from the repository root with ``.venv/bin/python -m eda.compute``. Each step
is skipped when its cache already exists; delete ``eda/outputs/features`` to
recompute. The notebooks call the same functions, so this step is optional.
"""

import time

from eda.challenge import challenge_summary, load_headers, signal_hashes
from eda.code15 import build_all_parts
from eda.mimic import load_machine_measurements, load_records, mimic_summary
from eda.ptbxl import load_metadata
from eda.ptbxl_signals import ptbxl_summary


def main() -> None:
    """Compute the caches for all sources in order of increasing cost."""
    start = time.time()
    steps = [
        ("PTB-XL signal features", lambda: ptbxl_summary(load_metadata())),
        ("Challenge headers and signal features", lambda: challenge_summary(load_headers())),
        ("Challenge signal hashes", lambda: signal_hashes(load_headers())),
        ("CODE-15 padding and signal features, all 18 parts", build_all_parts),
        ("MIMIC machine measurements", load_machine_measurements),
        ("MIMIC signal features", lambda: mimic_summary(load_records())),
    ]
    for name, step in steps:
        step()
        print(f"{name}: done after {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
