"""Execute frozen pilot admission for forward v2 and bind its added tests/runner."""

from __future__ import annotations

import json

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.ssl_synthetic_generator_pilot_qualification import qualify_pilot_domain


def main() -> None:
    """Execute all planned pairs and finalize this invocation's source-bound receipt.

    Returns
    -------
    None
        Exit nonzero with retained failed receipt if any original gate fails.
    """
    extra_sources = (
        "tests/test_ssl_synthetic_generator_v2.py",
        "scripts/validation/qualify_ssl_synthetic_generator_pilot_v2.py",
    )
    extra_hashes = {path: sha256_file(ROOT / path) for path in extra_sources}
    destination = ROOT / "outputs/ssl_synthetic_localization/generator_pilot_v2"
    receipt = qualify_pilot_domain(destination, "v2")
    receipt["source_sha256"].update(extra_hashes)
    receipt["command"] = (
        "PYTHONPATH=. .venv/bin/python -m scripts.validation.qualify_ssl_synthetic_generator_pilot_v2"
    )
    (destination / "qualification.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "all_passed": receipt["all_passed"],
                "count": receipt["case_admission"]["count"],
                "failed_cases": receipt["case_admission"]["failed_cases"],
                "wall_seconds": receipt["wall_seconds"],
            }
        )
    )
    raise SystemExit(0 if receipt["all_passed"] else 1)


if __name__ == "__main__":
    main()
