"""Independently verify a failed v1 quality resource admission from metadata only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

RELATIVE = "outputs/ssl_synthetic_localization/generated_quality_v1"


def require(value: bool, label: str) -> None:
    """Raise on a metadata inconsistency.

    Parameters
    ----------
    value : bool
        Required condition.
    label : str
        Failure message.
    """
    if not value:
        raise ValueError(label)


def digest(path: Path) -> str:
    """Hash a permitted small text file.

    Parameters
    ----------
    path : Path
        Metadata or authored source path.

    Returns
    -------
    str
        SHA-256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forecasts(profile: dict[str, Any]) -> dict[str, bool]:
    """Recompute the four prospective admission decisions independently.

    Parameters
    ----------
    profile : dict
        Executed metadata.

    Returns
    -------
    dict[str, bool]
        Wall, CPU, memory and output admission checks.
    """
    for kind in ("cpu", "wall"):
        value = profile[f"spent_{kind}_seconds"] + 260 * profile[f"{kind}_seconds"] + 30
        require(abs(value - profile[f"forecast_{kind}_seconds"]) <= 1e-12, "profile forecast arithmetic")
    return {
        "wall_admitted": profile["forecast_wall_seconds"] < 1200,
        "cpu_admitted": profile["forecast_cpu_seconds"] < 1800,
        "memory_admitted": profile["peak_allocated_bytes"] < 2 * 1024**3,
        "output_admitted": profile["forecast_saved_bytes"] < 512 * 1024**2,
    }


def run(root: Path, source_root: Path, failure_sha: str, script_sha: str) -> dict[str, Any]:
    """Execute one bounded copied-script metadata check.

    Parameters
    ----------
    root, source_root : Path
        Output host and committed source tree.
    failure_sha, script_sha : str
        Admitted immutable bindings.

    Returns
    -------
    dict[str, Any]
        Correct-stop metadata evidence.
    """
    clock = time.process_time(), time.perf_counter()
    study = root / RELATIVE
    destination = study / "independent_failure_audit_v1"
    require(Path(__file__).absolute() == destination / "audit_script.py", "exact copied checker location")
    require(
        not (destination / "summary.json").exists() and not (destination / "failure.json").exists(),
        "audit previously attempted",
    )
    receipt = "summary.json"
    main_cpu: float | None = None
    main_gpu: float | None = None
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA hidden")
        require(digest(Path(__file__)) == script_sha, "admitted checker SHA")
        require(digest(study / "failure.json") == failure_sha, "admitted failed-stage SHA")
        failure = json.loads((study / "failure.json").read_text())
        profile = json.loads((study / "profile.json").read_text())
        ledger = json.loads((study / "prospective_ledger.json").read_text())
        require(
            ledger["receipt"] == "failure.json" and ledger["receipt_sha256"] == failure_sha,
            "ledger failure binding",
        )
        main_cpu, main_gpu = ledger["cpu_seconds"], ledger["gpu_wall_seconds"]
        require(main_cpu >= failure["cpu_seconds"] > 0 and main_cpu < 1800, "terminal main CPU")
        require(1200 > ledger["wall_seconds"] >= failure["wall_seconds"] > 0, "terminal main wall")
        require(main_gpu == failure["gpu_lock_wall_seconds"] and 0 < main_gpu < 1200, "charged GPU lock")
        gate = forecasts(profile)
        require(
            gate
            == {
                "wall_admitted": False,
                "cpu_admitted": True,
                "memory_admitted": True,
                "output_admitted": True,
            },
            "wall-only admission failure",
        )
        expected = {"cpu_predecessor_replay": 4, "gpu_profile": 1, "gpu_generation": 0, "gpu_replay": 0}
        require(failure["forward_counts"] == failure["forward_attempts"] == expected, "exact forward counts")
        require(failure["completed_seeds"] == [], "no completed bank seeds")
        require(
            failure["error"] == "Actual-path profile fails the frozen conservative resource forecasts",
            "expected resource stop",
        )
        for name in (
            "summary.json",
            "raw.npy",
            "variants.npy",
            "strict_support.npy",
            "support.npy",
            "replay_raw.npy",
            "cases.jsonl",
        ):
            require(not (study / name).exists(), "no unadmitted bank artifacts")
        closure = json.loads((study / "predecessor_closure.json").read_text())
        require(
            closure["all_exact"] is True and closure["cpu_predecessor_forwards"] == 4,
            "source-bound predecessor closure",
        )
        for name, expected_sha in failure["source_sha256"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "relative source")
            require(digest(source_root / name) == expected_sha, "committed source binding")
            require(digest(study / "source_snapshot" / name) == expected_sha, "source snapshot")
        cpu, wall = time.process_time() - clock[0], time.perf_counter() - clock[1]
        require(cpu < 58 and wall < 58, "metadata audit reserve")
        result = {
            "all_passed": True,
            "correct_resource_stop": True,
            "admission": gate,
            "profile": profile,
            "forward_counts": expected,
            "bank_generated": False,
            "main_cpu_seconds": main_cpu,
            "main_gpu_wall_seconds": main_gpu,
            "input_payloads_source_bound": True,
            "independent_model_forwards": 0,
            "payload_reads": 0,
            "gpu_used": False,
            "failure_sha256": failure_sha,
            "script_sha256": script_sha,
            "cpu_seconds": cpu,
            "wall_seconds": wall,
        }
        with (destination / receipt).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except Exception as error:
        receipt = "failure.json"
        with (destination / receipt).open("x") as stream:
            json.dump(
                {
                    "all_passed": False,
                    "error": str(error),
                    "failure_sha256": failure_sha,
                    "script_sha256": script_sha,
                },
                stream,
                indent=2,
            )
            stream.write("\n")
        raise
    finally:
        receipt_sha = digest(destination / receipt)
        cpu, wall = time.process_time() - clock[0], time.perf_counter() - clock[1]
        with (destination / "prospective_ledger.json").open("x") as stream:
            json.dump(
                {
                    "receipt": receipt,
                    "sha256": receipt_sha,
                    "cpu_seconds": cpu,
                    "wall_seconds": wall,
                    "gpu_wall_seconds": 0.0,
                    "main_cpu_seconds": main_cpu,
                    "stage_main_plus_audit_cpu_seconds": main_cpu + cpu if main_cpu is not None else None,
                    "stage_gpu_wall_seconds": main_gpu,
                    "historical_costs_verified": False,
                    "audit_budget_passed": cpu < 60 and wall < 60,
                    "measurement_endpoint": "after audit receipt write/hash, before ledger serialization",
                },
                stream,
                indent=2,
            )
            stream.write("\n")


def main() -> None:
    """Parse the explicit copied-checker metadata audit invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--failure-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    args = parser.parse_args()
    print(
        json.dumps(run(args.root.absolute(), args.source_root.absolute(), args.failure_sha, args.script_sha))
    )


if __name__ == "__main__":
    main()
