"""Independently audit saved standalone Pulse2Pulse outputs without generator inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment.ssl_synthetic_pretrained_saved_audit import (
    SEEDS,
    checkpoint_digest,
    require,
    verify_case,
    verify_noise,
)

CORE_NAME = "ecg_experiment/ssl_synthetic_pretrained_saved_audit.py"
CORE_SHA256 = "650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09"
RELATIVE = "outputs/ssl_synthetic_localization/pretrained_generator_standalone_v1"


def sha256(path: Path, guard: Any) -> str:
    """Hash a permitted file while checking the audit deadline.

    Parameters
    ----------
    path : Path
        Authorized file.
    guard : callable
        Resource admission function.

    Returns
    -------
    str
        SHA-256 of the file bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            guard()
            block = stream.read(1024**2)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def local_path(root: Path, value: str) -> Path:
    """Reject absolute paths and escapes before opening a receipt binding.

    Parameters
    ----------
    root : Path
        Authorized directory.
    value : str
        Relative binding.

    Returns
    -------
    Path
        Lexically confined file path.
    """
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, "relative binding required")
    target = root / path
    require(not target.is_symlink(), "symlink input prohibited")
    return target


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object.

    Parameters
    ----------
    path : Path
        Authorized metadata file.

    Returns
    -------
    dict[str, Any]
        Decoded metadata.
    """
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "JSON object required")
    return value


def check_bindings(root: Path, study: Path, main: dict[str, Any], guard: Any) -> dict[str, str]:
    """Verify main source, public asset and saved-output catalogs.

    Parameters
    ----------
    root, study : Path
        Source repository and authorized study.
    main : dict
        Bound main receipt.
    guard : callable
        Resource guard.

    Returns
    -------
    dict[str, str]
        Freshly checked relative bindings.
    """
    bindings = {}
    for name, expected in main["source_sha256"].items():
        source = local_path(root, name)
        snapshot = local_path(study / "source_snapshot", name)
        require(sha256(source, guard) == expected, f"current source {name}")
        require(sha256(snapshot, guard) == expected, f"source snapshot {name}")
        bindings[f"source_snapshot/{name}"] = expected
    for catalog in (main["input_sha256"], main["output_sha256"]):
        for name, expected in catalog.items():
            require(sha256(local_path(study, name), guard) == expected, f"file binding {name}")
            bindings[name] = expected
    return bindings


def check_banks(study: Path, main: dict[str, Any], guard: Any) -> dict[str, bool]:
    """Regenerate inputs and independently verify every saved output and replay.

    Parameters
    ----------
    study : Path
        Authorized saved bank directory.
    main : dict
        Bound main receipt.
    guard : callable
        Resource guard.

    Returns
    -------
    dict[str, bool]
        Array-derived technical gates.
    """
    noise = np.load(study / "noise.npy", allow_pickle=False)
    verify_noise(noise)
    require(hashlib.sha256(noise.tobytes()).hexdigest() == main["input_array_sha256"], "noise digest")
    banks, checks = [], []
    for index in range(4):
        guard()
        expected = {kind: f"case_{index}_{kind}.npy" for kind in ("raw", "author_export", "canonical12")}
        require(main["case_files"][index] == expected, "case filenames")
        values = [np.load(study / expected[kind], allow_pickle=False) for kind in expected]
        checks.append(verify_case(*values))
        banks.append(values)
    require(checks == main["case_checks"], "all case check fields")
    replay_names = {kind: f"replay_{kind}.npy" for kind in ("raw", "author_export", "canonical12")}
    require(main["replay_files"] == replay_names, "replay filenames")
    replay = [np.load(study / replay_names[kind], allow_pickle=False) for kind in replay_names]
    verify_case(*replay)
    raw_bank = np.stack([values[0] for values in banks])
    require(hashlib.sha256(raw_bank.tobytes()).hexdigest() == main["raw_array_sha256"], "raw bank digest")
    return {
        "all_case_checks": all(all(row.values()) for row in checks),
        "four_fixed_inputs": main["completed_seeds"] == list(SEEDS),
        "exact_replay": all(
            a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            for a, b in zip(banks[0], replay, strict=True)
        ),
        "distinct_fixed_seed_outputs": all(
            not np.array_equal(banks[first][0], banks[second][0])
            for first in range(4)
            for second in range(first)
        ),
    }


def check_resources(study: Path, main: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Recompute saved profile, storage and additive ledger arithmetic.

    Parameters
    ----------
    study : Path
        Authorized study.
    main, config : dict
        Main receipt and frozen settings.

    Returns
    -------
    dict[str, Any]
        Verified resource quantities.
    """
    profile = read_json(study / "profile.json")
    require(profile == main["profile"], "profile binding")
    for kind in ("cpu", "wall"):
        expected = profile[f"spent_{kind}_seconds"] + 10 * profile[f"{kind}_seconds"] + 15
        require(abs(expected - profile[f"forecast_{kind}_seconds"]) < 1e-12, "profile forecast")
        require(0 < expected < 180, "profile admission")
        require(0 < main[f"{kind}_seconds"] < 180, "main time budget")
    require(main["accounting_scope"] == "prospective_standalone_only", "accounting scope")
    require(main["historical_receipts_available"] is False, "historical evidence unavailable")
    for field in ("prior_cpu_seconds", "prior_gpu_wall_seconds", "global_gpu_wall_seconds"):
        require(main[field] == 0, "zero-based standalone accounting")
    require(main["cumulative_cpu_seconds"] == main["cpu_seconds"], "main additive CPU")
    ledger = read_json(study / "prospective_ledger.json")
    require(len(ledger["entries"]) == 1, "exactly one main ledger entry")
    entry = ledger["entries"][0]
    require(entry["receipt"] == "summary.json", "ledger receipt")
    require(entry["cpu_seconds"] >= main["cpu_seconds"], "terminal CPU accounting")
    require(entry["wall_seconds"] >= main["wall_seconds"], "terminal wall accounting")
    require(entry["cpu_seconds"] < 180 and entry["wall_seconds"] < 180, "terminal main budget")
    require(
        ledger["cpu_seconds"] == entry["cpu_seconds"] and ledger["gpu_wall_seconds"] == 0, "ledger totals"
    )
    files = [path for path in study.rglob("*") if path.is_file() and "independent_audit_v1" not in path.parts]
    actual = sum(path.stat().st_size for path in files)
    before = (
        actual - (study / "summary.json").stat().st_size - (study / "prospective_ledger.json").stat().st_size
    )
    require(before == main["study_bytes_before_receipt"], "main exact bytes before receipt")
    require(actual < config["output_bytes_ceiling"] == 96 * 1024**2, "main byte cap")
    return {
        "main_cpu_seconds": entry["cpu_seconds"],
        "main_wall_seconds": entry["wall_seconds"],
        "main_bytes": actual,
        "main_summary_cpu_seconds": main["cpu_seconds"],
    }


def check_claims(main: dict[str, Any]) -> None:
    """Check source-bound runtime and claim boundaries in the saved receipt.

    Parameters
    ----------
    main : dict
        Main technical receipt.
    """
    require(main["completed_seeds"] == list(SEEDS), "fixed seeds")
    require(main["forward_attempts"] == main["completed_forwards"] == 6, "six completed forwards")
    require(
        main["forward_allocation"] == {"profile": 1, "fixed_seed_bank": 4, "replay": 1}, "forward allocation"
    )
    for field in (
        "new_training_updates",
        "real_waveform_reads",
        "external_inference_api_calls",
        "model_buffers",
    ):
        require(main[field] == 0, f"zero {field}")
    for field in (
        "gpu_used",
        "physical_units_verified",
        "clinical_claim",
        "regional_mask_control",
        "checkpoint_cohort_membership_verified",
    ):
        require(main[field] is False, f"false {field}")
    require(main["torch_threads"] == main["torch_interop_threads"] == 1, "single main torch threads")


def audit(root: Path, main_sha: str, script_sha: str) -> dict[str, Any]:
    """Run one bounded saved-output audit from its frozen copied script.

    Parameters
    ----------
    root : Path
        Repository hosting the authorized outputs.
    main_sha, script_sha : str
        Coordinator-admitted SHA-256 bindings.

    Returns
    -------
    dict[str, Any]
        Independent aggregate audit receipt.
    """
    clock = (time.process_time(), time.perf_counter())
    study = root / RELATIVE
    destination = study / "independent_audit_v1"
    require(Path(__file__).absolute() == destination / "audit_script.py", "execute exact copied checker")
    require(not (destination / "summary.json").exists(), "audit summary already exists")
    require(not (destination / "failure.json").exists(), "audit failure already exists")

    def guard() -> None:
        require(time.process_time() - clock[0] < 118, "audit CPU deadline")
        require(time.perf_counter() - clock[1] < 118, "audit wall deadline")
        require(
            sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()) < 15 * 1024**2,
            "audit byte reserve",
        )

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"audit timer {signum}")

    for timer, sig in ((signal.ITIMER_REAL, signal.SIGALRM), (signal.ITIMER_PROF, signal.SIGPROF)):
        require(signal.getitimer(timer) == (0, 0), "exclusive audit timer")
        signal.signal(sig, interrupted)
        signal.setitimer(timer, 118)
    try:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(name) == "1", "single numerical thread")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA hidden")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(sha256(Path(__file__), guard) == script_sha, "admitted checker SHA")
        require(sha256(root / CORE_NAME, guard) == CORE_SHA256, "frozen independent arithmetic core")
        require(sha256(study / "summary.json", guard) == main_sha, "admitted main SHA")
        main = read_json(study / "summary.json")
        config = read_json(
            study / "source_snapshot/configs/ssl_synthetic_pretrained_generator_standalone_v1.json"
        )
        bindings = check_bindings(root, study, main, guard)
        for asset in config["public_assets"]:
            path = local_path(study, asset["filename"])
            require(path.stat().st_size == asset["bytes"], "public asset byte count")
            require(sha256(path, guard) == asset["sha256"], "public asset declared digest")
        gates = check_banks(study, main, guard)
        state = torch.load(study / "pytorch_model.bin", map_location="cpu", weights_only=True)
        digest, elements = checkpoint_digest(state)
        require(main["initial_state_sha256"] == main["final_state_sha256"] == digest, "state byte identities")
        require(
            main["model_state_sha256"] == digest and main["model_parameters"] == elements, "state metadata"
        )
        require(
            main["architecture_proof"]
            == {"transpose_classes_exact": True, "generator_exact_except_unused_fc1": True},
            "architecture proof fields",
        )
        gates.update(
            strict_finite_tensor_loading=True,
            model_state_unchanged=True,
            architecture_equivalence=all(main["architecture_proof"].values()),
        )
        require(gates == main["gates"], "all technical gate values")
        require(main["technical_generation_feasible"] == all(gates.values()), "technical conjunction")
        require(all(gates.values()), "technical gate failure")
        check_claims(main)
        resources = check_resources(study, main, config)
        ledger = read_json(study / "prospective_ledger.json")
        require(ledger["entries"][0]["sha256"] == main_sha, "ledger main binding")
        require(check_bindings(root, study, main, guard) == bindings, "bindings unchanged after audit")
        require(sha256(study / "summary.json", guard) == main_sha, "main unchanged after audit")
        require(sha256(Path(__file__), guard) == script_sha, "checker unchanged after audit")
        require(sha256(root / CORE_NAME, guard) == CORE_SHA256, "core unchanged after audit")
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "main_sha256": main_sha,
            "script_sha256": script_sha,
            "core_sha256": CORE_SHA256,
            "gates": gates,
            "bindings_verified": len(bindings),
            "resources": resources,
            "independent_model_forwards": 0,
            "gpu_used": False,
            "raw_real_ecg_reads": 0,
            "saved_replay_verified": True,
            "physical_units_verified": False,
            "runtime_immutability_source_bound": True,
            "architecture_execution_source_bound": True,
            "clinical_claim": False,
            "accounting_scope": "prospective_standalone_only",
            "cpu_seconds": time.process_time() - clock[0],
            "wall_seconds": time.perf_counter() - clock[1],
        }
        result["cumulative_cpu_seconds"] = resources["main_cpu_seconds"] + result["cpu_seconds"]
        require(result["cumulative_cpu_seconds"] < 300, "additive standalone CPU cap")
        result["global_gpu_wall_seconds"] = 0.0
        result["study_bytes_before_audit_receipt"] = sum(
            p.stat().st_size for p in study.rglob("*") if p.is_file()
        )
        with (destination / "summary.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return result
    except Exception as error:
        with (destination / "failure.json").open("x") as stream:
            json.dump(
                {
                    "all_passed": False,
                    "failure_type": type(error).__name__,
                    "failure_detail": str(error),
                    "main_sha256": main_sha,
                    "script_sha256": script_sha,
                    "cpu_seconds": time.process_time() - clock[0],
                    "wall_seconds": time.perf_counter() - clock[1],
                },
                stream,
                indent=2,
            )
            stream.write("\n")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)


def main() -> None:
    """Parse explicit audit bindings and print aggregate status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root.absolute(), args.main_sha, args.script_sha)))


if __name__ == "__main__":
    main()
