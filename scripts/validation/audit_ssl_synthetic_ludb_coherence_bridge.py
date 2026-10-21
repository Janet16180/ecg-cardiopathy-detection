"""Execute one explicitly admitted independent real/generated coherence audit."""

import argparse
import hashlib
import json
import time
from pathlib import Path


def main() -> None:
    """Capture clocks before heavy imports and require literal checker/runner source identities."""
    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    parser.add_argument("--runner-sha", required=True)
    args = parser.parse_args()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != args.script_sha:
        raise ValueError("Literal copied checker identity differs")
    runner = args.source_root / "ecg_experiment/ssl_synthetic_ludb_coherence_audit_run.py"
    if hashlib.sha256(runner.read_bytes()).hexdigest() != args.runner_sha:
        raise ValueError("Literal independent runner identity differs")
    from ecg_experiment.ssl_synthetic_ludb_coherence_audit_run import run

    result = run(
        args.root, args.source_root, args.admission, args.main_sha, args.script_sha, args.runner_sha, clock
    )
    print(json.dumps({"all_passed": result["all_passed"], "mismatch_count": result["mismatch_count"]}))


if __name__ == "__main__":
    main()
