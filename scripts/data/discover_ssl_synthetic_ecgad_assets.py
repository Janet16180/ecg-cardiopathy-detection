"""Preflight or run the separately admitted public localization asset discovery."""

import time

STARTED = (time.perf_counter(), time.process_time())


def main() -> None:
    """Keep metadata-only admission checks distinct from actual bounded HTTP execution."""
    import argparse
    from pathlib import Path

    from ecg_experiment.ssl_synthetic_ecgad_asset_discovery import discover, preflight

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.root, args.admission)
        print({"preflight_passed": True, "requests": 0, "binary_downloads": 0})
        return
    if not discover(args.root, args.admission, STARTED)["technical_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
