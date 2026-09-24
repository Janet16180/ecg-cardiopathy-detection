"""Download the official PTB-XL v1.0.3 annotation tables (about 6.6 MB)."""

import argparse
import csv
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
FILES = {
    "ptbxl_database.csv": {"ecg_id", "patient_id", "scp_codes", "strat_fold"},
    "scp_statements.csv": {"diagnostic", "diagnostic_class"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/ptb-xl/1.0.3"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, required in FILES.items():
        destination = args.output_dir / name
        if destination.exists():
            payload = destination.read_bytes()
        else:
            with urlopen(BASE_URL + name, timeout=120) as response:
                payload = response.read()
        headers = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
        if not required.issubset(headers):
            raise ValueError(f"Unexpected columns in {name}; refusing to save it")
        if not destination.exists():
            temporary = destination.with_suffix(".csv.tmp")
            temporary.write_bytes(payload)
            temporary.replace(destination)
        digest = hashlib.sha256(payload).hexdigest()
        print(f"{destination}: {len(payload):,} bytes; sha256={digest}")


if __name__ == "__main__":
    main()
