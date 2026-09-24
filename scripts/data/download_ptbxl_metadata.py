"""Download the official PTB-XL v1.0.3 annotation tables (about 6.6 MB)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
REQUEST_TIMEOUT_SECONDS = 120
FILES = {
    "ptbxl_database.csv": {"ecg_id", "patient_id", "scp_codes", "strat_fold"},
    "scp_statements.csv": {"diagnostic", "diagnostic_class"},
}


def table_bytes(destination: Path, name: str) -> bytes:
    """
    Return the local table bytes, downloading them when the file is absent.

    Parameters
    ----------
    destination : Path
        Local path of the table.
    name : str
        File name under ``BASE_URL``.

    Returns
    -------
    bytes
        Table contents.
    """
    if destination.exists():
        return destination.read_bytes()
    with urlopen(BASE_URL + name, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read()


def save_table(destination: Path, name: str, required: set[str]) -> None:
    """
    Check a table's header and save it atomically if it is new.

    Parameters
    ----------
    destination : Path
        Local path of the table.
    name : str
        File name under ``BASE_URL``.
    required : set[str]
        Columns the header must contain.

    Raises
    ------
    ValueError
        If the header lacks a required column; nothing is saved.
    """
    payload = table_bytes(destination, name)
    headers = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    if not required.issubset(headers):
        raise ValueError(f"Unexpected columns in {name}; refusing to save it")
    if not destination.exists():
        temporary = destination.with_suffix(".csv.tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    digest = hashlib.sha256(payload).hexdigest()
    print(f"{destination}: {len(payload):,} bytes; sha256={digest}")


def main() -> None:
    """Download and check both annotation tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/ptb-xl/1.0.3")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, required in FILES.items():
        save_table(args.output_dir / name, name, required)


if __name__ == "__main__":
    main()
