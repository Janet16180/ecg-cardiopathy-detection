"""Download a checksum-verified Georgia pilot pool for unlabeled adaptation."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np
import wfdb

from scripts.download_ptbxl_waveforms import fetch_file, parse_checksums, sha256
from scripts.extract_pretrained import read_record


BASE = "https://physionet.org/files/challenge-2020/1.0.2"


def signal_hash(signal):
    # Canonical lead order and physical mV; exact equality after float32 decoding.
    return hashlib.sha256(np.ascontiguousarray(signal, dtype="<f4").tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/challenge-2020/1.0.2"))
    parser.add_argument("--ptbxl-dir", type=Path, default=Path("data/raw/ptb-xl/1.0.3"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/georgia_ssl_g1"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checksum_path = args.raw_dir / "SHA256SUMS.txt"
    if not checksum_path.exists():
        fetch_file(BASE + "/SHA256SUMS.txt", checksum_path, 60, 3)
    checksums = parse_checksums(checksum_path.read_text())
    prefix = "training/georgia/g1/"
    stems = sorted(name[:-4] for name in checksums if name.startswith(prefix) and name.endswith(".hea"))
    if not stems:
        raise ValueError("Official checksums contain no Georgia g1 headers")
    paths = [stem + suffix for stem in stems for suffix in (".hea", ".mat")]
    for path in paths:
        if path not in checksums or ".." in PurePosixPath(path).parts:
            raise ValueError(f"Missing checksum or unsafe path: {path}")

    def download(name):
        destination = args.raw_dir / name
        if not destination.exists() or sha256(destination) != checksums[name]:
            fetch_file(BASE + "/" + name, destination, 60, 3,
                       lambda file: sha256(file) == checksums[name])
        return destination.stat().st_size

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, name) for name in paths]
        for i, future in enumerate(as_completed(futures), 1):
            total_bytes += future.result()
            if i % 200 == 0 or i == len(paths):
                print(f"Verified {i}/{len(paths)} Georgia files", flush=True)

    # Remove exact decoded-waveform matches to ANY PTB-XL fold, including held-out ECGs.
    # This is an identity/integrity audit; no targets or reports enter the SSL manifest.
    ptb_hashes = set()
    with (args.ptbxl_dir / "ptbxl_database.csv").open() as handle:
        ptb_rows = list(csv.DictReader(handle))
    for i, row in enumerate(ptb_rows, 1):
        ptb_hashes.add(signal_hash(read_record(args.ptbxl_dir, row["filename_hr"])))
        if i % 5000 == 0:
            print(f"Audited {i}/{len(ptb_rows)} PTB-XL waveform identities", flush=True)
    accepted, excluded, seen = [], [], set()
    for stem in stems:
        try:
            header = wfdb.rdheader(str(args.raw_dir / stem))
            if header.units != ["mV"] * 12:
                raise ValueError(f"Expected all leads in mV; found {header.units}")
            signal = read_record(args.raw_dir, stem)
        except (ValueError, OSError) as error:
            excluded.append({"record": stem, "reason": "input_contract", "detail": str(error)})
            continue
        digest = signal_hash(signal)
        if digest in ptb_hashes or digest in seen:
            excluded.append({"record": stem, "reason": "exact_decoded_waveform_duplicate"})
            continue
        seen.add(digest)
        name = PurePosixPath(stem).name
        accepted.append({"ecg_id": "georgia:" + name,
                         "patient_id": "georgia:record_" + name,
                         "raw_dir": str(args.raw_dir.resolve()), "filename_hr": stem,
                         "source": "georgia"})
    destination = args.output_dir / "ssl_manifest.csv"
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ecg_id", "patient_id", "raw_dir", "filename_hr", "source"])
        writer.writeheader()
        writer.writerows(accepted)
    metadata = {
        "source_url": BASE, "license": "CC BY 4.0", "selection": "all records in g1 folder; convenience pilot, not random/full cohort",
        "downloaded_records": len(stems), "accepted_records": len(accepted), "excluded": excluded,
        "downloaded_bytes": total_bytes, "checksums_sha256": sha256(checksum_path),
        "manifest_sha256": sha256(destination), "signal_identity": "SHA256 of canonical lead-ordered float32 physical mV",
        "ptbxl_records_checked": len(ptb_rows),
        "patient_identity": "Not available; patient_id denotes record grouping only. Same-person records may remain.",
        "labels": "No Georgia diagnoses or reports used for training or evaluation.",
        "pretraining_exposure": "Released ECG-FM and HuBERT already include Georgia through PhysioNet pretraining.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
