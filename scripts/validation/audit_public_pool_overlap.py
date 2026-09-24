#!/usr/bin/env python3
"""Read-only exact-identity gate for appending Challenge candidates to frozen SSL data.

The novel output is a candidate manifest, not a scheduled union or a patient split.
MIMIC hashes come from its frozen preparation audit; raw MIMIC is not redecoded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from scripts.download_ptbxl_waveforms import parse_checksums, sha256
from scripts.extract_pretrained import read_record
from scripts.data.prepare_public_ecg import ROOT, signal_sha256


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def classify_overlap(rows: list[dict], references: dict[str, set[str]],
                     existing_ids: set[str]) -> tuple[list[dict], list[dict]]:
    if len({r['ecg_id'] for r in rows}) != len(rows) or len({r['signal_sha256'] for r in rows}) != len(rows):
        raise ValueError('Candidate contains duplicate identity')
    novel, overlaps = [], []
    for row in rows:
        matches = sorted(name for name, hashes in references.items() if row['signal_sha256'] in hashes)
        same_id = row['ecg_id'] in existing_ids
        if matches or same_id:
            overlaps.append({'ecg_id': row['ecg_id'], 'source': row['source'],
                             'signal_sha256': row['signal_sha256'],
                             'exact_signal_reference_pools': ';'.join(matches),
                             'existing_record_id': str(same_id).lower()})
        else:
            novel.append(row)
    return novel, overlaps


# Helpers receive the audit's recorder so every metadata dependency is pinned.
PinInput = Callable[[Path], Path]
OVERLAP_FIELDS = (
    "ecg_id", "source", "signal_sha256", "exact_signal_reference_pools",
    "existing_record_id",
)


def _load_candidates(candidate_dir: Path, pin: PinInput) -> tuple[list[dict], dict]:
    """Load the curated SSL candidate and enforce its declared eligibility."""
    candidate_path = pin(candidate_dir / 'challenge_ssl_curated_manifest.csv')
    receipt = json.loads(pin(candidate_dir / 'challenge_ssl_curated_receipt.json').read_text())
    if sha256(candidate_path) != receipt['curated_manifest_sha256']:
        raise ValueError('Curated candidate manifest mismatch')
    rows = read_csv(candidate_path)
    if not rows or len(rows) != receipt['counts']['selected_total']:
        raise ValueError('Candidate count mismatch')
    for row in rows:
        if (row['endpoint_supervised_eligible'] != 'false' or row['label_scope'] != 'ssl_only' or
                row['patient_independent_eval_eligible'] != 'false'):
            raise ValueError('Candidate claims unsupported supervised/evaluation eligibility')
    return rows, receipt


def _verify_candidate_views(rows: list[dict], receipt: dict, pin: PinInput) -> None:
    """Reconcile each candidate pointer with the published source view."""
    views = {}
    for view, receipt_key in (("strict_10s", "strict_manifest_sha256"),
                               ("cpsc_ssl_center_crop", "centered_manifest_sha256")):
        directory = ROOT / 'data/processed/challenge_ecg_views' / view
        manifest_path = pin(directory / 'manifest.csv')
        materialized = json.loads(pin(directory / 'metadata.json').read_text())
        if (not materialized.get('complete') or sha256(manifest_path) != receipt[receipt_key] or
                sha256(manifest_path) != materialized['manifest_sha256']):
            raise ValueError('Published Challenge view identity mismatch')
        views[view] = {row['ecg_id']: row for row in read_csv(manifest_path)}
    for row in rows:
        original = views.get(row['view'], {}).get(row['ecg_id'])
        if original is None or any(row[key] != original[key] for key in (
                'source', 'signal_sha256', 'shard_index', 'window_start', 'source_samples',
                'window_samples', 'sampling_rate_hz', 'units', 'lead_order')):
            raise ValueError('Candidate differs from published waveform reference')
        expected_path = ROOT / 'data/processed/challenge_ecg_views' / row['view'] / original['shard']
        if (ROOT / row['shard_path']).resolve() != expected_path.resolve():
            raise ValueError('Candidate shard path mismatch')


def _load_ptb_reference(pin: PinInput) -> set[str]:
    """Require the reference bound to completed official-byte verification."""
    reference_path = pin(ROOT / 'outputs/data_quality/ptbxl_reference_hashes_v2.json')
    reference = json.loads(reference_path.read_text())
    pilot_path = ROOT / 'data/processed/public_ecg_quality/astra_v2_pilot32/metadata.json'
    pilot = json.loads(pin(pilot_path).read_text())
    hashes_payload = json.dumps(reference['hashes'], separators=(',', ':')).encode()
    hashes_digest = hashlib.sha256(hashes_payload).hexdigest()
    if (reference.get('schema_version') != 2 or
            pilot['ptb_reference_receipt_sha256'] != sha256(reference_path) or
            reference['hashes_sha256'] != hashes_digest):
        raise ValueError('PTB reference lacks matching completed official-byte verification')
    return set(reference['hashes'])


def _load_mimic_reference(pin: PinInput) -> tuple[set[str], set[str]]:
    """Read frozen identities without opening or changing raw MIMIC records."""
    mimic = ROOT / 'data/processed/mimic_ssl_40k_cpc'
    metadata = json.loads(pin(mimic / 'metadata.json').read_text())
    manifest = pin(mimic / 'ssl_manifest.csv')
    if sha256(manifest) != metadata['manifest_sha256']:
        raise ValueError('Frozen MIMIC manifest mismatch')
    rows = read_csv(manifest)
    database = pin(mimic / 'audit.sqlite3')
    if Path(str(database) + '-wal').exists():
        raise ValueError('MIMIC audit is not a closed immutable SQLite snapshot')
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as connection:
        state = dict(connection.execute('SELECT key,value FROM state'))
        accepted = dict(connection.execute("SELECT name,signal_sha256 FROM outcomes WHERE status='accepted'"))
    if (state.get('selection_sha256') != metadata['selection_sha256'] or
            set(accepted) != {r['filename_hr'] for r in rows} or
            len(accepted) != metadata['accepted_records'] or
            len(set(accepted.values())) != len(accepted) or
            any(not isinstance(h, str) or len(h) != 64 for h in accepted.values())):
        raise ValueError('Frozen MIMIC audit identities do not match its accepted manifest')
    return set(accepted.values()), {row['ecg_id'] for row in rows}


def _load_georgia_reference(pin: PinInput) -> tuple[set[str], set[str]]:
    """Verify official raw bytes and decode the historical Georgia pilot."""
    georgia = ROOT / 'data/processed/georgia_ssl_g1'
    metadata = json.loads(pin(georgia / 'metadata.json').read_text())
    manifest = pin(georgia / 'ssl_manifest.csv')
    if sha256(manifest) != metadata['manifest_sha256']:
        raise ValueError('Frozen Georgia manifest mismatch')
    rows = read_csv(manifest)
    if len(rows) != metadata['accepted_records']:
        raise ValueError('Frozen Georgia count mismatch')
    raw = ROOT / 'data/raw/challenge-2020/1.0.2'
    sums_file = pin(raw / 'SHA256SUMS.txt')
    if sha256(sums_file) != metadata['checksums_sha256']:
        raise ValueError('Georgia official checksum reference mismatch')
    sums = parse_checksums(sums_file.read_text())
    hashes = set()
    for row in rows:
        stem = row['filename_hr']
        if Path(row['raw_dir']).resolve() != raw.resolve() or not stem.startswith('training/georgia/g1/'):
            raise ValueError('Georgia raw source mismatch')
        for suffix in ('.hea', '.mat'):
            name = stem + suffix
            path = (raw / name).resolve()
            if not path.is_relative_to(raw.resolve()) or sha256(path) != sums.get(name):
                raise ValueError('Georgia official raw checksum mismatch')
        hashes.add(signal_sha256(read_record(raw, stem)))
    return hashes, {row['ecg_id'] for row in rows}


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _publish_audit(output_dir: Path, rows: list[dict], novel: list[dict],
                   overlap: list[dict], pools: dict[str, set[str]], inputs: dict) -> dict:
    """Publish both manifests and their receipt as one immutable directory."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(f'.{output_dir.name}.staging-{uuid.uuid4().hex}')
    stage.mkdir()
    try:
        _write_csv(stage / 'overlaps.csv', OVERLAP_FIELDS, overlap)
        _write_csv(stage / 'novel_challenge_ssl_manifest.csv', tuple(rows[0]), novel)
        overlap_by_pool = {
            pool: sum(pool in row['exact_signal_reference_pools'].split(';') for row in overlap)
            for pool in pools
        }
        result = {
            'status': 'exact_overlap_audited_candidate_only_not_scheduled',
            'candidate_records': len(rows),
            'overlapping_records': len(overlap),
            'novel_records': len(novel),
            'unfiltered_candidate_append_gate_passed': not overlap,
            'reference_unique_hashes': {pool: len(hashes) for pool, hashes in pools.items()},
            'overlap_by_reference_pool': overlap_by_pool,
            'overlap_by_record_id': sum(row['existing_record_id'] == 'true' for row in overlap),
            'novel_source_counts': dict(Counter(row['source'] for row in novel)),
            'input_sha256': inputs,
            'source_sha256': sha256(Path(__file__)),
            'output_sha256': {
                name: sha256(stage / name)
                for name in ('overlaps.csv', 'novel_challenge_ssl_manifest.csv')
            },
            'limitations': [
                'MIMIC identities rely on its frozen preparation audit; raw MIMIC was not redecoded.',
                'PTB identities refer to the completed official-checksum-verified v2 preparation snapshot.',
                'Exact identities do not detect shifted/resampled/partial near-duplicates or unknown patient overlap.',
                'No labels mapped, patient split assigned, frozen pool replaced or training scheduled.',
            ],
        }
        (stage / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        if output_dir.exists():
            raise FileExistsError('Output appeared during audit')
        os.rename(stage, output_dir)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def audit(candidate_dir: Path, output_dir: Path) -> dict:
    """Validate inputs, classify overlap, then publish without changing any input."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {output_dir}')
    if output_dir.is_relative_to((ROOT / 'data/raw').resolve()):
        raise ValueError('Output must be outside raw data')

    inputs: dict[str, str] = {}

    def pin(path: Path) -> Path:
        inputs[str(path.resolve())] = sha256(path)
        return path

    rows, receipt = _load_candidates(candidate_dir, pin)
    _verify_candidate_views(rows, receipt, pin)
    pools = {'ptbxl_all_folds': _load_ptb_reference(pin)}
    pools['mimic_frozen_40k'], mimic_ids = _load_mimic_reference(pin)
    pools['georgia_frozen_g1'], georgia_ids = _load_georgia_reference(pin)
    novel, overlap = classify_overlap(rows, pools, mimic_ids | georgia_ids)

    for path, digest in inputs.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f'Input changed during audit: {path}')
    return _publish_audit(output_dir, rows, novel, overlap, pools, inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dir", type=Path, default=ROOT / "outputs/data_quality/processed_eda",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.candidate_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
