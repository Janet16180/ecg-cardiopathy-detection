#!/usr/bin/env python3
"""Publish a new canonical 500 Hz train union without altering frozen cohorts."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import wfdb

from ecg_experiment.run import partition_validation
from ecg_experiment.training_dataset import file_hash, read_csv, signal_hash, validate_signal, verify_dataset
from scripts.audit_public_pool_overlap import _load_candidates, _verify_candidate_views, _load_ptb_reference, _load_mimic_reference
from scripts.extract_pretrained import LEADS, read_record
from scripts.download_ptbxl_waveforms import parse_checksums

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('record_id', 'source_record_id', 'source', 'patient_id', 'patient_identity_known',
          'split', 'label_scope', 'shard', 'shard_index', 'signal_sha256',
          'origin_manifest', 'origin_path', 'origin_index', 'view', 'window_start',
          'source_samples', 'qc_flags')


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def prepare_inputs(pin):
    ptb = ROOT / 'data/processed/ptbxl/seed42_fraction1'
    raw_ptb = ROOT / 'data/raw/ptb-xl/1.0.3'
    ptb_checksums = parse_checksums(pin(raw_ptb / 'SHA256SUMS.txt').read_text())
    rows = []
    ptb_reference = _load_ptb_reference(pin)
    mimic_hashes, _ = _load_mimic_reference(pin)
    mimic = ROOT / 'data/processed/mimic_ssl_40k_cpc'
    with sqlite3.connect((mimic / 'audit.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True) as db:
        mimic_expected = dict(db.execute("SELECT name,signal_sha256 FROM outcomes WHERE status='accepted'"))
    pool = json.loads(pin(ROOT / 'data/processed/cpc_pool_40k/complete.json').read_text())
    for name, digest in pool['ptb_manifest_sha256'].items():
        if file_hash(pin(ptb / name)) != digest:
            raise ValueError('Frozen PTB manifest mismatch')
    for source, manifest, raw_dir in (
        ('ptbxl', ptb / 'all_train_ssl.csv', raw_ptb),
        ('mimic', mimic / 'ssl_manifest.csv', Path(json.loads((mimic / 'metadata.json').read_text())['raw_dir'])),
    ):
        for original in read_csv(pin(manifest)):
            source_id = original['ecg_id']
            rows.append({'record_id': f'ptbxl:{source_id}' if source == 'ptbxl' else source_id,
                         'source_record_id': source_id, 'source': source,
                         'patient_id': f"ptbxl:{original['patient_id']}" if source == 'ptbxl' else original['patient_id'],
                         'patient_identity_known': 'true', 'split': 'train',
                         'label_scope': 'ptbxl_proxy_available_separately' if source == 'ptbxl' else 'ssl_only',
                         'origin_manifest': str(manifest.relative_to(ROOT)),
                         'origin_path': str((raw_dir / original['filename_hr']).relative_to(ROOT)),
                         'origin_index': '', 'view': 'original_10s', 'window_start': 0,
                         'source_samples': 5000, 'qc_flags': '', 'raw_dir': raw_dir,
                         'filename_hr': original['filename_hr'],
                         'official_sha256': {suffix: ptb_checksums[original['filename_hr'] + suffix] for suffix in ('.hea','.dat')} if source == 'ptbxl' else {},
                         'expected_hash': mimic_expected[original['filename_hr']] if source == 'mimic' else ''})
    if Counter(r['source'] for r in rows) != {'ptbxl': 17418, 'mimic': 39457}:
        raise ValueError('Historical training pool changed')
    curated, receipt = _load_candidates(ROOT / 'outputs/data_quality/processed_eda', pin)
    _verify_candidate_views(curated, receipt, pin)
    overlap = ROOT / 'outputs/data_quality/astra_review/pool_overlap'
    audit = json.loads(pin(overlap / 'receipt.json').read_text())
    for name, digest in audit['input_sha256'].items():
        if file_hash(pin(Path(name))) != digest:
            raise ValueError('Overlap audit input changed')
    for name, digest in audit['output_sha256'].items():
        if file_hash(pin(overlap / name)) != digest:
            raise ValueError('Overlap audit output changed')
    novel = read_csv(overlap / 'novel_challenge_ssl_manifest.csv')
    overlaps = read_csv(overlap / 'overlaps.csv')
    if (len(novel) != 18844 or len(overlaps) != 945 or
            {r['ecg_id'] for r in curated} != {r['ecg_id'] for r in novel + overlaps} or
            any(r['exact_signal_reference_pools'] != 'georgia_frozen_g1' for r in overlaps)):
        raise ValueError('Unexpected append-overlap gate')
    overlay = read_csv(pin(ROOT / 'outputs/data_quality/processed_eda/training_exclusion_overlay.csv'))
    rail_ids = {r['ecg_id'] for r in overlay}
    if rail_ids & {r['ecg_id'] for r in curated}:
        raise ValueError('Rail-affected candidate entered train')
    frozen_hashes = ptb_reference | mimic_hashes
    for original in curated:
        if original['signal_sha256'] in frozen_hashes:
            raise ValueError('Challenge duplicates frozen PTB/MIMIC')
        rows.append({'record_id': original['ecg_id'], 'source_record_id': original['ecg_id'],
                     'source': original['source'], 'patient_id': '', 'patient_identity_known': 'false',
                     'split': 'train', 'label_scope': 'ssl_only',
                     'origin_manifest': 'outputs/data_quality/processed_eda/challenge_ssl_curated_manifest.csv',
                     'origin_path': original['shard_path'], 'origin_index': original['shard_index'],
                     'view': original['view'], 'window_start': original['window_start'],
                     'source_samples': original['source_samples'], 'qc_flags': original['qc_flags'],
                     'expected_hash': original['signal_sha256']})
    labels = {}
    for budget in ('1', '0.1'):
        path = ROOT / f'data/processed/ptbxl/seed42_fraction{budget}/labeled_train.csv'
        labels[budget] = [{'record_id': 'ptbxl:' + r['ecg_id'], 'target': r['target'],
                           'patient_id': 'ptbxl:' + r['patient_id']} for r in read_csv(pin(path))]
    dev, cal = partition_validation(read_csv(ptb / 'validation.csv'))
    references = []
    train_patients = {r['patient_id'] for r in rows if r['source'] == 'ptbxl'}
    seen_patients = set(train_patients)
    for split, original_rows in (('development', dev), ('calibration', cal), ('test', read_csv(ptb / 'test.csv'))):
        patients = {'ptbxl:' + r['patient_id'] for r in original_rows}
        if seen_patients & patients:
            raise ValueError('PTB patient split leakage')
        seen_patients |= patients
        references.extend({'record_id': 'ptbxl:' + r['ecg_id'], 'patient_id': 'ptbxl:' + r['patient_id'],
                           'split': split, 'target': r['target'],
                           'raw_path': str((raw_ptb / r['filename_hr']).relative_to(ROOT))} for r in original_rows)
    return rows, labels, references, ptb_reference


def decode(row):
    if row['origin_index'] != '':
        signal = np.array(np.load(ROOT / row['origin_path'], mmap_mode='r', allow_pickle=False)[int(row['origin_index'])], copy=True)
    else:
        for suffix, expected in row['official_sha256'].items():
            if file_hash(ROOT / (row['origin_path'] + suffix)) != expected:
                raise ValueError(f"Official PTB checksum mismatch: {row['record_id']}{suffix}")
        header = wfdb.rdheader(str(ROOT / row['origin_path']))
        if header.units != ['mV'] * 12:
            raise ValueError(f"Unverified amplitude units: {row['record_id']}")
        signal = read_record(row['raw_dir'], row['filename_hr'])
        for suffix, expected in row['official_sha256'].items():
            if file_hash(ROOT / (row['origin_path'] + suffix)) != expected:
                raise ValueError('PTB raw file changed while decoding')
    digest = signal_hash(signal)
    if row['expected_hash'] and digest != row['expected_hash']:
        raise ValueError(f"Source waveform identity changed: {row['record_id']}")
    if signal.shape != (12, 5000) or signal.dtype != np.float32 or not np.isfinite(signal).all():
        raise ValueError('Source waveform contract failed')
    constant = [LEADS[i] for i in np.flatnonzero(np.ptp(signal, axis=1) == 0)]
    return signal, digest, constant


def _build_locked(output, workers=4, shard_size=128):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if output.is_relative_to((ROOT / 'data/raw').resolve()):
        raise ValueError('Output must be outside raw data')
    if not 1 <= workers <= 8 or shard_size < 1:
        raise ValueError('Invalid workers or shard size')
    inputs = {}
    def pin(path):
        inputs[str(path.resolve())] = file_hash(path)
        return path
    rows, labels, references, ptb_reference = prepare_inputs(pin)
    print(f'Inputs verified: {len(rows):,} train candidates; {len(references):,} held-out references', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name('.' + output.name + '.staging-' + uuid.uuid4().hex)
    stage.mkdir()
    started = time.monotonic()
    try:
        accepted, exclusions, arrays, shard_rows, shards, seen = [], [], [], [], {}, set()
        def flush():
            if not arrays:
                return
            name = f'shard_{len(shards):05d}.npy'
            matrix = np.stack(arrays)
            np.save(stage / name, matrix, allow_pickle=False)
            shards[name] = {'sha256': file_hash(stage / name), 'shape': list(matrix.shape)}
            for index, row in enumerate(shard_rows):
                row.update(shard=name, shard_index=index)
                accepted.append(row)
            arrays.clear(); shard_rows.clear()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(rows), 128):
                batch = rows[start:start + 128]
                for row, (signal, digest, constants) in zip(batch, pool.map(decode, batch)):
                    if row['source'] == 'ptbxl' and digest not in ptb_reference:
                        raise ValueError('PTB waveform not in checksum-verified reference')
                    if digest in seen:
                        raise ValueError('Unexpected exact duplicate within new train union')
                    seen.add(digest)
                    if constants:
                        exclusions.append({'record_id': row['record_id'], 'source': row['source'],
                                           'reason': 'full_constant_lead', 'detail': ';'.join(constants),
                                           'signal_sha256': digest})
                        continue
                    row['signal_sha256'] = digest
                    arrays.append(signal); shard_rows.append(row)
                    if len(arrays) == shard_size:
                        flush()
                if start % 2048 == 0:
                    print(f"Read {min(start+128,len(rows)):,}/{len(rows):,}; excluded {len(exclusions)}; {time.monotonic()-started:.1f}s", flush=True)
        flush()
        write_csv(stage / 'train_manifest.csv', FIELDS, accepted)
        write_csv(stage / 'exclusions.csv', ('record_id','source','reason','detail','signal_sha256'), exclusions)
        # These historical records never enter the candidates, and are tracked separately.
        quarantine = json.loads(pin(ROOT / 'outputs/data_quality/astra_review/frozen_georgia_constant_leads.json').read_text())
        (stage / 'historical_georgia_quarantine.json').write_text(json.dumps(quarantine, indent=2) + '\n')
        kept = {r['record_id']: r for r in accepted}
        label_counts = {}
        for budget, original in labels.items():
            selected = [r for r in original if r['record_id'] in kept]
            if any(kept[r['record_id']]['patient_id'] != r['patient_id'] for r in selected):
                raise ValueError('PTB label patient mismatch')
            write_csv(stage / f'labels_fraction{budget}.csv', ('record_id','patient_id','target'), selected)
            label_counts[budget] = {'input': len(original), 'retained': len(selected)}
        write_csv(stage / 'heldout_references.csv', ('record_id','patient_id','split','target','raw_path'), references)
        for path, expected in inputs.items():
            if file_hash(path) != expected:
                raise ValueError(f'Input changed while building: {path}')
        metadata = {'schema_version': 1, 'complete': True, 'dataset_id': output.name,
                    'record_count': len(accepted), 'candidate_count': len(rows),
                    'source_counts': dict(Counter(r['source'] for r in accepted)),
                    'new_exclusions_by_source': dict(Counter(r['source'] for r in exclusions)),
                    'shape_per_record': [12,5000], 'dtype': 'float32', 'units': 'mV',
                    'sampling_rate_hz': 500, 'lead_order': list(LEADS),
                    'preprocessing': 'Original physical mV; no filtering, clipping, resampling, normalization or fitted parameters',
                    'label_counts': label_counts, 'heldout_reference_counts': dict(Counter(r['split'] for r in references)),
                    'split_policy': 'PTB official frozen train only; existing seed9001 patient development/calibration partition retained as references; no held-out arrays in train shards',
                    'label_policy': 'PTB diagnostic annotation proxy only, opt-in exact retained frozen label budgets; all other sources SSL-only',
                    'patient_identity': 'PTB/MIMIC namespaced official patient IDs; Challenge unknown, patient_id deliberately empty; source independence is not proven',
                    'composition': 'Historical PTB+MIMIC 56875 candidates plus curated Challenge 19789; Challenge includes 945 clean Georgia pilot overlaps and 18844 novel records; 3 historical Georgia constants never candidates',
                    'limitations': ['No clinical quality or performance validation', 'Exact hashes do not detect transformed near-duplicates or prove cross-source patient independence', 'CODE native and incomplete Chapman/MIMIC200k are excluded', 'New data-scaling cohort; not an equivalent replacement for frozen experiment comparisons'],
                    'input_sha256': inputs, 'shards': shards,
                    'table_sha256': {p.name: file_hash(p) for p in stage.iterdir() if p.suffix in ('.csv','.json')},
                    'git_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    'source_sha256': {str(p.relative_to(ROOT)): file_hash(p) for p in
                                      (Path(__file__), ROOT/'ecg_experiment/training_dataset.py', ROOT/'ecg_experiment/run.py', ROOT/'scripts/extract_pretrained.py', ROOT/'scripts/audit_public_pool_overlap.py')},
                    'command': [sys.executable, *sys.argv], 'build_seconds': time.monotonic()-started}
        (stage / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print('Independently verifying every new shard and both label budgets', flush=True)
        verification = verify_dataset(stage)
        verification.update(metadata_sha256=file_hash(stage/'metadata.json'), elapsed_seconds=time.monotonic()-started)
        (stage/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
        if output.exists():
            raise FileExistsError('Output appeared during build')
        os.rename(stage, output)
        return verification
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def build(output, workers=4, shard_size=128):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if output.is_relative_to((ROOT / 'data/raw').resolve()):
        raise ValueError('Output must be outside raw data')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name('.' + output.name + '.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A builder already holds this dataset lock') from None
        lock.seek(0); lock.truncate(); lock.write(str(os.getpid()) + '\n'); lock.flush()
        return _build_locked(output, workers, shard_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_dataset(args.output_dir) if args.verify_only else build(args.output_dir, args.workers), indent=2))


if __name__ == '__main__':
    main()
