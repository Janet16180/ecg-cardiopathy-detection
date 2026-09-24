"""Versioned canonical training data; labels are opt-in and SSL never exposes them."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def signal_hash(signal):
    return hashlib.sha256(np.ascontiguousarray(signal, dtype='<f4').tobytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def checked_path(directory, relative):
    path = (directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError('Dataset path escapes its directory')
    return path


def validate_signal(signal):
    if signal.dtype != np.float32 or signal.shape != (12, 5000) or not np.isfinite(signal).all():
        raise ValueError('Expected finite float32 [12,5000] waveform')
    if np.any(np.ptp(signal, axis=1) == 0):
        raise ValueError('Full constant lead')


class TrainingECGDataset:
    """PyTorch-compatible map dataset returning a signal and masked/available target.

    ``purpose='ssl'`` returns every train row with target=-1 and target_available=False.
    ``purpose='supervised'`` selects only PTB labels from budget '1' or '0.1'.
    Arrays are copied out of read-only mmap storage; no preprocessing is fitted.
    """

    def __init__(self, directory, purpose='ssl', label_budget='1', max_open_shards=4):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / 'metadata.json').read_text())
        if (not self.metadata.get('complete') or self.metadata.get('schema_version') != 1 or
                self.metadata.get('shape_per_record') != [12, 5000] or
                self.metadata.get('sampling_rate_hz') != 500 or self.metadata.get('units') != 'mV'):
            raise ValueError('Invalid or incomplete canonical dataset')
        if purpose not in ('ssl', 'supervised') or label_budget not in ('1', '0.1'):
            raise ValueError('Unsupported purpose or label budget')
        if max_open_shards < 1:
            raise ValueError('max_open_shards must be positive')
        for name, expected in self.metadata['table_sha256'].items():
            if file_hash(checked_path(self.directory, name)) != expected:
                raise ValueError(f'Dataset table hash mismatch: {name}')
        self.rows = read_csv(self.directory / 'train_manifest.csv')
        if len(self.rows) != self.metadata['record_count']:
            raise ValueError('Record count mismatch')
        if any(row['split'] != 'train' for row in self.rows):
            raise ValueError('SSL split or label leakage')
        if len({r['record_id'] for r in self.rows}) != len(self.rows):
            raise ValueError('Duplicate training record identity')
        self.targets = {}
        if purpose == 'supervised':
            labels = read_csv(self.directory / f'labels_fraction{label_budget}.csv')
            self.targets = {r['record_id']: int(r['target']) for r in labels}
            if len(self.targets) != len(labels) or not set(self.targets.values()) <= {0, 1}:
                raise ValueError('Invalid endpoint labels')
            selected = [r for r in self.rows if r['record_id'] in self.targets]
            if len(selected) != len(labels) or any(r['source'] != 'ptbxl' for r in selected):
                raise ValueError('Labels must belong to training PTB rows')
            patients = {r['record_id']: r['patient_id'] for r in selected}
            if any(patients[r['record_id']] != r['patient_id'] for r in labels):
                raise ValueError('Label patient identity mismatch')
            self.rows = selected
        self.max_open_shards = max_open_shards
        self._arrays = OrderedDict()
        self._verified_shards = set()

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_arrays'] = OrderedDict()
        return state

    def __getitem__(self, index):
        row = self.rows[index]
        name = row['shard']
        if name not in self._arrays:
            path = checked_path(self.directory, name)
            expected = self.metadata['shards'][name]
            if name not in self._verified_shards:
                if file_hash(path) != expected['sha256']:
                    raise ValueError(f'Shard hash mismatch: {name}')
                self._verified_shards.add(name)
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(array.shape) != expected['shape'] or array.dtype != np.float32:
                raise ValueError('Shard contract mismatch')
            self._arrays[name] = array
            while len(self._arrays) > self.max_open_shards:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(name)
        signal = np.array(self._arrays[name][int(row['shard_index'])], copy=True)
        validate_signal(signal)
        if signal_hash(signal) != row['signal_sha256']:
            raise ValueError('Waveform hash mismatch')
        available = row['record_id'] in self.targets
        return {'signal': signal, 'target': self.targets.get(row['record_id'], -1),
                'target_available': available, 'record_id': row['record_id'],
                'source': row['source'], 'patient_id': row['patient_id']}


def verify_dataset(directory):
    """Independently reread every published array and both supervised selections."""
    dataset = TrainingECGDataset(directory, max_open_shards=1)
    records, signals, pointers = set(), set(), set()
    for index, row in enumerate(dataset.rows):
        item = dataset[index]
        pointer = (row['shard'], row['shard_index'])
        if row['record_id'] in records or row['signal_sha256'] in signals or pointer in pointers:
            raise ValueError('Duplicate training identity or pointer')
        if item['target_available'] or item['target'] != -1 or row['split'] != 'train':
            raise ValueError('SSL split or label leakage')
        known = row['source'] in ('ptbxl', 'mimic')
        if (row['patient_identity_known'] != str(known).lower() or
                (not known and row['patient_id']) or row['label_scope'] != ('ptbxl_proxy_available_separately' if row['source'] == 'ptbxl' else 'ssl_only')):
            raise ValueError('Unsupported identity or label semantics')
        records.add(row['record_id']); signals.add(row['signal_sha256']); pointers.add(pointer)
    expected = sum(info['shape'][0] for info in dataset.metadata['shards'].values())
    if len(pointers) != expected:
        raise ValueError('Unreferenced shard rows')
    budgets = {budget: len(TrainingECGDataset(directory, 'supervised', budget)) for budget in ('1', '0.1')}
    return {'verified_records': len(dataset), 'verified_shards': len(dataset.metadata['shards']),
            'supervised_records': budgets, 'ssl_targets_masked': True}
