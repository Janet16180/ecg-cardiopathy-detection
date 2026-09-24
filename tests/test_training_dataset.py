import json
import fcntl
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

from ecg_experiment.training_dataset import TrainingECGDataset, file_hash, signal_hash, verify_dataset
from scripts.build_training_dataset import build, decode, write_csv, FIELDS


@pytest.fixture
def dataset(tmp_path):
    signal = np.tile(np.linspace(-1, 1, 5000, dtype=np.float32), (12, 1))
    signals = np.stack([signal, signal * 2])
    np.save(tmp_path / 'shard_00000.npy', signals)
    rows = []
    for index, source in enumerate(('ptbxl', 'georgia')):
        rows.append(dict(record_id=f'{source}:1', source_record_id='1', source=source,
                         patient_id='ptbxl:99' if index == 0 else '',
                         patient_identity_known='true' if index == 0 else 'false', split='train',
                         label_scope='ptbxl_proxy_available_separately' if index == 0 else 'ssl_only',
                         shard='shard_00000.npy', shard_index=index, signal_sha256=signal_hash(signals[index])))
    write_csv(tmp_path / 'train_manifest.csv', FIELDS, rows)
    for budget in ('1', '0.1'):
        write_csv(tmp_path / f'labels_fraction{budget}.csv', ('record_id','patient_id','target'),
                  [dict(record_id='ptbxl:1', patient_id='ptbxl:99', target=1)])
    metadata = dict(complete=True, schema_version=1, record_count=2, shape_per_record=[12,5000],
                    sampling_rate_hz=500, units='mV', shards={'shard_00000.npy':
                    dict(sha256=file_hash(tmp_path/'shard_00000.npy'), shape=[2,12,5000])},
                    table_sha256={p.name:file_hash(p) for p in tmp_path.glob('*.csv')})
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    return tmp_path


def test_ssl_masks_labels_and_supervised_loader_trains(dataset):
    ssl = TrainingECGDataset(dataset)
    batch = next(iter(DataLoader(ssl, batch_size=2)))
    assert batch['signal'].shape == (2,12,5000)
    assert batch['target'].tolist() == [-1,-1]
    assert not batch['target_available'].any()
    supervised = TrainingECGDataset(dataset, purpose='supervised', label_budget='0.1')
    assert len(supervised) == 1 and supervised[0]['target'] == 1
    assert verify_dataset(dataset)['verified_records'] == 2


def test_shard_tampering_fails(dataset):
    array = np.load(dataset/'shard_00000.npy')
    array[0,0,0] += 0.1
    np.save(dataset/'shard_00000.npy', array)
    with pytest.raises(ValueError, match='Shard hash'):
        TrainingECGDataset(dataset)[0]


def test_unknown_source_cannot_receive_endpoint_label(dataset):
    path = dataset/'labels_fraction1.csv'
    write_csv(path, ('record_id','patient_id','target'), [dict(record_id='georgia:1',patient_id='',target=1)])
    metadata = json.loads((dataset/'metadata.json').read_text())
    metadata['table_sha256'][path.name] = file_hash(path)
    (dataset/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='training PTB'):
        TrainingECGDataset(dataset, purpose='supervised')


def test_cannot_overwrite_published_dataset(dataset):
    with pytest.raises(FileExistsError):
        build(dataset)


def test_split_semantic_tampering_fails_after_rehash(dataset):
    path = dataset/'train_manifest.csv'
    path.write_text(path.read_text().replace(',train,', ',test,'))
    metadata = json.loads((dataset/'metadata.json').read_text())
    metadata['table_sha256'][path.name] = file_hash(path)
    (dataset/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='split or label leakage'):
        verify_dataset(dataset)


def test_single_builder_lock_prevents_duplicate_work(tmp_path):
    destination = tmp_path/'new_dataset'
    with (tmp_path/'.new_dataset.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already holds'):
            build(destination)
    assert not destination.exists()


def test_swapped_ptb_bytes_fail_before_decoding(tmp_path, monkeypatch):
    import scripts.build_training_dataset as builder
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    (tmp_path/'example.hea').write_text('other valid record bytes')
    row = dict(origin_index='', origin_path='example', record_id='ptbxl:1',
               official_sha256={'.hea':'0'*64})
    with pytest.raises(ValueError, match='Official PTB checksum'):
        decode(row)


def test_semantically_bad_waveform_rejected_even_with_new_file_hash(dataset):
    path = dataset/'shard_00000.npy'
    signals = np.load(path)
    signals[0, 10] = 0
    np.save(path, signals)
    metadata = json.loads((dataset/'metadata.json').read_text())
    metadata['shards'][path.name]['sha256'] = file_hash(path)
    (dataset/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Full constant lead'):
        TrainingECGDataset(dataset)[0]
