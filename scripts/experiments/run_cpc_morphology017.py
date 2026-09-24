#!/usr/bin/env python3
"""Experiment 017: matched causal morphology-template CPC development pilot."""

import argparse
import fcntl
import hashlib
import json
import random
import signal
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F

from ecg_experiment.bounded_waveform_cache import BoundedWaveformCache
from ecg_experiment.cpc import CPCClassifier
from ecg_experiment.cpc_morphology import MorphologyCPCClassifier, TEMPLATES, SUPPORT
from ecg_experiment.data import read_manifest
from ecg_experiment.evaluation import select_threshold
from ecg_experiment.run import cpu_state, partition_validation
from scripts.experiments.run_cpc_experiment import Pool, atomic_json, atomic_torch, digest_file, seed_all

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'data/processed/cpc_pool_40k'
MANIFEST = ROOT / 'data/processed/ptbxl'
SSL = ROOT / 'outputs/experiment004_cpc_40k/cpc_ssl/encoder.pt'
NORMALIZATION = ROOT / 'outputs/experiment004_cpc_40k/normalization.json'
OUTPUT = ROOT / 'outputs/experiment017_morphology_templates'
POOL_VERIFICATION = ROOT / 'outputs/experiment015_jepa_cpc_distillation/provenance/verification.json'
LOCK = Path('/tmp/ecg_project_gpu.lock')
SEED = 42
EPOCHS = 5
BATCH = 128
SAVE_EVERY = 20
STOP = False


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_identity(path):
    """Identity fields recorded by the frozen Experiment 015 pool verifier."""
    stat = Path(path).stat()
    return {'device': stat.st_dev, 'inode': stat.st_ino, 'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def verified_pool_hashes(args, pool):
    """Reuse 015's complete SHA check only while all source files are identical."""
    if not args.pool_verification.exists():
        raise FileNotFoundError(f'Verified immutable CPC pool receipt is missing: {args.pool_verification}')
    before = {name: file_identity(args.cache_dir / name)
              for name in ('signals.npy', 'rows.csv', 'ecg_ids.npy')}
    receipt = json.loads(args.pool_verification.read_text())
    if receipt.get('stage') != 'check' or receipt.get('fingerprint') != digest_json(receipt.get('provenance')):
        raise ValueError('Invalid pool verification receipt')
    source_hash = receipt.get('provenance', {}).get('code', {}).get('scripts/run_jepa_cpc_distillation.py')
    if source_hash != digest_file(ROOT / 'scripts/experiments/run_jepa_cpc_distillation.py'):
        raise ValueError('Frozen pool verifier source changed')
    hashes = receipt['provenance']['pool_content_sha256']
    for filename, key in (('signals.npy', 'signals_sha256'), ('rows.csv', 'rows_sha256'),
                          ('ecg_ids.npy', 'ecg_ids_sha256')):
        if receipt['pool_file_stats'].get(filename) != before[filename] or hashes.get(filename) != pool.metadata[key]:
            raise ValueError(f'CPC pool differs from verified receipt: {filename}')
    if any(file_identity(args.cache_dir / name) != before[name] for name in before):
        raise ValueError('CPC pool changed while checking receipt')
    return hashes, before, source_hash, digest_file(args.pool_verification)


def stopping(_signum, _frame):
    global STOP
    STOP = True


def fixed_batches(size, exposed, epoch):
    """One exposure per training row; 1,518 labels distributed across batches."""
    rng = np.random.default_rng(SEED + 1009 * epoch)
    count = (size + BATCH - 1) // BATCH
    labels = rng.permutation(np.flatnonzero(exposed))
    hidden = rng.permutation(np.flatnonzero(~exposed))
    if len(labels) < count:
        raise ValueError('Too few exposed labels for one labeled row per batch')
    groups = np.array_split(labels, count)
    result = []
    offset = 0
    for index, labeled in enumerate(groups):
        capacity = min(BATCH, size - index * BATCH) - len(labeled)
        batch = rng.permutation(np.concatenate((labeled, hidden[offset:offset + capacity])))
        offset += capacity
        result.append(batch.tolist())
    flat = [i for batch in result for i in batch]
    if len(flat) != size or len(set(flat)) != size or set(flat) != set(range(size)) or not all(exposed[b].any() for b in result):
        raise AssertionError('Training schedule lost rows or labels')
    return result


def template_windows(cache, train_count, mean, std):
    """Seed-fixed label-blind windows from training rows only, in normalized units."""
    rng = np.random.default_rng(SEED + 17017)
    rows = rng.choice(train_count, size=TEMPLATES, replace=False)
    halves = rng.integers(0, 2, size=TEMPLATES)
    ends = rng.integers(SUPPORT - 1, 1250, size=TEMPLATES)
    bank = np.empty((TEMPLATES, 12, SUPPORT), dtype=np.float32)
    receipt = []
    for i, (row, half, end) in enumerate(zip(rows, halves, ends)):
        start = int(half) * 1250 + int(end) - SUPPORT + 1
        window = np.array(cache.signals[int(row), :, start:start + SUPPORT], copy=True)
        window -= mean
        window /= std
        if not np.isfinite(window).all():
            raise ValueError('Nonfinite initial template')
        bank[i] = window
        receipt.append({'ecg_id': cache.rows[int(row)]['ecg_id'], 'half': int(half),
                        'end_in_half': int(end)})
    return torch.from_numpy(bank), receipt


def load_inputs(args):
    began = time.monotonic()
    pool = Pool(args.cache_dir)
    full = read_manifest(args.manifest_dir / 'seed42_fraction1/labeled_train.csv')
    limited = read_manifest(args.manifest_dir / 'seed42_fraction0.1/labeled_train.csv')
    validation = read_manifest(args.manifest_dir / 'seed42_fraction1/validation.csv')
    development, calibration = partition_validation(validation)
    test = read_manifest(args.manifest_dir / 'seed42_fraction1/test.csv')
    if tuple(map(len, (full, limited, development, calibration, test))) != (15360, 1518, 1306, 564, 1896):
        raise ValueError('Frozen partition counts changed')
    by_id = {r['ecg_id']: r for r in full}
    if len(by_id) != len(full) or any(r['ecg_id'] not in by_id or
        (r['patient_id'], r['target']) != (by_id[r['ecg_id']]['patient_id'], by_id[r['ecg_id']]['target'])
        for r in limited):
        raise ValueError('Limited budget differs from frozen full training set')
    for name, expected in (('validation', validation), ('test', test)):
        if read_manifest(args.manifest_dir / f'seed42_fraction0.1/{name}.csv') != expected:
            raise ValueError(f'Budget-specific {name} manifest changed')
    for rows, split in ((full, 'train'), (development, 'validation'),
                        (calibration, 'validation'), (test, 'test')):
        for row in rows:
            old = pool.rows[pool.index[row['ecg_id']]]
            if (old['source'], old['split'], old['patient_id']) != ('ptbxl', split, row['patient_id']):
                raise ValueError(f'Manifest/cache mismatch: {row["ecg_id"]}')
    patients = [set(r['patient_id'] for r in rows) for rows in (full, development, calibration, test)]
    if any(patients[i] & patients[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError('Patient partitions overlap')
    norm = json.loads(args.normalization.read_text())
    ssl_config = json.loads((args.ssl.parent / 'config.json').read_text())
    saved = torch.load(args.ssl, map_location='cpu', weights_only=True)
    final = torch.load(args.ssl.parent / 'epoch_state.pt', map_location='cpu', weights_only=False)
    if saved.get('variant') != 'cpc' or saved.get('epochs') != 20 or saved.get('fingerprint') != ssl_config['fingerprint'] or final['epoch'] != 20 or final['fingerprint'] != ssl_config['fingerprint']:
        raise ValueError('Not the final ordinary Experiment 004 CPC encoder')
    if any(not torch.equal(value, final['model'][f'encoder.{key}']) for key, value in saved['encoder'].items()):
        raise ValueError('CPC checkpoint differs from final SSL state')
    del saved, final
    hashes, pool_stats, verifier_source_hash, pool_receipt_hash = verified_pool_hashes(args, pool)
    if norm['source']['signals_sha256'] != hashes['signals.npy'] or norm['source']['train_ids_sha256'] != digest_json([r['ecg_id'] for r in pool.train_rows]):
        raise ValueError('Normalization not fitted on frozen CPC training pool')
    mean = np.asarray(norm['mean'], dtype=np.float32)[:, None]
    std = np.asarray(norm['std'], dtype=np.float32)[:, None]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid train-only normalization')
    manifest_paths = [args.manifest_dir / f'seed42_fraction{budget}' / f'{name}.csv'
                      for budget in ('1', '0.1') for name in ('labeled_train', 'validation', 'test')]
    manifest_paths.append(args.manifest_dir / 'seed42_fraction1/all_train_ssl.csv')
    for name, expected in pool.metadata.get('ptb_manifest_sha256', {}).items():
        if digest_file(args.manifest_dir / 'seed42_fraction1' / name) != expected:
            raise ValueError(f'Frozen pool PTB manifest changed: {name}')
    code_paths = ('scripts/experiments/run_cpc_morphology017.py', 'ecg_experiment/cpc_morphology.py',
                  'ecg_experiment/cpc.py', 'ecg_experiment/bounded_waveform_cache.py',
                  'scripts/experiments/run_cpc_experiment.py', 'ecg_experiment/data.py',
                  'ecg_experiment/run.py', 'ecg_experiment/evaluation.py',
                  'docs/experiment-017-morphology.md')
    provenance = {'sources': {str(path.resolve()): digest_file(path) for path in
                              [args.ssl, args.ssl.parent / 'epoch_state.pt',
                               args.ssl.parent / 'config.json', args.normalization, *manifest_paths]},
                  'code': {name: digest_file(ROOT / name) for name in code_paths},
                  'pool_complete_sha256': digest_file(args.cache_dir / 'complete.json'),
                  'pool_content_sha256': hashes,
                  'pool_file_stats': pool_stats,
                  'pool_verifier_source_sha256': verifier_source_hash,
                  'ssl_fingerprint': ssl_config['fingerprint'],
                  'settings': {'seed': SEED, 'epochs': EPOCHS, 'batch': BATCH,
                               'encoder_lr': 3e-4, 'head_and_branch_lr': 1e-3,
                               'weight_decay': 0.01, 'templates': TEMPLATES,
                               'support_samples': SUPPORT, 'save_every': SAVE_EVERY}}
    precheck_seconds = time.monotonic() - began
    if args.stage == 'check':
        sample = np.array(pool.signals[pool.indices(full[:1])[0]], copy=True)
        if sample.shape != (12, 2500) or not np.isfinite(sample).all():
            raise ValueError('Invalid waveform sample')
        return locals()
    preload = time.monotonic()
    waveforms = BoundedWaveformCache(pool, full + development, max_bytes=args.max_cache_bytes,
                                     reserve_bytes=args.reserve_bytes, expected_source='ptbxl')
    bank, template_receipt = template_windows(waveforms, len(full), mean, std)
    preload_seconds = time.monotonic() - preload
    return locals()


def make_model(args, kind, bank):
    seed_all(SEED)
    # Construct the common original classifier first: adding a branch must not
    # shift the classifier-head RNG stream across comparison arms.
    reference = CPCClassifier()
    reference_head = cpu_state(reference.head)
    encoder_keys = set(reference.encoder.state_dict())
    model = MorphologyCPCClassifier(kind, bank if kind != 'none' else None)
    model.head.load_state_dict(reference_head)
    saved = torch.load(args.ssl, map_location='cpu', weights_only=True)
    model.encoder.convs.load_state_dict({key[len('convs.'):]: val for key, val in saved['encoder'].items() if key.startswith('convs.')}, strict=True)
    model.encoder.context.load_state_dict({key[len('context.'):]: val for key, val in saved['encoder'].items() if key.startswith('context.')}, strict=True)
    if set(saved['encoder']) != encoder_keys:
        raise ValueError('Bootstrap encoder source keys changed')
    model = model.to(args.device)
    # The first training dropout mask is also common across arms. Resume state
    # overrides this with its exact saved RNG state.
    seed_all(SEED + 12345)
    return model


def optimizer_for(model):
    encoder = model.encoder
    base = list(encoder.convs.parameters()) + list(encoder.context.parameters())
    groups = [{'params': base, 'lr': 3e-4}, {'params': model.head.parameters(), 'lr': 1e-3}]
    if encoder.kind != 'none':
        groups.append({'params': [encoder.bank, *encoder.branch_hidden.parameters(),
                                   *encoder.branch_final.parameters()], 'lr': 1e-3})
    return torch.optim.AdamW(groups, weight_decay=0.01)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_state(directory, fingerprint, model, optimizer, epoch, batch, history, best, totals, elapsed):
    atomic_torch(directory / 'resume.pt', {'fingerprint': fingerprint, 'model': cpu_state(model),
        'optimizer': optimizer.state_dict(), 'rng': rng_state(), 'epoch': epoch,
        'batch': batch, 'history': history, 'best': best, 'totals': totals, 'elapsed_seconds': elapsed})
    atomic_json(directory / 'history.json', history)


def load_state(directory, fingerprint, model, optimizer):
    path = directory / 'resume.pt'
    if not path.exists():
        if (directory / 'history.json').exists():
            raise ValueError('History exists without checkpoint')
        return 0, 0, [], {'auc': -1.0, 'epoch': 0}, {}, 0.0
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['fingerprint'] != fingerprint:
        raise ValueError('Resume fingerprint mismatch')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return (state['epoch'], state['batch'], state['history'], state['best'],
            state['totals'], state['elapsed_seconds'])


def normalized_batch(cache, indices, mean, std, device):
    values = np.array(cache.signals[indices], copy=True)
    values -= mean
    values /= std
    return torch.from_numpy(values).to(device, non_blocking=True)


@torch.inference_mode()
def predict_development(model, data, device):
    model.eval()
    logits = []
    for start in range(len(data['full']), len(data['full']) + len(data['development']), BATCH):
        stop = min(start + BATCH, len(data['full']) + len(data['development']))
        x = normalized_batch(data['waveforms'], list(range(start, stop)), data['mean'], data['std'], device)
        logits.extend(model(x).cpu().numpy().tolist())
    return np.asarray(logits, dtype=np.float64)


def development_screen(rows, logits):
    labels = np.asarray([int(r['target']) for r in rows])
    groups = np.asarray([r['patient_id'] for r in rows])
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    auc = float(roc_auc_score(labels, probabilities))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    counts = {'tp': 0, 'fn': 0, 'tn': 0, 'fp': 0}
    for train, held in splitter.split(logits, labels, groups):
        threshold = select_threshold(labels[train], probabilities[train], 0.95)
        pred = probabilities[held] >= threshold
        counts['tp'] += int(np.sum((labels[held] == 1) & pred))
        counts['fn'] += int(np.sum((labels[held] == 1) & ~pred))
        counts['tn'] += int(np.sum((labels[held] == 0) & ~pred))
        counts['fp'] += int(np.sum((labels[held] == 0) & pred))
    return {'auroc': auc, 'fold_sensitivity': counts['tp'] / (counts['tp'] + counts['fn']),
            'fold_specificity': counts['tn'] / (counts['tn'] + counts['fp']),
            'fold_confusion': counts}


def identity(data, budget, kind):
    return {'provenance': data['provenance'], 'budget': budget, 'arm': kind,
            'train_ids': [r['ecg_id'] for r in data['full']],
            'development_ids': [r['ecg_id'] for r in data['development']],
            'template_selection': data['template_receipt']}


def run_arm(args, data, budget, kind, directory, max_epochs, deadline):
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = digest_json(identity(data, budget, kind))
    completion = directory / 'completion.json'
    if completion.exists():
        done = json.loads(completion.read_text())
        if done['fingerprint'] != fingerprint or digest_file(directory / 'history.json') != done['history_sha256'] or digest_file(directory / 'best_model.pt') != done['best_model_sha256']:
            raise ValueError('Completed arm fingerprint or artifact mismatch')
        return json.loads((directory / 'history.json').read_text())
    config = directory / 'config.json'
    if config.exists() and json.loads(config.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Existing output uses different inputs or code')
    model = make_model(args, kind, data['bank'])
    optimizer = optimizer_for(model)
    epoch, batch_pos, history, best, totals, elapsed = load_state(directory, fingerprint, model, optimizer)
    atomic_json(config, {'fingerprint': fingerprint, 'identity': identity(data, budget, kind),
                         'parameter_count': sum(p.numel() for p in model.parameters()),
                         'exposed_labels': 15360 if budget == '1' else 1518})
    limited_ids = {r['ecg_id'] for r in data['limited']}
    exposed = np.asarray([budget == '1' or r['ecg_id'] in limited_ids for r in data['full']], dtype=bool)
    targets = np.asarray([float(r['target']) if mask else 0.0 for r, mask in zip(data['full'], exposed)], dtype=np.float32)
    if int(exposed.sum()) != (15360 if budget == '1' else 1518):
        raise ValueError('Exposed label count changed')
    started = time.monotonic()
    while epoch < max_epochs:
        model.train()
        batches = fixed_batches(len(data['full']), exposed, epoch)
        for position in range(batch_pos, len(batches)):
            indices = batches[position]
            x = normalized_batch(data['waveforms'], indices, data['mean'], data['std'], args.device)
            y = torch.from_numpy(targets[indices]).to(args.device)
            mask = torch.from_numpy(exposed[indices]).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits[mask], y[mask])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient):
                raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            totals['updates'] = totals.get('updates', 0) + 1
            totals['record_exposures'] = totals.get('record_exposures', 0) + len(indices)
            totals['label_exposures'] = totals.get('label_exposures', 0) + int(mask.sum())
            totals['loss_sum'] = totals.get('loss_sum', 0.0) + float(loss.detach())
            batch_pos = position + 1
            expired = deadline is not None and time.monotonic() >= deadline
            if batch_pos % SAVE_EVERY == 0 or STOP or expired:
                save_state(directory, fingerprint, model, optimizer, epoch, batch_pos,
                           history, best, totals, elapsed + time.monotonic() - started)
            if STOP or expired:
                raise SystemExit(75)
        logits = predict_development(model, data, args.device)
        screen = development_screen(data['development'], logits)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite model weights')
        if screen['auroc'] > best['auc']:
            best = {'auc': screen['auroc'], 'epoch': epoch + 1}
            atomic_torch(directory / 'best_model.pt', {'fingerprint': fingerprint,
                         'epoch': epoch + 1, 'model': cpu_state(model)})
        row = {'epoch': epoch + 1, 'budget': budget, 'arm': kind,
               'mean_batch_loss': totals['loss_sum'] / totals['updates'],
               'optimizer_updates': totals['updates'], 'record_exposures': totals['record_exposures'],
               'label_exposures': totals['label_exposures'], 'development': screen,
               'best_epoch': best['epoch'], 'elapsed_seconds': elapsed + time.monotonic() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        epoch += 1
        batch_pos, totals = 0, {}
        save_state(directory, fingerprint, model, optimizer, epoch, 0, history, best, totals,
                   elapsed + time.monotonic() - started)
    if epoch == EPOCHS and max_epochs == EPOCHS:
        atomic_json(completion, {'fingerprint': fingerprint, 'best_epoch': best['epoch'],
            'best_development_auroc': best['auc'],
            'history_sha256': digest_file(directory / 'history.json'),
            'best_model_sha256': digest_file(directory / 'best_model.pt')})
    return history


def verify_roundtrip(args, data, kind, directory):
    original = torch.load(directory / 'resume.pt', map_location='cpu', weights_only=False)
    probe = make_model(args, kind, data['bank'])
    opt = optimizer_for(probe)
    epoch, batch, *_ = load_state(directory, digest_json(identity(data, '1', kind)), probe, opt)
    if (epoch, batch) != (1, 0):
        raise ValueError('Profile did not complete one epoch')
    if any(not torch.equal(value, probe.state_dict()[key]) for key, value in original['model'].items()):
        raise ValueError('Profile model roundtrip differs')
    for key, state in original['optimizer']['state'].items():
        for field, value in state.items():
            other = opt.state_dict()['state'][key][field]
            if torch.is_tensor(value) and not torch.equal(value, other):
                raise ValueError('Profile optimizer roundtrip differs')
    saved_rng = original['rng']
    active_rng = rng_state()
    if (active_rng['python'] != saved_rng['python'] or
            active_rng['numpy'][0] != saved_rng['numpy'][0] or
            not np.array_equal(active_rng['numpy'][1], saved_rng['numpy'][1]) or
            active_rng['numpy'][2:] != saved_rng['numpy'][2:] or
            not torch.equal(active_rng['torch'], saved_rng['torch']) or
            len(active_rng['cuda']) != len(saved_rng['cuda']) or
            any(not torch.equal(a, b) for a, b in zip(active_rng['cuda'], saved_rng['cuda']))):
        raise ValueError('Profile RNG roundtrip differs')
    return {'epoch': epoch, 'batch': batch, 'model_tensors': len(original['model']),
            'optimizer_slots': len(original['optimizer']['state']), 'rng_roundtrip': True}


def report_pilot(output):
    selected = {}
    lines = ['# Experiment 017 development-only morphology pilot', '',
             'Five fixed epochs per arm. No calibration or test predictions were used.', '',
             '| Label budget | Arm | Best epoch | Development AUROC | Patient-fold specificity | Patient-fold sensitivity | Updates | Labeled exposures | Wall seconds |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for budget in ('1', '0.1'):
        for kind in ('none', 'conv', 'template'):
            directory = output / f'{kind}_fraction{budget}_seed42'
            if not (directory / 'completion.json').exists():
                return
            history = json.loads((directory / 'history.json').read_text())
            if len(history) != EPOCHS:
                raise ValueError('Completed arm lacks five epochs')
            epoch = json.loads((directory / 'completion.json').read_text())['best_epoch']
            row = history[epoch - 1]
            selected[(budget, kind)] = row['development']
            dev = row['development']
            lines.append(f"| {budget} | {kind} | {epoch} | {dev['auroc']:.4f} | {dev['fold_specificity']:.4f} | {dev['fold_sensitivity']:.4f} | {sum(r['optimizer_updates'] for r in history)} | {sum(r['label_exposures'] for r in history)} | {history[-1]['elapsed_seconds']:.1f} |")
    lines.extend(['', '## Prespecified comparison', ''])
    advance = False
    safe = True
    for budget in ('1', '0.1'):
        template = selected[(budget, 'template')]
        conv = selected[(budget, 'conv')]
        plain = selected[(budget, 'none')]
        gain_conv = template['auroc'] - conv['auroc']
        gain_plain = template['auroc'] - plain['auroc']
        sens_delta = template['fold_sensitivity'] - conv['fold_sensitivity']
        passed = gain_conv >= 0.002 and gain_plain >= 0.002 and sens_delta >= -0.005
        advance |= passed
        safe &= gain_conv >= -0.002 and sens_delta >= -0.005
        lines.append(f"Budget {budget}: template minus convolution AUROC {gain_conv:+.4f}, template minus no-branch AUROC {gain_plain:+.4f}, template minus convolution patient-fold sensitivity {sens_delta:+.4f}; positive criteria {'met' if passed else 'not met'}.")
    advance &= safe
    lines.extend(['', 'A second matched seed is warranted before calibration or test.' if advance else
                  'The prespecified development screen did not pass; retain results as exploratory evidence.', '',
                  'Template matches are not validated explanations. The binary endpoint is an ECG diagnostic annotation proxy, not verified health or referral need.', ''])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.md').write_text('\n'.join(lines))
    atomic_json(output / 'completion.json', {'status': 'development_pilot_complete',
        'report_sha256': digest_file(output / 'report.md'), 'advance_to_second_seed': bool(advance),
        'arm_completions_sha256': {f'{kind}_fraction{budget}_seed42': digest_file(output / f'{kind}_fraction{budget}_seed42/completion.json')
            for budget in ('1', '0.1') for kind in ('none', 'conv', 'template')}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'profile', 'train'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--cache-dir', type=Path, default=CACHE)
    parser.add_argument('--manifest-dir', type=Path, default=MANIFEST)
    parser.add_argument('--ssl', type=Path, default=SSL)
    parser.add_argument('--normalization', type=Path, default=NORMALIZATION)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--pool-verification', type=Path, default=POOL_VERIFICATION)
    parser.add_argument('--max-cache-bytes', type=int, default=2_400_000_000)
    parser.add_argument('--reserve-bytes', type=int, default=1_000_000_000)
    parser.add_argument('--max-wall-seconds', type=int, default=7200)
    args = parser.parse_args()
    if args.threads < 1 or args.max_cache_bytes < 2_100_000_000 or args.reserve_bytes < 0 or args.max_wall_seconds < 1:
        parser.error('Invalid thread, memory, or time limits')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if args.stage == 'profile' and args.device != 'cuda':
        parser.error('Profile must measure the real V100 GPU data path')
    torch.set_num_threads(args.threads)
    signal.signal(signal.SIGTERM, stopping)
    deadline = time.monotonic() + args.max_wall_seconds
    with LOCK.open('a+') as lock:
        if args.device == 'cuda':
            print(f'Waiting for GPU lock {LOCK}', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        data = load_inputs(args)
        if args.stage == 'check':
            args.output_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output_dir / 'provenance/verification.json', {
                'stage': 'check', 'fingerprint': digest_json(data['provenance']),
                'provenance': data['provenance'],
                'source_015_receipt_sha256_for_audit': data['pool_receipt_hash'],
                'source_015_receipt': str(args.pool_verification.resolve())})
            print(json.dumps({'stage': 'check', 'train': len(data['full']), 'limited': len(data['limited']),
                'development': len(data['development']), 'calibration': len(data['calibration']),
                'test': len(data['test']), 'precheck_seconds': data['precheck_seconds'],
                'fingerprint': digest_json(data['provenance'])}), flush=True)
            return
        if args.device == 'cuda':
            check_path = args.output_dir / 'provenance/verification.json'
            if not check_path.exists() or json.loads(check_path.read_text()).get('fingerprint') != digest_json(data['provenance']):
                raise ValueError('Verified Experiment 017 CPU check is missing or changed')
        print(json.dumps({'stage': args.stage, 'precheck_seconds': data['precheck_seconds'],
            'preload_seconds': data['preload_seconds'], 'cache_bytes': int(data['waveforms'].signals.nbytes),
            'template_selection': data['template_receipt']}), flush=True)
        if args.stage == 'profile':
            with tempfile.TemporaryDirectory(prefix='experiment017_profile_') as temporary:
                durations, roundtrips = {}, {}
                for kind in ('none', 'conv', 'template'):
                    start = time.monotonic()
                    directory = Path(temporary) / kind
                    run_arm(args, data, '1', kind, directory, 1, deadline)
                    roundtrips[kind] = verify_roundtrip(args, data, kind, directory)
                    durations[kind] = time.monotonic() - start
                training_estimate = (data['precheck_seconds'] + data['preload_seconds'] +
                                     2 * EPOCHS * sum(durations.values()))
                estimate = data['precheck_seconds'] + data['preload_seconds'] + sum(durations.values()) + training_estimate
                receipt = {'stage': 'profile', 'device': args.device, 'full_epoch_seconds': durations,
                    'checkpoint_roundtrips': roundtrips,
                    'precheck_seconds': data['precheck_seconds'], 'preload_seconds': data['preload_seconds'],
                    'projected_six_run_training_seconds': training_estimate,
                    'conservative_profile_plus_six_run_seconds': estimate,
                    'planning_gate_seconds': 7200,
                    'gate_passed': estimate <= 7200,
                    'peak_gpu_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                    'fingerprint': digest_json(data['provenance'])}
                args.output_dir.mkdir(parents=True, exist_ok=True)
                atomic_json(args.output_dir / 'profile.json', receipt)
                print(json.dumps(receipt), flush=True)
            return
        if args.device == 'cuda':
            path = args.output_dir / 'profile.json'
            if not path.exists():
                raise ValueError('Real complete-pass GPU profile required before training')
            profile = json.loads(path.read_text())
            if (profile.get('stage') != 'profile' or profile.get('device') != 'cuda' or
                    profile['fingerprint'] != digest_json(data['provenance']) or not profile['gate_passed']):
                raise ValueError('GPU profile fingerprint or two-hour planning gate failed')
        for budget in ('1', '0.1'):
            for kind in ('none', 'conv', 'template'):
                if time.monotonic() >= deadline:
                    raise SystemExit(75)
                run_arm(args, data, budget, kind,
                        args.output_dir / f'{kind}_fraction{budget}_seed42', EPOCHS, deadline)
        report_pilot(args.output_dir)


if __name__ == '__main__':
    main()
