"""CPU correctness checks for Experiment 017's native-grid morphology branches."""

import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from ecg_experiment.cpc import CPCClassifier
from ecg_experiment.cpc_morphology import (
    MorphologyCPCClassifier, local_response, native_windows,
    TEMPLATES, LEADS, SUPPORT, TOKEN_COUNT, WINDOW_ELEMENTS,
)
from scripts.run_cpc_morphology017 import (ROOT, SSL, fixed_batches, make_model,
    template_windows, save_state, load_state, file_identity, verified_pool_hashes)
from scripts.run_cpc_experiment import digest_file


class MorphologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hand_distances_and_alignment(self):
        halves = torch.arange(2 * LEADS * 1250, dtype=torch.float32).reshape(2, LEADS, 1250) / 1000
        bank = torch.arange(TEMPLATES * LEADS * SUPPORT, dtype=torch.float32).reshape(TEMPLATES, LEADS, SUPPORT) / 10000
        distance = local_response(halves, bank, 'template')
        dot = local_response(halves, bank, 'conv')
        self.assertEqual(tuple(distance.shape), (2, TEMPLATES, TOKEN_COUNT))
        for half, template, token in ((0, 0, 0), (0, 13, 3), (1, 31, 78)):
            edge = 16 * token
            window = F.pad(halves[half:half+1], (SUPPORT-1, 0))[0, :, edge:edge+SUPPORT]
            expected_distance = -((window - bank[template]).square().sum() / WINDOW_ELEMENTS)
            expected_dot = 2 * (window * bank[template]).sum() / WINDOW_ELEMENTS
            torch.testing.assert_close(distance[half, template, token], expected_distance,
                                       rtol=2e-5, atol=1e-5)
            torch.testing.assert_close(dot[half, template, token], expected_dot,
                                       rtol=2e-5, atol=1e-5)
        self.assertEqual(int(16 * (TOKEN_COUNT - 1)), 1248)

    def test_half_boundary_and_causality(self):
        torch.manual_seed(3)
        bank = torch.randn(TEMPLATES, LEADS, SUPPORT) * 0.05
        signal = torch.randn(1, LEADS, 2500) * 0.1
        halves = native_windows(signal)
        original = local_response(halves, bank, 'template')
        changed = signal.clone()
        changed[:, :, 1250:] += 100
        response = local_response(native_windows(changed), bank, 'template')
        torch.testing.assert_close(original[0], response[0], rtol=0, atol=0)
        changed = signal.clone()
        changed[:, :, 17:1250] += 50
        response = local_response(native_windows(changed), bank, 'template')
        torch.testing.assert_close(original[0, :, :2], response[0, :, :2], rtol=0, atol=0)
        self.assertFalse(torch.equal(original[0, :, 2], response[0, :, 2]))
        model = MorphologyCPCClassifier('template', bank).eval()
        with torch.no_grad():
            model.encoder.branch_final.weight.fill_(0.001)
            model.encoder.branch_final.bias.zero_()
            z0, c0 = model.encoder(signal)
            after = signal.clone()
            after[:, :, 161:1250] += 1
            z1, c1 = model.encoder(after)
        torch.testing.assert_close(z0[:, 0, :11], z1[:, 0, :11], rtol=0, atol=0)
        torch.testing.assert_close(c0[:, 0, :11], c1[:, 0, :11], rtol=0, atol=0)
        torch.testing.assert_close(z0[:, 1], z1[:, 1], rtol=0, atol=0)

    def test_exact_bootstrap_and_matching_parameters(self):
        if not SSL.exists():
            self.skipTest('Experiment 004 SSL checkpoint unavailable')
        torch.manual_seed(4)
        bank = torch.randn(TEMPLATES, LEADS, SUPPORT) * 0.05
        args = SimpleNamespace(ssl=SSL, device='cpu')
        models = {kind: make_model(args, kind, bank).eval()
                  for kind in ('none', 'conv', 'template')}
        self.assertEqual(sum(p.numel() for p in models['conv'].parameters()),
                         sum(p.numel() for p in models['template'].parameters()))
        for key in ('bank', 'branch_hidden.weight', 'branch_hidden.bias',
                    'branch_final.weight', 'branch_final.bias'):
            torch.testing.assert_close(models['conv'].encoder.state_dict()[key],
                                       models['template'].encoder.state_dict()[key], rtol=0, atol=0)
        for kind in ('conv', 'template'):
            torch.testing.assert_close(models['none'].head.weight, models[kind].head.weight, rtol=0, atol=0)
            torch.testing.assert_close(models['none'].head.bias, models[kind].head.bias, rtol=0, atol=0)
        waveforms = torch.randn(2, LEADS, 2500) * 0.1
        with torch.inference_mode():
            baseline = models['none'](waveforms)
            for kind in ('conv', 'template'):
                torch.testing.assert_close(models[kind](waveforms), baseline, rtol=0, atol=0)
                z, c = models[kind].encoder(waveforms)
                z0, c0 = models['none'].encoder(waveforms)
                torch.testing.assert_close(z, z0, rtol=0, atol=0)
                torch.testing.assert_close(c, c0, rtol=0, atol=0)
        # Explicitly compare the no-branch arm with original CPCClassifier.
        from scripts.run_cpc_experiment import seed_all
        seed_all(42)
        reference = CPCClassifier().eval()
        saved = torch.load(SSL, map_location='cpu', weights_only=True)
        reference.encoder.load_state_dict(saved['encoder'])
        with torch.inference_mode():
            torch.testing.assert_close(models['none'](waveforms), reference(waveforms), rtol=0, atol=0)

    def test_branch_learns_after_zero_start(self):
        torch.manual_seed(9)
        bank = torch.randn(TEMPLATES, LEADS, SUPPORT) * 0.05
        model = MorphologyCPCClassifier('template', bank)
        x = torch.randn(2, LEADS, 2500) * 0.1
        labels = torch.tensor([0., 1.])
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(x), labels).backward()
        self.assertIsNotNone(model.encoder.branch_final.weight.grad)
        self.assertGreater(float(model.encoder.branch_final.weight.grad.abs().sum()), 0)
        opt.step()
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(x), labels).backward()
        self.assertGreater(float(model.encoder.bank.grad.abs().sum()), 0)
        self.assertGreater(float(model.encoder.branch_hidden.weight.grad.abs().sum()), 0)

    def test_training_only_seeded_bank_and_schedule(self):
        class Cache:
            pass
        cache = Cache()
        cache.signals = np.zeros((40, LEADS, 2500), dtype=np.float32)
        cache.rows = [{'ecg_id': str(i)} for i in range(40)]
        for i in range(40):
            cache.signals[i] = i
        a, receipt_a = template_windows(cache, 35, np.zeros((LEADS, 1), np.float32),
                                        np.ones((LEADS, 1), np.float32))
        b, receipt_b = template_windows(cache, 35, np.zeros((LEADS, 1), np.float32),
                                        np.ones((LEADS, 1), np.float32))
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(receipt_a, receipt_b)
        self.assertTrue(all(int(item['ecg_id']) < 35 for item in receipt_a))
        exposed = np.zeros(15360, dtype=bool)
        exposed[:1518] = True
        batches = fixed_batches(15360, exposed, 0)
        self.assertEqual(len(batches), 120)
        self.assertEqual(len({i for batch in batches for i in batch}), 15360)
        self.assertTrue(all(exposed[batch].any() for batch in batches))

    def test_resume_state_and_fingerprint(self):
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.AdamW(model.parameters())
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        opt.step()
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_state(directory, 'fixed-fingerprint', model, opt, 0, 20, [],
                       {'auc': -1, 'epoch': 0}, {'updates': 20}, 4.0)
            with torch.no_grad():
                model.weight.zero_()
            epoch, batch, _, _, totals, elapsed = load_state(
                directory, 'fixed-fingerprint', model, opt)
            self.assertEqual((epoch, batch, totals['updates'], elapsed), (0, 20, 20, 4.0))
            for key, value in saved.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                load_state(directory, 'wrong', model, opt)

    def test_verified_pool_receipt_reuse_and_mutation_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = ('signals.npy', 'rows.csv', 'ecg_ids.npy')
            keys = ('signals_sha256', 'rows_sha256', 'ecg_ids_sha256')
            for name in names:
                (directory / name).write_bytes(name.encode())
            hashes = {name: digest_file(directory / name) for name in names}
            pool = SimpleNamespace(metadata=dict(zip(keys, hashes.values())))
            receipt = {'stage': 'check',
                       'pool_file_stats': {name: file_identity(directory / name) for name in names},
                       'provenance': {'pool_content_sha256': hashes,
                         'code': {'scripts/run_jepa_cpc_distillation.py':
                                  digest_file(ROOT / 'scripts/run_jepa_cpc_distillation.py')}}}
            from scripts.run_cpc_morphology017 import digest_json
            receipt['fingerprint'] = digest_json(receipt['provenance'])
            receipt_path = directory / 'receipt.json'
            receipt_path.write_text(json.dumps(receipt))
            args = SimpleNamespace(cache_dir=directory, pool_verification=receipt_path)
            verified, stats, _, audit_hash = verified_pool_hashes(args, pool)
            self.assertEqual(verified, hashes)
            self.assertEqual(stats, receipt['pool_file_stats'])
            receipt['checked_at_utc'] = 'a later check'
            receipt_path.write_text(json.dumps(receipt))
            self.assertNotEqual(audit_hash, verified_pool_hashes(args, pool)[3])
            (directory / 'signals.npy').write_bytes(b'mutated')
            with self.assertRaisesRegex(ValueError, 'differs'):
                verified_pool_hashes(args, pool)


if __name__ == '__main__':
    unittest.main()
