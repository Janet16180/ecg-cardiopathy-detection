"""Version 2 correctness: CPU profile restore and source fingerprinting."""

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.experiments import run_cpc_morphology017 as original
from scripts.experiments import run_cpc_morphology017_v2 as versioned
from scripts.experiments.run_cpc_experiment import digest_file


class VersionedProfileTests(unittest.TestCase):
    def test_cpu_probe_even_for_cuda_profile(self):
        seen = {}
        def fake_verify(args, data, kind, directory):
            seen['device'] = args.device
            seen['kind'] = kind
            return {'rng_roundtrip': True}
        requested = argparse.Namespace(device='cuda', ssl=Path('unused'))
        with patch.object(versioned, 'ORIGINAL_VERIFY_ROUNDTRIP', fake_verify):
            result = versioned.verify_roundtrip(requested, {}, 'template', Path('unused'))
        self.assertEqual(seen, {'device': 'cpu', 'kind': 'template'})
        self.assertEqual(requested.device, 'cuda')
        self.assertTrue(result['rng_roundtrip'])

    def test_wrapper_sources_enter_fingerprint(self):
        data = {'provenance': {'code': {'original.py': 'original-hash'}}}
        with patch.object(versioned, 'ORIGINAL_LOAD_INPUTS', return_value=data):
            loaded = versioned.load_inputs(SimpleNamespace())
        self.assertEqual(loaded['provenance']['code']['original.py'], 'original-hash')
        for name in ('scripts/experiments/run_cpc_morphology017_v2.py',
                     'docs/experiment-017-morphology-v2.md'):
            self.assertEqual(loaded['provenance']['code'][name],
                             digest_file(versioned.ROOT / name))

    def test_entry_point_wires_versioned_functions_and_output(self):
        prior = (original.load_inputs, original.verify_roundtrip, original.OUTPUT)
        observed = {}
        try:
            with patch.object(original, 'main', lambda: observed.update({
                'load': original.load_inputs,
                'verify': original.verify_roundtrip,
                'output': original.OUTPUT,
            })):
                versioned.main()
        finally:
            original.load_inputs, original.verify_roundtrip, original.OUTPUT = prior
        self.assertIs(observed['load'], versioned.load_inputs)
        self.assertIs(observed['verify'], versioned.verify_roundtrip)
        self.assertEqual(observed['output'], versioned.OUTPUT)

    def test_actual_cpu_checkpoint_roundtrip(self):
        # Exercise v1's complete model/optimizer/RNG verification through v2.
        if not original.SSL.exists():
            self.skipTest('Experiment 004 SSL checkpoint unavailable')
        torch.set_num_threads(1)
        bank = torch.zeros(32, 12, 50)
        args = argparse.Namespace(ssl=original.SSL, device='cpu')
        model = original.make_model(args, 'template', bank)
        optimizer = original.optimizer_for(model)
        waveforms = torch.randn(2, 12, 2500)
        optimizer.zero_grad(set_to_none=True)
        model(waveforms).sum().backward()
        optimizer.step()
        data = {'provenance': {'sample': 'fixed'},
                'full': [{'ecg_id': 'training'}],
                'development': [{'ecg_id': 'development'}],
                'template_receipt': [{'ecg_id': 'training', 'half': 0, 'end_in_half': 49}],
                'bank': bank}
        fingerprint = original.digest_json(original.identity(data, '1', 'template'))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original.save_state(directory, fingerprint, model, optimizer, 1, 0, [],
                                {'auc': -1., 'epoch': 0}, {}, 2.0)
            requested = argparse.Namespace(ssl=original.SSL, device='cuda')
            receipt = versioned.verify_roundtrip(requested, data, 'template', directory)
        self.assertEqual((receipt['epoch'], receipt['batch']), (1, 0))
        self.assertTrue(receipt['rng_roundtrip'])


if __name__ == '__main__':
    unittest.main()
