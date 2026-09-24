"""CPU-only real-data loader/backpropagation check; not a model experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ecg_experiment.training_dataset import TrainingECGDataset, file_hash, read_csv


def smoke(directory):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    ssl = TrainingECGDataset(directory)
    representatives = {}
    for index, row in enumerate(ssl.rows):
        representatives.setdefault(row['source'], index)
    batch = next(iter(DataLoader(Subset(ssl, list(representatives.values())), batch_size=len(representatives))))
    assert tuple(batch['signal'].shape) == (len(representatives), 12, 5000)
    assert not batch['target_available'].any() and torch.all(batch['target'] == -1)
    labeled = TrainingECGDataset(directory, purpose='supervised', label_budget='0.1')
    batch = next(iter(DataLoader(labeled, batch_size=8)))
    assert batch['target_available'].all() and set(batch['source']) == {'ptbxl'}
    model = torch.nn.Sequential(torch.nn.Conv1d(12, 4, 25, stride=25), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(4,1))
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch['signal']).squeeze(1), batch['target'].float())
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert any(not torch.equal(a,b) for a,b in zip(before, model.parameters()))
    patients = {r['patient_id'] for r in ssl.rows if r['source']=='ptbxl'}
    references = read_csv(Path(directory)/'heldout_references.csv')
    reference_sets = {split: {r['patient_id'] for r in references if r['split']==split}
                      for split in ('development','calibration','test')}
    for split, identities in reference_sets.items():
        assert not patients & identities
        for other, others in reference_sets.items():
            if split != other:
                assert not identities & others
    return {'status':'passed_cpu_loader_smoke_not_performance_result', 'device':'cpu',
            'dataset_metadata_sha256':file_hash(Path(directory)/'metadata.json'),
            'smoke_source_sha256':file_hash(Path(__file__)), 'ssl_sources_checked':list(representatives),
            'ssl_records':len(ssl), 'ssl_targets_masked':True, 'supervised_fraction0.1_records':len(labeled),
            'supervised_fraction1_records':len(TrainingECGDataset(directory,'supervised','1')),
            'real_batch_shape':list(batch['signal'].shape), 'finite_backward_and_optimizer_step':True,
            'ptb_train_development_calibration_test_patients_disjoint':True,
            'heldout_reference_records':len(references)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(smoke(args.dataset_dir), indent=2))
