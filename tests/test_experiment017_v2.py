"""Version 2 correctness: CPU profile restore and source fingerprinting."""

from pathlib import Path
from types import SimpleNamespace

import torch

from ecg_experiment.cpc import CPCPretrainer
from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.pilot import Progress, save_state
from ecg_experiment.reproducibility import cpu_state, seed_everything
from scripts.experiments import run_cpc_morphology017 as original
from scripts.experiments import run_cpc_morphology017_v2 as versioned


def test_entry_point_passes_versioned_output_code_and_cpu_roundtrip(monkeypatch):
    observed = {}
    monkeypatch.setattr(original, "main", lambda **kwargs: observed.update(kwargs))
    versioned.main()
    assert observed == {"output": versioned.OUTPUT, "extra_code": versioned.EXTRA_CODE,
                        "roundtrip_device": "cpu"}


def test_wrapper_sources_enter_fingerprint():
    hashes = original.code_hashes(versioned.EXTRA_CODE)
    assert set(original.CODE_PATHS) < set(hashes)
    for name in ("scripts/experiments/run_cpc_morphology017_v2.py", "docs/experiment-017-morphology-v2.md"):
        assert hashes[name] == sha256_file(versioned.ROOT / name)
    assert set(original.code_hashes()) == set(original.CODE_PATHS)


def test_versioned_output_is_the_default_output_dir():
    args = original.parse_args(["--stage", "check", "--device", "cpu"], versioned.OUTPUT)
    assert args.output_dir == versioned.OUTPUT
    default = original.parse_args(["--stage", "check", "--device", "cpu"], original.OUTPUT)
    assert default.output_dir == original.OUTPUT


def test_actual_cpu_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    seed_everything(5)
    ssl = tmp_path / "encoder.pt"
    torch.save({"encoder": cpu_state(CPCPretrainer().encoder)}, ssl)
    bank = torch.zeros(32, 12, 50)
    model = original.make_model(ssl, "template", bank, "cpu")
    optimizer = original.optimizer_for(model)
    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(2, 12, 2500)).sum().backward()
    optimizer.step()
    data = SimpleNamespace(provenance={"sample": "fixed"},
                           partitions=SimpleNamespace(full=[{"ecg_id": "training"}],
                                                      development=[{"ecg_id": "development"}]),
                           template_receipt=[{"ecg_id": "training", "half": 0, "end_in_half": 49}], bank=bank)
    fingerprint = sha256_json(original.identity(data, "1", "template"))
    directory = Path(tmp_path / "arm")
    directory.mkdir()
    save_state(directory, fingerprint, model, optimizer, Progress(1, 0, [], {"auc": -1.0, "epoch": 0}), 2.0)
    receipt = original.verify_roundtrip(SimpleNamespace(ssl=ssl), data, "template", directory, "cpu")
    assert (receipt["epoch"], receipt["batch"]) == (1, 0)
    assert receipt["rng_roundtrip"]
    assert receipt["model_tensors"] == len(model.state_dict())
