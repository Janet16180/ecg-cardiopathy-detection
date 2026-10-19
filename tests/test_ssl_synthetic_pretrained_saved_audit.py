"""Artificial independent generator-output audit checks."""

import numpy as np
import pytest
import torch

from ecg_experiment.ssl_synthetic_pretrained_saved_audit import (
    SEEDS,
    checkpoint_digest,
    noise_for_seed,
    verify_case,
    verify_noise,
)


def fake_case():
    raw = np.tile(np.linspace(-0.9, 0.9, 5000, dtype=np.float32), (8, 1))
    export = (torch.from_numpy(raw) * 6000).to(torch.int32).T.numpy()
    values = export.T.astype(np.float32)
    canonical = np.stack(
        [
            values[0],
            values[1],
            values[1] - values[0],
            -(values[0] + values[1]) / 2,
            values[0] - values[1] / 2,
            values[1] - values[0] / 2,
            *values[2:],
        ]
    )
    return raw, export, canonical


def test_noise_independently_reproduces_uniform():
    bank = []
    for seed in SEEDS:
        original = (
            torch.empty((1, 8, 5000), dtype=torch.float32)
            .uniform_(-1, 1, generator=torch.Generator(device="cpu").manual_seed(seed))
            .numpy()
        )
        assert np.array_equal(original, noise_for_seed(seed))
        bank.append(original)
    verify_noise(np.stack(bank))
    bank[0][0, 0, 0] += np.float32(0.1)
    with pytest.raises(ValueError, match="noise bytes"):
        verify_noise(np.stack(bank))


def test_conversion_algebra_and_corruption():
    raw, export, canonical = fake_case()
    assert all(verify_case(raw, export, canonical).values())
    export[0, 0] += 1
    with pytest.raises(ValueError, match="export arithmetic"):
        verify_case(raw, export, canonical)
    raw, export, canonical = fake_case()
    canonical[2, 12] += 1
    with pytest.raises(ValueError, match="canonical algebra"):
        verify_case(raw, export, canonical)


def test_constant_lead_fails_variation_gate():
    raw, _, _ = fake_case()
    raw[2] = 0
    export = (raw * np.float32(6000)).astype(np.int32).T
    values = export.T.astype(np.float32)
    canonical = np.vstack(
        (
            values[:2],
            values[1] - values[0],
            -(values[0] + values[1]) / 2,
            values[0] - values[1] / 2,
            values[1] - values[0] / 2,
            values[2:],
        )
    )
    assert not verify_case(raw, export, canonical)["every_raw_lead_varies"]


def test_checkpoint_digest_sensitive_and_rejects_nonfinite():
    first, size = checkpoint_digest({"a": torch.ones(3), "b": torch.zeros(2)})
    second, _ = checkpoint_digest({"b": torch.zeros(2), "a": torch.ones(3)})
    assert size == 5
    assert first == second
    changed, _ = checkpoint_digest({"a": torch.zeros(3), "b": torch.zeros(2)})
    assert changed != first
    with pytest.raises(ValueError, match="finite checkpoint"):
        checkpoint_digest({"a": torch.tensor([float("nan")])})


def test_saved_bank_verifies_replay_and_detects_change(tmp_path):
    import hashlib

    from scripts.validation.audit_ssl_synthetic_pretrained_standalone import check_banks

    noises = np.stack([noise_for_seed(seed) for seed in SEEDS])
    np.save(tmp_path / "noise.npy", noises)
    banks, files, checks = [], [], []
    kinds = ("raw", "author_export", "canonical12")
    for index in range(4):
        raw, _, _ = fake_case()
        raw *= np.float32((index + 1) / 4)
        export = (raw * np.float32(6000)).astype(np.int32).T
        values = export.T.astype(np.float32)
        canonical = np.vstack(
            (
                values[:2],
                values[1] - values[0],
                -(values[0] + values[1]) / 2,
                values[0] - values[1] / 2,
                values[1] - values[0] / 2,
                values[2:],
            )
        )
        arrays = (raw, export, canonical)
        case_files = {kind: f"case_{index}_{kind}.npy" for kind in kinds}
        for kind, array in zip(kinds, arrays, strict=True):
            np.save(tmp_path / case_files[kind], array)
        banks.append(arrays)
        files.append(case_files)
        checks.append(verify_case(*arrays))
    replay_files = {kind: f"replay_{kind}.npy" for kind in kinds}
    for kind, array in zip(kinds, banks[0], strict=True):
        np.save(tmp_path / replay_files[kind], array)
    main = {
        "input_array_sha256": hashlib.sha256(noises.tobytes()).hexdigest(),
        "raw_array_sha256": hashlib.sha256(np.stack([b[0] for b in banks]).tobytes()).hexdigest(),
        "case_files": files,
        "case_checks": checks,
        "replay_files": replay_files,
        "completed_seeds": list(SEEDS),
    }
    assert all(check_banks(tmp_path, main, lambda: None).values())
    for kind, array in zip(kinds, banks[1], strict=True):
        np.save(tmp_path / replay_files[kind], array)
    assert not check_banks(tmp_path, main, lambda: None)["exact_replay"]


def test_path_escape_rejected(tmp_path):
    from scripts.validation.audit_ssl_synthetic_pretrained_standalone import local_path

    with pytest.raises(ValueError, match="relative binding"):
        local_path(tmp_path, "../secret")
    with pytest.raises(ValueError, match="relative binding"):
        local_path(tmp_path, "/secret")


def test_resource_ledger_and_forecast_integrity(tmp_path):
    import json

    from scripts.validation.audit_ssl_synthetic_pretrained_standalone import check_resources

    profile = {
        "cpu_seconds": 0.1,
        "wall_seconds": 0.1,
        "spent_cpu_seconds": 1,
        "spent_wall_seconds": 2,
        "forecast_cpu_seconds": 17,
        "forecast_wall_seconds": 18,
    }
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    before = (tmp_path / "profile.json").stat().st_size
    main = {
        "profile": profile,
        "accounting_scope": "prospective_standalone_only",
        "historical_receipts_available": False,
        "prior_cpu_seconds": 0,
        "prior_gpu_wall_seconds": 0,
        "global_gpu_wall_seconds": 0,
        "cumulative_cpu_seconds": 3,
        "cpu_seconds": 3,
        "wall_seconds": 4,
        "study_bytes_before_receipt": before,
    }
    (tmp_path / "summary.json").write_text(json.dumps(main))
    entry = {"receipt": "summary.json", "cpu_seconds": 3.01, "wall_seconds": 4.01}
    ledger = {"entries": [entry], "cpu_seconds": 3.01, "gpu_wall_seconds": 0}
    (tmp_path / "prospective_ledger.json").write_text(json.dumps(ledger))
    assert check_resources(tmp_path, main, {"output_bytes_ceiling": 96 * 1024**2})["main_cpu_seconds"] == 3.01
    profile["forecast_cpu_seconds"] = 16
    main["profile"] = profile
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="profile forecast"):
        check_resources(tmp_path, main, {"output_bytes_ceiling": 96 * 1024**2})


def test_terminal_audit_ledger_charges_failure_and_unknown_prior(tmp_path):
    import json
    import time

    from scripts.validation.audit_ssl_synthetic_pretrained_standalone import write_audit_ledger

    (tmp_path / "failure.json").write_text('{"all_passed":false}')
    write_audit_ledger(
        tmp_path, "failure.json", "main", 3.0, (time.process_time() - 0.2, time.perf_counter() - 0.3)
    )
    value = json.loads((tmp_path / "prospective_ledger.json").read_text())
    assert value["cpu_seconds"] >= 0.2
    assert value["cumulative_cpu_seconds"] == 3 + value["cpu_seconds"]
    assert value["entries"][0]["receipt"] == "failure.json"
    assert value["gpu_wall_seconds"] == 0
