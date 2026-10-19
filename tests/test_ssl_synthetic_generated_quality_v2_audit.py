"""Artificial v2 audit accounting and fixed-figure contract checks."""

import hashlib
import json
import struct

import pytest

from scripts.validation.audit_ssl_synthetic_generated_quality_v2 import (
    PARENT,
    verify_figures,
    verify_resources,
)


def test_fixed_png_dimensions_and_corruption(tmp_path):
    sizes = {
        "fixed_first_eight.png": (2600, 2080),
        "known_support.png": (2860, 2600),
        "quality_diagnostics.png": (1950, 520),
    }
    for name, size in sizes.items():
        (tmp_path / name).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + struct.pack(">II", *size))
    verify_figures(tmp_path)
    (tmp_path / "known_support.png").write_bytes(b"bad")
    with pytest.raises(ValueError, match="PNG signature"):
        verify_figures(tmp_path)


def test_v2_prior_self_cost_ledger_is_counted_once(tmp_path):
    root = tmp_path
    study = root / "outputs/ssl_synthetic_localization/generated_quality_v2"
    study.mkdir(parents=True)
    paths = [
        f"{PARENT}/prospective_ledger.json",
        f"{PARENT}/independent_audit_v1/prospective_ledger.json",
        "outputs/ssl_synthetic_localization/generated_quality_v1/prospective_ledger.json",
        "outputs/ssl_synthetic_localization/generated_quality_v1/independent_failure_audit_v1/prospective_ledger.json",
    ]
    entries = []
    for index, name in enumerate(paths):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        prior = {
            "cpu_seconds": index + 1,
            "gpu_wall_seconds": 6 if index == 2 else 0,
            "stage_main_plus_audit_cpu_seconds": 999,
        }
        path.write_text(json.dumps(prior))
        entries.append(
            {
                "stage": str(index),
                "path": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "cpu_seconds": prior["cpu_seconds"],
                "gpu_wall_seconds": prior["gpu_wall_seconds"],
            }
        )
    profile = {
        "cpu_seconds": 0.1,
        "wall_seconds": 0.2,
        "spent_cpu_seconds": 1,
        "spent_wall_seconds": 2,
        "forecast_cpu_seconds": 57,
        "forecast_wall_seconds": 84,
        "forecast_saved_bytes": 300_000_000,
        "peak_allocated_bytes": 45_000_000,
    }
    (study / "profile.json").write_text(json.dumps(profile))
    main = {
        "profile": profile,
        "cpu_seconds": 3,
        "wall_seconds": 4,
        "gpu_lock_wall_seconds": 1,
        "forward_counts": {
            "cpu_predecessor_replay": 4,
            "gpu_profile": 1,
            "gpu_generation": 128,
            "gpu_replay": 1,
        },
        "gpu_metadata": {
            "free_bytes_at_admission": 3 * 1024**3,
            "peak_allocated_bytes": 45_000_000,
            "deterministic": True,
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "torch_threads": 1,
            "interop_threads": 1,
        },
        "amendment_input_sha256": {},
    }
    main["forward_attempts"] = main["forward_counts"]
    (study / "amendment_admission.json").write_text(
        json.dumps({"prior_stage_entries": entries, "amendment_input_sha256": {}})
    )
    main["output_bytes_before_receipt"] = sum(p.stat().st_size for p in study.iterdir())
    (study / "summary.json").write_text(json.dumps(main))
    ledger = {
        "cpu_seconds": 3.1,
        "wall_seconds": 4.1,
        "gpu_wall_seconds": 1,
        "receipt_sha256": hashlib.sha256((study / "summary.json").read_bytes()).hexdigest(),
        "prior_stage_entries": entries,
        "prior_accounting_complete": True,
        "cumulative_cpu_seconds": 13.1,
        "cumulative_gpu_wall_seconds": 7,
    }
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    actual = verify_resources(root, study, main, lambda: None)
    assert actual["prior_cpu_seconds"] == 10
    assert actual["current_machine_cpu_through_main"] == 13.1
    assert actual["current_machine_gpu_wall_seconds"] == 7
    ledger["cumulative_cpu_seconds"] = 999
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="additive CPU ledger"):
        verify_resources(root, study, main, lambda: None)


def test_bindings_keep_amendment_root_separate_from_parent_and_study(tmp_path):
    from scripts.validation.audit_ssl_synthetic_generated_quality_v2 import bindings

    root = tmp_path / "output_host"
    source_root = tmp_path / "source_checkout"
    study = root / "outputs/ssl_synthetic_localization/generated_quality_v2"
    parent = root / PARENT
    study.mkdir(parents=True)
    parent.mkdir(parents=True)
    source_root.mkdir()
    (source_root / "author.py").write_text("source")
    snapshot = study / "source_snapshot/author.py"
    snapshot.parent.mkdir()
    snapshot.write_text("source")
    (parent / "input.json").write_text("parent")
    (study / "result.json").write_text("result")
    amendment = {}
    for name in ("failure.json", "profile.json", "prospective_ledger.json"):
        relative = f"outputs/ssl_synthetic_localization/generated_quality_v1/{name}"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(name)
        amendment[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    main = {
        "source_sha256": {"author.py": hashlib.sha256(snapshot.read_bytes()).hexdigest()},
        "input_sha256": {"input.json": hashlib.sha256((parent / "input.json").read_bytes()).hexdigest()},
        "output_sha256": {"result.json": hashlib.sha256((study / "result.json").read_bytes()).hexdigest()},
        "amendment_input_sha256": amendment,
    }
    assert bindings(root, source_root, study, parent, main, lambda: None) == 7
    (root / next(iter(amendment))).write_text("corrupt")
    with pytest.raises(ValueError, match="amendment receipt binding"):
        bindings(root, source_root, study, parent, main, lambda: None)
