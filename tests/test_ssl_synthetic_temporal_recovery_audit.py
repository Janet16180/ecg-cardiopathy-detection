"""Artificial recovery-byte, roundoff, binding-root and once-cost checks."""

import copy
import json

import numpy as np
import pytest

from ecg_experiment.ssl_synthetic_temporal_recovery_saved_audit import row_admission


def raw_rows():
    rows = [
        {
            "fit_id": fit,
            "arm": arm,
            "index": index,
            "seed": 2026103000 + index,
            "mode": mode,
            "active_leads": [0],
            "active_lead_AP": [0.2],
            "active_lead_temporal_AP": 0.2,
            "lead_presence_AP": 0.3,
        }
        for fit in (1, 2, 3)
        for arm in ("B", "P", "K", "PK")
        for index in range(2000, 2064)
        for mode in ("focal", "repeated")
    ]
    lookup = {(row["fit_id"], row["arm"], row["index"], row["mode"]): row for row in rows}
    eps = np.finfo(np.float64).eps
    lookup[(1, "P", 2004, "focal")].update(
        active_leads=[8], active_lead_AP=[1 + eps], active_lead_temporal_AP=1 + eps, lead_presence_AP=1.0
    )
    for identity, values in (
        ((1, "P", 2046, "focal"), [1 + eps, 1 + 2 * eps, 0.2, 1 + eps, 0.2]),
        ((2, "PK", 2028, "focal"), [1 + eps, 0.2, 0.3, 0.4, 0.5]),
    ):
        lookup[identity].update(
            active_leads=[0, 2, 3, 4, 5],
            active_lead_AP=values,
            active_lead_temporal_AP=float(np.mean(values)),
        )
    return rows


def test_all_five_individual_excursions_preserved_and_outside_envelope_rejected():
    rows = raw_rows()
    before = copy.deepcopy(rows)
    result = row_admission(rows)
    assert rows == before
    assert len(result["strict_per_lead_outliers"]) == 5
    assert result["strict_range_offending_rows"] == 1
    assert result["tolerance"] == 8 * np.finfo(np.float64).eps
    assert not result["raw_values_changed"]
    rows[0]["active_lead_AP"] = [1 + 9 * np.finfo(np.float64).eps, 0.1]
    rows[0]["active_leads"] = [0, 1]
    rows[0]["active_lead_temporal_AP"] = float(np.mean(rows[0]["active_lead_AP"]))
    with pytest.raises(ValueError, match="per-lead roundoff envelope"):
        row_admission(rows)
    clipped = raw_rows()
    selected = next(
        row
        for row in clipped
        if (row["fit_id"], row["arm"], row["index"], row["mode"]) == (1, "P", 2004, "focal")
    )
    selected.update(active_lead_AP=[1.0], active_lead_temporal_AP=1.0)
    with pytest.raises(ValueError, match="single-ULP endpoint"):
        row_admission(clipped)


def test_nested_binding_roots_do_not_mix_parent_and_failed_attempt(tmp_path):
    from ecg_experiment.synthetic_audit_io import file_sha256
    from scripts.validation.audit_ssl_synthetic_temporal_recovery_v2 import INPUT_NAMES, bindings

    source, study, parent, failed = (tmp_path / name for name in ("source", "new", "B", "C1"))
    for directory in (source, study / "source_snapshot"):
        directory.mkdir(parents=True)
        (directory / "source.py").write_text("frozen")

    def files(directory, names):
        result = {}
        for name in names:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(directory) + name)
            result[name] = file_sha256(path)
        return result

    allowed = files(parent, INPUT_NAMES)
    recovery_names = (
        "cases.jsonl",
        "predecessor_closure.json",
        "failure.json",
        "prospective_ledger.json",
        "independent_failure_audit_v1/summary.json",
        "independent_failure_audit_v1/prospective_ledger.json",
    )
    recovered = files(failed, recovery_names)
    (study / "roundoff_admission.json").write_text("raw evidence")
    main = {
        "source_sha256": {"source.py": file_sha256(source / "source.py")},
        "input_sha256": {"parent": allowed, "recovery": recovered},
        "output_sha256": {"roundoff_admission.json": file_sha256(study / "roundoff_admission.json")},
    }
    config = {"allowed_parent_files": list(INPUT_NAMES), "recovery_inputs": list(recovery_names)}
    admission = {"parent": {"sha256": allowed}, "recovery": {"sha256": recovered}}
    assert bindings(source, study, parent, failed, main, config, admission, lambda: None) == 38
    (failed / "cases.jsonl").write_text("clipped values")
    with pytest.raises(ValueError, match="retained failed-attempt binding"):
        bindings(source, study, parent, failed, main, config, admission, lambda: None)


def test_failed_main_not_double_added_after_narrow_audit_anchor(tmp_path, monkeypatch):
    from scripts.validation import audit_ssl_synthetic_temporal_recovery_v2 as audit

    study, failed = tmp_path / "C2", tmp_path / "C1"
    study.mkdir()
    prior = failed / "independent_failure_audit_v1/prospective_ledger.json"
    prior.parent.mkdir(parents=True)
    cpu, gpu = 410.97305985599996, 447.98139213304967
    prior.write_text(
        json.dumps(
            {
                "current_machine_cumulative_cpu_seconds": cpu,
                "current_machine_cumulative_gpu_wall_seconds": gpu,
            }
        )
    )
    old_hash = audit.file_sha256

    def identity(path, guard=lambda: None):
        return (
            "4ee0f57ef15ee12c63368ffd7d1eb35bd6cd4e106c6d272e965f776f9fed67a8"
            if path == prior
            else old_hash(path, guard)
        )

    monkeypatch.setattr(audit, "file_sha256", identity)
    main = {"cpu_seconds": 2}
    (study / "summary.json").write_text(json.dumps(main))
    ledger = {
        "receipt": "summary.json",
        "receipt_sha256": old_hash(study / "summary.json"),
        "prior_ledger_sha256": identity(prior),
        "cpu_seconds": 2.1,
        "wall_seconds": 3,
        "gpu_wall_seconds": 0.0,
        "prior_current_machine_cpu_seconds": cpu,
        "prior_current_machine_gpu_wall_seconds": gpu,
        "current_machine_cumulative_cpu_seconds": cpu + 2.1,
        "current_machine_cumulative_gpu_wall_seconds": gpu,
    }
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    admission = {
        "prior_ledger": {
            "sha256": identity(prior),
            "current_machine_cumulative_cpu_seconds": cpu,
            "current_machine_cumulative_gpu_wall_seconds": gpu,
        }
    }
    assert audit.verified_accounting(study, failed, main, admission, lambda: None)["main_cpu_seconds"] == 2.1
    ledger["current_machine_cumulative_cpu_seconds"] += 20.793865505
    (study / "prospective_ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="once-only cumulative addition"):
        audit.verified_accounting(study, failed, main, admission, lambda: None)
