"""Artificial-only checks for complete selection, physical perturbations and protected data access."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import wfdb

from ecg_experiment import ssl_synthetic_real_background as real
from ecg_experiment import ssl_synthetic_real_background_preparation as prep
from ecg_experiment.files import sha256_file
from ecg_experiment.resample import resample_full
from ecg_experiment.ssl_synthetic_generator import couple_limb_leads
from ecg_experiment.waveforms import LEADS


@pytest.fixture
def config():
    return json.loads(Path("configs/ssl_synthetic_real_background_v1.json").read_bytes())


def record(patient, ecg, role):
    return {
        "patient_id": str(patient),
        "ecg_id": str(ecg),
        "role": role,
        "filename_hr": "records500/artificial",
    }


def audit(rows, excluded=()):
    return [
        dict(
            row,
            train_optimization_eligible=row["ecg_id"] not in excluded,
            quality_exclusions=["flat_segment"] if row["ecg_id"] in excluded else [],
            review_flags=[],
            payload_sha256={},
        )
        for row in rows
    ]


def study_row(role="calibration", patient="900001"):
    return dict(
        record(patient, 1, "validation"),
        study_role=role,
        cache_index=0 if role == "calibration" else 64,
        subject_index=0 if role == "calibration" else 64,
        role_subject_index=0,
    )


def background():
    time = np.arange(2500) / 250
    signal = np.zeros(2500)
    for center in np.arange(0.6, 9.6, 0.5):
        signal += np.exp(-0.5 * ((time - center) / 0.012) ** 2)
    return (
        signal[None, :] * np.linspace(0.4, 1.5, 12)[:, None] + 0.03 * np.sin(2 * np.pi * time)[None, :]
    ).astype(np.float32)


def test_selection_is_stable_excludes_all_ssl_and_uses_lowest_eligible_train(config):
    rows = [
        record(1, 11, "train"),
        record(1, 1, "train"),
        record(2, 2, "train"),
        record(3, 3, "train"),
        record(4, 4, "train"),
        record(10, 10, "validation"),
        record(10, 100, "validation"),
        record(11, 12, "validation"),
        record(20, 20, "engineering_test"),
    ]
    small = config | {"train_reserved_patients": 3, "calibration_patients": 1, "evaluation_patients": 1}
    selected = real.select_patients(rows, audit(rows, ("1", "10")), {"2"}, small)
    assert selected == real.select_patients(list(reversed(rows)), audit(rows, ("1", "10")), {"2"}, small)
    assert all(row["patient_id"] != "2" for row in selected)
    patient1 = [row for row in selected if row["patient_id"] == "1"]
    assert len(patient1) == 1
    assert patient1[0]["ecg_id"] == "11"
    val = [row for row in selected if row["role"] == "validation"]
    assert [row["cache_index"] for row in val] == [0, 1]
    assert next(row for row in val if row["patient_id"] == "10")["ecg_id"] == "10"
    assert next(row for row in val if row["patient_id"] == "10")["quality_exclusions"] == ["flat_segment"]


@pytest.mark.parametrize(
    ("role", "study"),
    [
        ("train", "train_reserved"),
        ("engineering_test", "evaluation"),
        ("excluded", "calibration"),
        ("validation", "train_reserved"),
    ],
)
def test_reserved_protected_roles_rejected_before_path_or_file_access(role, study, monkeypatch):
    row = dict(record(1, 1, role), study_role=study)
    monkeypatch.setattr(prep, "checked_relative", lambda *_: pytest.fail("Constructed protected path"))
    monkeypatch.setattr(prep, "verify_files", lambda *_: pytest.fail("Opened protected payload"))
    with pytest.raises(ValueError, match="must not open payloads"):
        prep.prepare_background(row, {})


def test_anchor_replay_limits_and_insufficient_record(config):
    source = background()
    before = source.copy()
    found = real.find_anchors(source, config)
    assert found == real.find_anchors(source, config)
    assert 3 <= len(found["anchors"]) <= 17
    assert all(100 <= a <= 2375 for a in found["anchors"])
    assert found["retained_before_cap"] >= len(found["anchors"])
    assert np.array_equal(source, before)
    assert not real.find_anchors(np.zeros_like(source), config)["minimum_anchor_gate_passed"]


@pytest.mark.parametrize("role", ["calibration", "evaluation"])
def test_all_four_cases_replay_control_baseline_truth_and_native_background_preserved(config, role):
    source = background()
    original = source.copy()
    cases = real.cases_for_background(
        study_row(role), [150, 400, 650, 900, 1150, 1400, 1650, 1900, 2150], real.array_sha256(source), config
    )
    pairs = [real.render_case(source, case) for case in cases]
    for case, pair in zip(cases, pairs, strict=True):
        assert not real.qualify_pair(pair, case, config)["failures"]
        assert pair["edited"].dtype == np.float32
        assert pair["edited"].shape == (12, 2500)
        assert np.array_equal(
            pair["support"], np.abs(pair["edited"].astype(float) - pair["baseline"].astype(float)) > 0.005
        )
        assert all(
            pair[key].tobytes() == real.render_case(source, case)[key].tobytes()
            for key in ("baseline", "edited", "support", "candidate")
        )
    for index in (0, 1):
        assert (
            pairs[index]["baseline"].tobytes()
            == pairs[3 if cases[index]["nuisance"] else 2]["baseline"].tobytes()
        )
    for index in (2, 3):
        assert pairs[index]["baseline"].tobytes() == pairs[index]["edited"].tobytes()
    assert np.array_equal(source, original)
    assert np.max(np.abs(source[1] - source[0] - source[2])) > 0.05


def test_final_cast_formula_and_no_rounded_baseline_reedit(config):
    source = background() + np.float32(0.12345)
    cases = real.cases_for_background(
        study_row(patient="900002"), [150, 400, 650, 900], real.array_sha256(source), config
    )
    case = cases[1]
    case["nuisance"] = True
    pair = real.render_case(source, case)
    params, rng = real._nuisance_parameters(case["seed"])
    time = np.arange(2500) / 250
    noise = rng.normal(0, params["noise_sd_mv"], (8, 2500))
    noise += params["drift_amplitude_mv"] * np.sin(
        2 * np.pi * params["drift_hz"] * time[None, :] + np.array(params["phases"])[:, None]
    )
    disturbance = couple_limb_leads(noise)
    analytic = pair["analytic_delta"]
    expected = (params["gain"] * (source.astype(float) + analytic / params["gain"]) + disturbance).astype(
        np.float32
    )
    assert np.array_equal(pair["edited"], expected)
    rounded_reedit = (pair["baseline"].astype(float) + analytic).astype(np.float32)
    assert not np.array_equal(pair["edited"], rounded_reedit)


def test_patterns_global_sign_and_limb_chest_rule(config):
    patterns = set()
    for seed in range(64):
        edit = real._edit_parameters(seed, [100, 400, 700], config)
        coefficients = np.array(edit["primitive_coefficients"])
        patterns.add(edit["pattern"])
        assert 1 <= np.count_nonzero(coefficients[2:]) <= 3
        if edit["pattern"]:
            assert np.count_nonzero(coefficients[2:]) == 1
        if edit["pattern"] == 3:
            assert coefficients[0] == coefficients[1] == 1
        assert edit["sign"] in (-1, 1)
        assert 0.04 <= abs(edit["amplitude_mv"]) <= 0.30
        assert 0.14 <= edit["width_seconds"] <= 0.19
    assert patterns == {0, 1, 2, 3}


def test_frozen_background_and_nuisance_hashes_cannot_change(config):
    source = background()
    case = real.cases_for_background(study_row(), [100, 400, 700], real.array_sha256(source), config)[0]
    changed = source.copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="background hash changed"):
        real.render_case(changed, case)
    changed_case = copy.deepcopy(case)
    changed_case["nuisance_parameters"]["gain"] += 0.01
    with pytest.raises(ValueError, match="nuisance parameters changed"):
        real.render_case(source, changed_case)


def test_native_decode_checksum_and_header_units(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    raw = tmp_path / "data/raw/ptb-xl/1.0.3"
    folder = raw / "records500"
    folder.mkdir(parents=True)
    time = np.arange(5000) / 500
    source = np.sin(2 * np.pi * 2 * time)[:, None] * np.linspace(0.1, 0.9, 12)[None, :]
    wfdb.wrsamp(
        "artificial",
        fs=500,
        units=["mV"] * 12,
        sig_name=list(LEADS),
        p_signal=source,
        fmt=["16"] * 12,
        write_dir=str(folder),
    )
    checks = {p.relative_to(raw).as_posix(): sha256_file(p) for p in folder.iterdir()}
    row = study_row()
    row["audit_payload_sha256"] = {ext: checks["records500/artificial" + ext] for ext in (".hea", ".dat")}
    decoded, _ = prep.prepare_background(row, checks)
    assert np.allclose(decoded, resample_full(source.T, 1, 2), atol=1e-4)
    with (folder / "artificial.dat").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum_mismatch"):
        prep.prepare_background(row, checks)


def test_exact_storage_forecast_and_overwrite_guard(tmp_path, monkeypatch):
    forecast = prep.storage_forecast()
    assert forecast["array_payload_bytes"] == 253440000
    assert forecast["array_header_bytes"] == 5 * 128
    monkeypatch.setattr(prep, "OUTPUT", tmp_path)
    monkeypatch.setattr(
        prep, "metadata_inputs", lambda: pytest.fail("Read metadata before overwrite refusal")
    )
    with pytest.raises(FileExistsError, match="must be new"):
        prep.prepare()


def test_complete_selection_is_written_before_any_processing(tmp_path, monkeypatch, config):
    rows = [
        dict(study_row(), study_role="train_reserved", cache_index=None, subject_index=None)
        for _ in range(500)
    ]
    rows += [
        dict(study_row("calibration" if i < 64 else "evaluation"), cache_index=i, subject_index=i)
        for i in range(192)
    ]
    binding = {"input_sha256": {}, "input_audit_source_sha256": {}, "prior_cumulative_cpu_seconds": 0}
    monkeypatch.setattr(prep, "OUTPUT", tmp_path / "new")
    monkeypatch.setattr(prep, "SOURCE_FILES", ())
    monkeypatch.setattr(prep, "stream_bytes", lambda: 0)
    monkeypatch.setattr(prep, "metadata_inputs", lambda: (rows, {}, binding, config))

    def stop(*_):
        saved = json.loads((prep.OUTPUT / "selection.json").read_bytes())
        assert len(saved["records"]) == 692
        assert saved["waveforms_opened_when_written"] == 0
        raise RuntimeError("artificial stop before payloads")

    monkeypatch.setattr(prep, "_process", stop)
    with pytest.raises(RuntimeError, match="artificial stop"):
        prep.prepare()


def test_frozen_receipt_payload_hash_rejected_before_selection(tmp_path, monkeypatch):
    from ecg_experiment.ssl_synthetic_preparation import verify_hashes

    payload = tmp_path / "frozen.txt"
    payload.write_text("original")
    receipt = tmp_path / "summary.json"
    receipt.write_text(json.dumps({"input_sha256": {"frozen.txt": sha256_file(payload)}}))
    payload.write_text("modified")
    monkeypatch.setattr(prep, "verify_inference_hashes", lambda values: verify_hashes(values, tmp_path))
    with pytest.raises(ValueError, match="Frozen input/source hash changed"):
        prep._bound_receipt(receipt)


def test_ssl_identity_must_match_existing_frozen_binding_not_just_self_declared_hash(
    tmp_path, monkeypatch, config
):
    root = tmp_path
    base = root / "outputs/ssl_synthetic_localization"
    folder = base / "real_cache_v1"
    folder.mkdir(parents=True)
    source = root / "configs"
    source.mkdir()
    (source / "ssl_synthetic_real_background_v1.json").write_text(json.dumps(config))
    cache = folder / "summary.json"
    cache.write_text("{}")
    selection = folder / "selection.json"
    selection.write_text("{}")
    closure = base / "pilot_v1/final_audit_v1/reproduction_binding_closure.json"
    closure.parent.mkdir(parents=True)
    closure.write_text("{}")
    expected = {path.relative_to(root).as_posix(): sha256_file(path) for path in (cache, selection, closure)}
    expected[selection.relative_to(root).as_posix()] = "0" * 64

    def bound(path):
        if path.name == "summary.json":
            return {"audit_completed": True, "input_sha256": expected}
        return {"all_passed": True, "mismatch_count": 0}

    monkeypatch.setattr(prep, "ROOT", root)
    monkeypatch.setattr(prep, "BASE", base)
    monkeypatch.setattr(prep, "load_audited_inputs", lambda *_: ({"records": []}, [], {}, {}))
    monkeypatch.setattr(prep, "_bound_receipt", bound)
    monkeypatch.setattr(prep, "to_stored", lambda path: path.relative_to(root).as_posix())
    monkeypatch.setattr(
        prep, "select_patients", lambda *_: pytest.fail("Selected from tampered SSL identities")
    )
    with pytest.raises(ValueError, match="differs from frozen original bindings"):
        prep.metadata_inputs()


def test_cumulative_cpu_allowance_exhausted_before_output_or_payloads(tmp_path, monkeypatch, config):
    rows = [
        dict(study_row(), study_role="train_reserved", cache_index=None, subject_index=None)
        for _ in range(500)
    ]
    rows += [
        dict(study_row("calibration" if i < 64 else "evaluation"), cache_index=i, subject_index=i)
        for i in range(192)
    ]
    binding = {"input_sha256": {}, "input_audit_source_sha256": {}, "prior_cumulative_cpu_seconds": 14400}
    monkeypatch.setattr(prep, "OUTPUT", tmp_path / "new")
    monkeypatch.setattr(prep, "metadata_inputs", lambda: (rows, {}, binding, config))
    monkeypatch.setattr(prep, "_process", lambda *_: pytest.fail("Opened exhausted-stage payload"))
    with pytest.raises(ValueError, match="CPU allowance exhausted before payload access"):
        prep.prepare()
    assert not (tmp_path / "new").exists()
