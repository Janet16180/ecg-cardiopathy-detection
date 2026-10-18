"""Artificial-only factorial selection, mixed-bank routing, native math and provenance checks."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_counterfactual_factorial_common as common
from ecg_experiment import ssl_synthetic_counterfactual_factorial_data as data
from ecg_experiment import ssl_synthetic_counterfactual_factorial_preparation as prep
from ecg_experiment import ssl_synthetic_paired_background_data as old_data
from ecg_experiment.files import sha256_file
from ecg_experiment.ssl_synthetic_real_background import array_sha256, find_anchors


@pytest.fixture
def config():
    return json.loads(Path("configs/ssl_synthetic_counterfactual_factorial_v1.json").read_bytes())


def record(patient, role="validation", ecg=None):
    return {
        "patient_id": str(patient),
        "ecg_id": str(patient if ecg is None else ecg),
        "role": role,
        "filename_hr": "records500/artificial",
    }


def metadata():
    train = [dict(record(i, "train"), study_role="train") for i in range(1, 501)]
    first = {"records": [dict(record(i), study_role="calibration") for i in range(1000, 1192)]}
    previous = {"records": train + [dict(record(i), study_role="evaluation") for i in range(1200, 1392)]}
    ssl = {"records": [record(i, "train") for i in range(30000, 32048)]}
    rows = (
        [record(i) for i in range(1000, 1192)]
        + [record(i) for i in range(1200, 1392)]
        + [record(i) for i in range(2000, 2300)]
    )
    rows += [record(2000, ecg=9999), record(99999, "engineering_test")]
    audited = [
        dict(row, status="decoded", quality_exclusions=["retained_flag"], review_flags=[], payload_sha256={})
        for row in rows
    ]
    return rows, audited, [first, previous], ssl


def row(index=0):
    return dict(
        record(900001 + index),
        study_role="calibration" if index < 64 else "evaluation",
        cache_index=500 + index,
        subject_index=500 + index,
        background_row=index,
        role_subject_index=index if index < 64 else index - 64,
    )


def signal():
    time = np.arange(2500) / 250
    peaks = sum(np.exp(-0.5 * ((time - center) / 0.012) ** 2) for center in np.arange(0.6, 9.6, 0.5))
    return (peaks[None] * np.linspace(0.4, 1.5, 12)[:, None]).astype(np.float32)


def test_new_selection_excludes_all384_and_uses_lowest_without_quality_filter(config):
    rows, audited, prior, ssl = metadata()
    selected = data.select_validation(rows, audited, prior, ssl, config)
    assert selected == data.select_validation(list(reversed(rows)), audited, prior, ssl, config)
    actual = selected["records"]
    expected = sorted(
        range(2000, 2300),
        key=lambda i: (
            hashlib.sha256(f"ssl_counterfactual_factorial_v1:validation:{i}".encode()).hexdigest(),
            i,
        ),
    )[:192]
    assert [r["patient_id"] for r in actual] == list(map(str, expected))
    assert all(r["ecg_id"] == r["patient_id"] for r in actual)
    assert all(r["quality_exclusions"] == ["retained_flag"] for r in actual)
    assert [r["background_row"] for r in actual] == list(range(192))
    assert [r["cache_index"] for r in actual] == list(range(500, 692))
    assert selected["waveforms_opened_when_written"] == 0
    assert prior == metadata()[2]


@pytest.mark.parametrize(
    ("subject", "case", "local_bg", "local_mask"),
    [(500, 2000, 0, 0), (563, 2255, 63, 255), (564, 2256, 64, 256), (691, 2767, 191, 767)],
)
def test_new_virtual_boundaries(subject, case, local_bg, local_mask):
    assert data.background_row(subject) == local_bg
    assert data.new_case_row(case) == local_mask


@pytest.mark.parametrize(
    ("role", "study", "index"),
    [
        ("engineering_test", "evaluation", 64),
        ("excluded", "calibration", 0),
        ("train", "calibration", 0),
        ("validation", "train", 0),
    ],
)
def test_protected_roles_rejected_before_decode(role, study, index, monkeypatch):
    selected = dict(row(index), role=role, study_role=study)
    monkeypatch.setattr(data, "prepare_background", lambda *_: pytest.fail("Protected raw payload reached"))
    with pytest.raises(ValueError, match="validation"):
        data.decode_background(selected, {})


@pytest.mark.parametrize("index", [0, 63, 64, 191])
def test_new_native_cases_hashes_replays_and_matched_control_identity(config, index):
    background = signal()
    selected = row(index)
    anchors = find_anchors(background, config)["anchors"]
    support, candidate = {}, {}
    cases = prep.qualify_cases(background, selected, anchors, config, support, candidate)
    seed = int.from_bytes(
        hashlib.sha256(
            f"ssl_counterfactual_factorial_v1:{selected['study_role']}:{selected['patient_id']}".encode()
        ).digest()[:8],
        "big",
    )
    for offset, case in enumerate(cases):
        assert case["seed"] == seed
        assert case["case_index"] == 2000 + index * 4 + offset
        assert case["mask_row"] == index * 4 + offset
        assert case["profile"] == ("A" if index < 64 else "B")
        assert case["deterministic_replay_passed"]
        assert not case["qualification"]["failures"]
    for case in cases[:2]:
        control = cases[3 if case["nuisance"] else 2]
        assert case["baseline_sha256"] == control["edited_sha256"]
        assert case["matched_control_case_index"] == control["case_index"]


def test_mixed_mask_routing_never_uses_old_validation_rows():
    class Routed:
        def __init__(self, limit):
            self.limit = limit
            self.seen = []

        def __getitem__(self, index):
            assert 0 <= index < self.limit
            self.seen.append(index)
            return np.array([index])

    old, new = Routed(2000), Routed(768)
    context = {"train_support": old, "train_candidate": old, "new_support": new, "new_candidate": new}
    for index in (0, 1999, 2000, 2255, 2256, 2767):
        expected = index if index < 2000 else index - 2000
        assert common.support_for_case(context, index)[0] == expected
        assert common.candidate_for_case(context, index)[0] == expected
    assert old.seen == [0, 0, 1999, 1999]
    assert new.seen == [0, 0, 255, 255, 256, 256, 767, 767]


def test_changed_active_train_row_fails_before_renderer(config, monkeypatch):
    background = signal()
    selected = dict(
        record(900000, "train"), study_role="train", cache_index=0, subject_index=0, role_subject_index=0
    )
    cases = old_data.cases_for_background(selected, [150, 400, 650], array_sha256(background), config)
    original = copy.deepcopy(cases)
    changed = background.copy()
    changed[0, 0] += 0.1
    context = {"cases": cases, "train_backgrounds": {0: changed}}
    monkeypatch.setattr(common, "render_case", lambda *_: pytest.fail("Changed TRAIN row reached model path"))
    with pytest.raises(ValueError, match="Active background row changed"):
        common.render_checked(context, 0)
    assert cases == original


def test_changed_frozen_source_and_conflicting_declarations_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    file = tmp_path / "ecg_experiment/waveforms.py"
    file.parent.mkdir()
    file.write_text("frozen helper")
    authority = {"ecg_experiment/waveforms.py": sha256_file(file)}
    assert common.source_bindings(tuple(authority), authority) == authority
    file.write_text("changed helper")
    with pytest.raises(ValueError, match="Frozen reused source"):
        common.source_bindings(tuple(authority), authority)
    with pytest.raises(ValueError, match="Conflicting"):
        common.verify_bindings({"same": "one"}, {"same": "two"}, {})


def test_unused_waveform_deferred_without_open(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    name = "outputs/ssl_synthetic_localization/paired_background_v1/preparation/backgrounds.npy"
    monkeypatch.setattr(common, "sha256_file", lambda *_: pytest.fail("Unused old waveform file opened"))
    observed = common.verify_bindings({}, {name: "frozen"}, {name: "frozen"})
    assert observed[name]["whole_payload_read"] is False
    with pytest.raises(ValueError, match="authority"):
        common.verify_bindings({}, {name: "frozen"}, {})


def test_forecast_no_preparation_double_count_and_overwrite(tmp_path, monkeypatch, config):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    study = tmp_path / "study"
    study.mkdir()
    (study / "existing").write_bytes(b"x" * 100)
    monkeypatch.setattr(common, "CONFIG_PATH", cfg)
    monkeypatch.setattr(common, "STUDY", study)
    monkeypatch.setattr(common, "stream_bytes", lambda: 1000)
    projected = common.storage_projection()
    assert projected["projected_whole_stream_bytes"] == 1000 + 504869760 - 100
    monkeypatch.setattr(prep, "PREPARATION", tmp_path)
    monkeypatch.setattr(prep, "predecessor_metadata", lambda: pytest.fail("Metadata read before refusal"))
    with pytest.raises(FileExistsError):
        prep.prepare()


def test_complete_selection_written_before_raw(tmp_path, monkeypatch, config):
    rows, audited, previous, ssl = metadata()
    context = {
        "config": config,
        "partition": {"records": rows},
        "eligibility": {"records": audited},
        "first_selection": previous[0],
        "predecessor_selection": previous[1],
        "ssl_selection": ssl,
        "predecessor_cases": {"cases": []},
        "prior_cpu_seconds": 0,
        "inputs": {},
        "sources": {},
    }
    destination = tmp_path / "new"
    checks = tmp_path / common.CHECKSUM_PATH
    checks.parent.mkdir(parents=True)
    checks.write_text("")
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    monkeypatch.setattr(prep, "PREPARATION", destination)
    monkeypatch.setattr(prep, "predecessor_metadata", lambda: context)
    monkeypatch.setattr(prep, "storage_projection", lambda: {"projected_whole_stream_bytes": 0})
    monkeypatch.setattr(prep, "read_checksums", lambda *_: {})

    def inspect(*_):
        selected = json.loads((destination / "selection.json").read_bytes())
        assert len(selected["records"]) == 192
        assert selected["waveforms_opened_when_written"] == 0
        assert not (destination / "backgrounds.npy").exists()
        raise RuntimeError("artificial stop before raw")

    monkeypatch.setattr(prep, "process_rows", inspect)
    with pytest.raises(RuntimeError, match="artificial stop"):
        prep.prepare()


def test_admission_timeout_retains_measured_failure(tmp_path, monkeypatch):
    destination = tmp_path / "new"
    monkeypatch.setattr(prep, "PREPARATION", destination)

    def fail():
        raise TimeoutError("artificial admission budget failure")

    monkeypatch.setattr(prep, "predecessor_metadata", fail)
    with pytest.raises(TimeoutError, match="artificial admission"):
        prep.prepare()
    result = json.loads((destination / "failure.json").read_bytes())
    assert result["failure_type"] == "TimeoutError"
    assert result["cpu_seconds"] >= 0
    assert result["wall_seconds"] >= 0
    assert result["selection_written"] is False
    with pytest.raises(FileExistsError):
        prep.prepare()


def test_first_anchor_case_failure_retained_stops_further_raw(config, monkeypatch):
    class EmptyMap:
        def __setitem__(self, *_):
            pass

        def __getitem__(self, _):
            return np.full((12, 2500), np.nan, dtype=np.float32)

        def flush(self):
            pass

    opened = []

    def decode(selected, _):
        opened.append(selected["background_row"])
        return signal(), {"payload_sha256": {}, "verified_bytes": 1}

    def fail_cases(background, selected, anchors, cfg, *_):
        cases = data.cases_for_background(selected, anchors, array_sha256(background), cfg)
        for case in cases:
            case["qualification"] = {"failures": ["artificial_anchor_failure"], "metrics": {}}
        return cases

    monkeypatch.setattr(prep.np.lib.format, "open_memmap", lambda *_a, **_kw: EmptyMap())
    monkeypatch.setattr(prep, "decode_background", decode)
    monkeypatch.setattr(prep, "qualify_cases", fail_cases)
    monkeypatch.setattr(prep, "guard", lambda *_: None)
    monkeypatch.setattr(prep, "find_anchors", lambda *_: {"anchors": [], "minimum_anchor_gate_passed": False})
    info, cases = prep.process_rows([row(0), row(1)], {"config": config}, {}, 0, 0)
    assert opened == [0]
    assert len(info) == 2
    assert len(cases) == 8
    assert info[1]["status"] == "processing_failed"
    assert all(case["qualification"]["failures"] for case in cases)
