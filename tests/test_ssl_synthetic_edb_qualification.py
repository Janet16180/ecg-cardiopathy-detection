"""Artificial-only contracts for complete EDB annotation decoding and qualification."""

import hashlib
import json

import numpy as np
import pytest
from wfdb.io import annotation as wfdb_annotation

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_edb_qualification as core


def config():
    policy = json.loads((ROOT / core.CONFIG).read_text())
    policy["contract"] = json.loads((ROOT / core.CONTRACT).read_text())
    return policy


def event(sample, code=1, aux="", subtype=0, channel=0, num=0):
    payload = aux.encode("latin1")
    return {
        "sample": sample,
        "type_code": code,
        "subtype": subtype,
        "channel": channel,
        "num": num,
        "aux_length": len(payload),
        "aux_hex": payload.hex(),
    }


def qualify(*events):
    ordered = [{**value, "event_index": index} for index, value in enumerate(events)]
    return core.qualify_events(ordered, config())


def episode(start, stop, family="ST", channel=0, sign="+", axis=False, peak=None):
    code = 22 if axis else 18 if family == "ST" else 19
    name, prefix = (family.lower(), "a") if axis else (family, "A")
    return [
        event(start, code, f"({name}{channel}{sign}", channel=1 - channel),
        event(start if peak is None else peak, code, f"{prefix}{name}{channel}{sign}100"),
        event(stop, code, f"{name}{channel}{sign})"),
    ]


def word(code, delta=0):
    return bytes((delta & 255, code * 4 + (delta >> 8)))


def aux_word(text, padding=b"\x91"):
    payload = text.encode("latin1")
    return word(63, len(payload)) + payload + (padding if len(payload) % 2 else b"")


def artificial_record(tmp_path, payload, header=None):
    base = tmp_path / "fixture"
    if header is None:
        header = (
            "fixture 2 250 1800000\nfixture.dat 212 200 12 0 0 0 0 literal0\n"
            "fixture.dat 212 200 12 0 0 0 0 literal1\n"
            "# Unread clinical comment must never enter the sanitized header\n"
        )
    base.with_suffix(".hea").write_text(header)
    base.with_suffix(".atr").write_bytes(payload)
    return base


def assert_record_issue(result, kind):
    assert any(issue["kind"] == kind for issue in result["issues"])
    assert not result["engineering_passed"]
    assert np.all(result["labels"] == 0)
    assert np.all(result["reasons"] == 128)


def test_complete_raw_keeps_sample_zero_axis_and_code_zero(tmp_path):
    payload = (
        word(22)
        + aux_word("(st0+")
        + word(0, 1)
        + word(14, 1)
        + word(61, 17)
        + word(22, 1)
        + aux_word("ast0+100")
        + word(22, 1)
        + aux_word("st0+)")
        + b"\0\0"
    )
    record = artificial_record(tmp_path, payload)
    decoded = core.decode_record(record, tmp_path / "sanitized", config())
    metadata = decoded["metadata"]
    assert metadata["event_count"] == 5
    assert metadata["wfdb_removed_indices"] == [0, 1]
    assert metadata["wfdb_rdann_event_count"] == 3
    assert metadata["wfdb_effective_view_verified"]
    assert metadata["wfdb_filtered_event_sha256"] == metadata["wfdb_rdann_event_sha256"]
    assert metadata["source_signal_names"] == ["literal0", "literal1"]
    sanitized = (tmp_path / "sanitized/fixture.hea").read_text()
    assert len(sanitized.splitlines()) == 3
    assert "clinical" not in sanitized
    result = core.qualify_events(decoded["events"], config())
    assert result["engineering_passed"]
    assert result["episodes"][0]["start"] == 0
    assert result["episodes"][0]["category"] == "axis"


def test_skip_signed_fields_inheritance_and_logical_aux_padding(tmp_path):
    skip = word(59) + bytes((0, 0, 0xD0, 0x07))
    payload = (
        word(1, 2)
        + word(60, 254)
        + word(61, 255)
        + word(62, 7)
        + aux_word("\x00\xffx")
        + skip
        + word(1, 3)
        + b"\0\0"
    )
    record = artificial_record(tmp_path, payload)
    decoded = core.decode_record(record, tmp_path / "sanitized", config())
    first, second = decoded["events"]
    assert (first["sample"], first["num"], first["subtype"], first["channel"]) == (2, -2, -1, 7)
    assert (first["aux_length"], first["aux_hex"]) == (3, "00ff78")
    assert (second["sample"], second["num"], second["subtype"], second["channel"]) == (2005, -2, 0, 7)
    assert second["aux_hex"] == ""
    rows = [[value[key] for key in core.EVENT_FIELDS] for value in decoded["events"]]
    expected = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    assert core.event_digest(decoded["events"]) == expected


@pytest.mark.parametrize("payload", [b"", b"\x01", word(1, 2)])
def test_invalid_annotation_envelope_raises(tmp_path, payload):
    record = artificial_record(tmp_path, payload)
    with pytest.raises(ValueError, match="EOF marker"):
        core.decode_record(record, tmp_path / "sanitized", config())


def test_header_geometry_failure_raises_before_annotation_decode(tmp_path, monkeypatch):
    record = artificial_record(tmp_path, word(1, 2) + b"\0\0", "fixture 12 500 5000\na\nb\n")
    monkeypatch.setattr(core.wfdb_annotation, "proc_ann_bytes", lambda *args: pytest.fail("decoded"))
    with pytest.raises(ValueError, match="geometry"):
        core.decode_record(record, tmp_path / "sanitized", config())


def test_no_quality_is_unknown_never_reference():
    result = qualify()
    assert result["engineering_passed"]
    assert result["labels"].shape == (2, 720)
    assert result["labels"].dtype == np.uint8
    assert result["reasons"].dtype == np.uint16
    assert np.all(result["labels"] == 0)
    assert np.all(result["reasons"] == 1)


@pytest.mark.parametrize(
    ("subtype", "bits"),
    [
        (0, (0, 0)),
        (1, (2, 0)),
        (2, (0, 2)),
        (3, (2, 2)),
        (17, (4, 0)),
        (18, (4, 2)),
        (32, (0, 4)),
        (33, (2, 4)),
        (51, (4, 4)),
    ],
)
def test_published_nine_quality_states_are_channel_local(subtype, bits):
    result = qualify(event(0, 14, subtype=subtype))
    assert result["engineering_passed"]
    for channel, bit in enumerate(bits):
        assert np.all(result["reasons"][channel] == bit)
        assert np.all(result["labels"][channel] == (2 if bit == 0 else 0))


def test_unknown_quality_prefix_partial_windows_and_coincident_original_order():
    result = qualify(event(100, 14, subtype=3), event(2500, 14, subtype=51), event(2500, 14))
    assert result["reasons"][0, 0] == 3
    assert result["labels"][0, 1] == 2
    assert result["quality_spans"] == [
        {"start": 0, "end": 100, "states": ["unknown", "unknown"]},
        {"start": 100, "end": 2500, "states": ["n", "n"]},
        {"start": 2500, "end": 1800000, "states": ["c", "c"]},
    ]


@pytest.mark.parametrize(
    ("events", "kind"),
    [
        ([event(0, 14, subtype=4)], "unknown_quality_subtype"),
        ([event(100, 14), event(99, 14)], "decreasing_quality_sample"),
        ([event(-1)], "event_out_of_bounds"),
        ([event(1800000)], "event_out_of_bounds"),
    ],
)
def test_record_issues_make_every_cell_unknown128(events, kind):
    assert_record_issue(qualify(*events), kind)


def test_unrelated_global_time_decrease_is_only_diagnostic():
    result = qualify(event(0, 14), event(100), event(20))
    assert result["engineering_passed"]
    assert len(result["diagnostics"]) == 1
    assert np.all(result["labels"] == 2)


def test_st_positive_precedes_margin_and_other_channel_remains_reference():
    result = qualify(event(0, 14), *episode(2500, 10000, peak=10000))
    assert result["engineering_passed"]
    assert result["labels"][0, :8].tolist() == [0, 1, 1, 1, 0, 0, 0, 2]
    assert np.all(result["labels"][1] == 2)
    assert np.all(result["reasons"][result["labels"] != 0] == 0)
    assert result["episodes"][0]["peak_samples"] == [10000]


def test_two_intersecting_st_intervals_including_partial_are_unknown():
    result = qualify(event(0, 14), *episode(0, 10000), *episode(5000, 6250, sign="-"))
    assert result["engineering_passed"]
    assert result["labels"][0, 2] == 0
    assert result["reasons"][0, 2] == 8 | 16 | 64
    assert result["labels"][0, 1] == 1


def test_axis_intersection_is_channel_local_and_half_open():
    result = qualify(event(0, 14), *episode(0, 10000), *episode(2500, 5000, axis=True))
    assert result["engineering_passed"]
    assert result["labels"][0, :4].tolist() == [1, 0, 1, 1]
    assert result["reasons"][0, 1] == 32 | 64
    assert np.all(result["labels"][1] == 2)
    assert result["overlap_pairs"][0]["cross_category"]


def test_t_and_axis_t_extremes_need_no_peaks_but_need_same_parent():
    events = [event(0, 14)]
    for axis in (False, True):
        code, name = (22, "t") if axis else (19, "T")
        events += episode(0, 10000, family="T", axis=axis)
        events += [event(2500, code, f"({name}0++"), event(5000, code, f"{name}0++)")]
    result = qualify(*events)
    assert result["engineering_passed"]
    assert result["counts"]["extreme_episodes"] == 2
    for interval in result["episodes"]:
        if interval["extreme"]:
            assert interval["peak_samples"] == []
            assert len(interval["parent_begin_indices"]) == 1
    assert result["counts"]["ST_positive_windows"] == 0
    orphan = qualify(event(100, 19, "(T0++"), event(200, 19, "T0++)"))
    assert_record_issue(orphan, "extreme_T_parent_count")


@pytest.mark.parametrize(
    ("events", "kind"),
    [
        ([event(0, 18, "(ST0+"), event(100, 18, "ST0+)")], "missing_episode_peak"),
        ([event(0, 18, "(ST0+"), event(1, 18, "(ST0+")], "nested_same_key_begin"),
        ([event(1, 18, "ST0+)")], "orphan_end"),
        ([event(1, 18, "AST0+100")], "orphan_peak"),
        ([event(1, 18, "(ST0+")], "unclosed_episode"),
        (episode(100, 50), "nonpositive_episode_duration"),
        (episode(100, 200, peak=201), "peak_outside_boundaries"),
        ([event(1, 18, "(ST0++")], "double_sign_ST_not_allowed"),
        ([event(1, 18, "(T0+")], "episode_family_type_mismatch"),
        ([event(1, 22, "(ST0+")], "unsupported_episode_candidate"),
        ([event(1, 19, "(T0+-")], "unsupported_episode_candidate"),
        ([event(1, 18, " (ST0+")], "unsupported_episode_candidate"),
        ([event(1, 18, "AST0+¹²³")], "unsupported_episode_candidate"),
    ],
)
def test_unresolved_episode_grammar_is_never_repaired(events, kind):
    assert_record_issue(qualify(*events), kind)


def test_multiple_peaks_and_equal_boundary_markers_preserve_multiplicity():
    result = qualify(
        event(0, 14),
        event(0, 18, "(ST0+"),
        event(0, 18, "AST0+100"),
        event(2500, 18, "AST0+100"),
        event(2500, 18, "AST0+100"),
        event(2500, 18, "ST0+)"),
    )
    assert result["engineering_passed"]
    assert result["episodes"][0]["peak_samples"] == [0, 2500, 2500]
    assert result["labels"][0, 0] == 1


def test_definitions_only_actual_syntax_and_exact_interpreted_meaning():
    row = wfdb_annotation.ann_label_table.loc[18]
    valid = [
        event(0, 22, "## time resolution: 250"),
        event(0, 22, "## annotation type definitions"),
        event(0, 22, f"18 {row['symbol']} {row['description']}"),
        event(0, 22, "## end of definitions"),
    ]
    result = qualify(*valid, event(0, 14))
    assert result["engineering_passed"]
    assert len(result["definitions"]) == 4
    assert len(result["selected_events"]) == 1
    valid[2] = event(0, 22, f"18 {row['symbol']} changed meaning")
    assert_record_issue(qualify(*valid), "unresolved_annotation_definition")
    assert_record_issue(qualify(event(0, 22, "## time resolution: 500")), "unresolved_annotation_definition")
    assert_record_issue(qualify(event(0, 22, "##unknown")), "unresolved_annotation_definition")


def test_known_nonadvancing_wfdb_definition_loop_is_reported_before_rdann(tmp_path, monkeypatch):
    record = artificial_record(tmp_path, word(22) + aux_word("## unknown definition") + b"\0\0")
    monkeypatch.setattr(core.wfdb, "rdann", lambda *a, **k: pytest.fail("unsafe rdann called"))
    decoded = core.decode_record(record, tmp_path / "sanitized", config())
    assert decoded["metadata"]["decoder_issues"][0]["kind"] == "WFDB_unsafe_definition_scan"
    assert decoded["metadata"]["wfdb_rdann_event_count"] is None


def test_unrelated_comment_aux_is_not_persisted_in_selected_output():
    secret = "Unrelated free-text comment"
    result = qualify(event(0, 14, secret), event(1, 22, secret))
    assert result["engineering_passed"]
    output = {key: value for key, value in result.items() if key not in {"labels", "reasons"}}
    assert secret not in json.dumps(output)
    assert len(result["selected_events"]) == 1


def test_unexpected_errors_propagate_and_contract_drift_raises():
    malformed = event(1)
    malformed["aux_length"] = 1
    with pytest.raises(ValueError, match="AUX length"):
        qualify(malformed)
    policy = config()
    policy["contract"]["unknown_reason_bits"]["record_issue"] = 256
    with pytest.raises(ValueError, match="clarification"):
        core.qualify_events([], policy)


@pytest.mark.parametrize(
    ("notes", "safe"),
    [
        (["## time resolution: 250"], True),
        (["## time resolution: 250", "## time resolution: 250"], False),
        (["## end of definitions"], False),
        (["## unknown"], False),
        (["## annotation type definitions"], False),
        (["## annotation type definitions", "bad entry", "## end of definitions"], False),
        (["## annotation type definitions", "18 s ST change", "## end of definitions"], True),
        (["##unknown"], True),
        (["(st0+"], True),
    ],
)
def test_wfdb_guard_covers_exact_nonadvancing_and_unbounded_source_branches(notes, safe):
    events = [{**event(0, 22, note), "event_index": index} for index, note in enumerate(notes)]
    assert core._wfdb_definition_safe(events, len(events)) is safe


def test_wfdb_guard_uses_installed_prefix_indexing_not_removed_event_positions():
    events = [event(100, 22, "## unknown"), event(0, 22, "(st0+")]
    assert not core._wfdb_definition_safe(events, 1)
    events = [event(100, 22, "ordinary comment"), event(0, 22, "## unknown")]
    assert core._wfdb_definition_safe(events, 1)


def test_multiple_containing_parents_are_all_retained_as_diagnostic():
    result = qualify(
        event(0, 14),
        *episode(0, 10000, family="T"),
        *episode(0, 10000, family="T"),
        event(2500, 19, "(T0++"),
        event(5000, 19, "T0++)"),
    )
    assert result["engineering_passed"]
    extreme = result["episodes"][-1]
    assert extreme["parent_count"] == 2
    assert extreme["parent_begin_indices"] == [1, 4]
    assert any(item["kind"] == "multiple_containing_extreme_T_parents" for item in result["diagnostics"])


def test_cross_channel_and_category_overlaps_are_descriptive_only():
    result = qualify(event(0, 14), *episode(0, 10000), *episode(0, 10000, channel=1, axis=True))
    assert result["engineering_passed"]
    assert result["overlap_pairs"] == [
        {"interval_indices": [0, 1], "cross_channel": True, "cross_category": True}
    ]
    assert np.all(result["labels"][0, :4] == 1)


def test_decoder_disagreement_is_structured_without_claiming_comparison_pass(tmp_path, monkeypatch):
    record = artificial_record(tmp_path, word(1, 2) + b"\0\0")
    original = core.wfdb.rdann

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result.sample[0] += 1
        return result

    monkeypatch.setattr(core.wfdb, "rdann", changed)
    metadata = core.decode_record(record, tmp_path / "sanitized", config())["metadata"]
    assert not metadata["wfdb_effective_view_verified"]
    assert metadata["decoder_issues"] == [{"kind": "WFDB_effective_view_differs_from_declared_filter"}]


def test_unexpected_rdann_exception_is_not_swallowed(tmp_path, monkeypatch):
    record = artificial_record(tmp_path, word(1, 2) + b"\0\0")

    def fail(*args, **kwargs):
        raise RuntimeError("unexpected decoder failure")

    monkeypatch.setattr(core.wfdb, "rdann", fail)
    with pytest.raises(RuntimeError, match="unexpected decoder failure"):
        core.decode_record(record, tmp_path / "sanitized", config())
