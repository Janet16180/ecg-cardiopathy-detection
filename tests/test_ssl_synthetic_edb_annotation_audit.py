"""Artificial binary, inherited-field and raw-versus-effective annotation integrity checks."""

import struct

import numpy as np
import pytest
from wfdb.io.annotation import proc_ann_bytes

from ecg_experiment import ssl_synthetic_edb_annotation_audit as audit


def word(code: int, value: int) -> bytes:
    """Encode an artificial MIT annotation word without reading source assets."""
    return struct.pack("<H", (code << 10) | value)


def auxiliary(value: bytes, pad: bytes = b"\x00") -> bytes:
    """Encode exact logical auxiliary bytes and one independent odd-length padding byte."""
    return word(63, len(value)) + value + (pad if len(value) % 2 else b"")


def skip(delta: int) -> bytes:
    """Encode the format's signed high-word-first long sample increment."""
    value = delta & 0xFFFFFFFF
    return word(59, 0) + struct.pack("<HH", value >> 16, value & 0xFFFF)


def test_actual_wfdb_full_parser_matches_independent_fields_and_logical_auxiliary_bytes() -> None:
    """Nonzero padding, signed metadata and inheritance cannot alter logical AUX or event identity."""
    payload = (
        word(22, 0)
        + auxiliary(b"(st1+", b"\x7f")
        + skip(-1)
        + word(0, 1)
        + word(18, 100)
        + word(61, 17)
        + word(62, 255)
        + word(60, 254)
        + auxiliary(b"(ST0+", b"\x7f")
        + word(1, 20)
        + auxiliary(b"\xff\x00x", b"\x7f")
        + word(1, 1)
        + b"\x00\x00"
    )
    actual = audit.decode(payload)
    assert not actual["errors"]
    assert actual["EOF"]
    fields = proc_ann_bytes(np.frombuffer(payload, dtype=np.uint8).reshape(-1, 2), None)
    expected = [
        [
            int(sample),
            int(code),
            int(subtype),
            int(channel),
            int(number),
            len(aux),
            aux.encode("latin1").hex(),
        ]
        for sample, code, subtype, channel, number, aux in zip(*fields, strict=True)
    ]
    assert audit.canonical_events(actual["events"]) == expected
    assert actual["events"][3]["subtype"] == 0
    assert actual["events"][3]["channel"] == 255
    assert actual["events"][3]["num"] == -2
    assert actual["events"][3]["aux_hex"] == "ff0078"
    assert actual["events"][4]["aux_length"] == 0
    assert actual["events"][4]["aux_hex"] == ""
    assert actual["auxiliary_segments"][0][0]["padding_hex"] == "7f"


def test_sample_zero_axis_and_code_zero_are_retained_even_when_wfdb_effective_filter_removes_them() -> None:
    """A software filter is not evidence that a real sample-zero episode marker is metadata."""
    payload = word(22, 0) + auxiliary(b"(st1+") + skip(-1) + word(0, 1) + word(1, 5) + b"\x00\x00"
    parsed = audit.decode(payload)
    views = audit.decoder_views(parsed["events"])
    assert views["full_raw_events"] == 3
    assert views["wfdb_effective_events"] == 1
    assert views["removed_raw_indices"] == [0, 1]
    assert parsed["events"][0]["aux_hex"] == b"(st1+".hex()
    assert views["full_raw_digest"] != views["wfdb_effective_digest"]


def test_auxiliary_truncation_is_retained_as_failure_without_filling_missing_bytes() -> None:
    """A malformed source cannot be repaired into a plausible annotation string."""
    parsed = audit.decode(word(18, 100) + word(63, 9) + b"(ST")
    assert not parsed["EOF"]
    assert {error["kind"] for error in parsed["errors"]} >= {"truncated_AUX", "incomplete_auxiliary_bytes"}
    assert parsed["events"][0]["aux_hex"] == b"(ST".hex()
    assert parsed["events"][0]["aux_length"] == 9


def test_unknown_type_and_duplicate_metadata_are_exposed_without_event_deduplication() -> None:
    """Unknown types and repeated same-time events remain visible; ambiguous AUX is flagged."""
    payload = word(58, 10) + auxiliary(b"one") + auxiliary(b"two") + word(58, 0) + b"\x00\x00"
    parsed = audit.decode(payload)
    assert len(parsed["events"]) == 2
    assert [event["type_code"] for event in parsed["events"]] == [58, 58]
    assert [event["sample"] for event in parsed["events"]] == [10, 10]
    assert any(error["kind"] == "multiple_auxiliary_fields" for error in parsed["errors"])
    assert len(parsed["auxiliary_segments"][0]) == 2


def event(sample, code, text="", subtype=0, channel=0):
    return {
        "sample": sample,
        "type_code": code,
        "subtype": subtype,
        "channel": channel,
        "num": 0,
        "aux_length": len(text.encode("ascii")),
        "aux_hex": text.encode().hex(),
    }


def test_interleaved_channel_keys_extreme_parent_and_sample_zero_axis():
    events = [
        event(0, 14),
        event(0, 22, "(st1+"),
        event(1, 18, "(ST0-", channel=255),
        event(2, 19, "(T1+"),
        event(3, 19, "(T1++"),
        event(4, 1),
        event(5, 18, "AST0-123"),
        event(6, 22, "ast1+1234"),
        event(7, 19, "T1++)"),
        event(8, 19, "AT1+123"),
        event(9, 22, "st1+)"),
        event(10, 18, "ST0-)"),
        event(11, 19, "T1+)"),
    ]
    parsed = audit.episodes(events, samples=20)
    assert parsed["issues"] == []
    assert len(parsed["intervals"]) == 4
    assert parsed["selected"][0]["category"] == "axis"
    assert parsed["selected"][1]["channel"] == 0
    extreme = next(item for item in parsed["intervals"] if item["extreme"])
    assert extreme["peak_samples"] == []
    assert any(item["cross_channel"] for item in parsed["overlaps"])
    assert audit.decoder_views(events)["removed_raw_indices"] == [1]


def test_equal_sample_peak_order_and_no_repair():
    parsed = audit.episodes(
        [event(0, 18, "AST0+123"), event(0, 18, "(ST0+"), event(2, 18, "ST0+)"), event(2, 18, "AST0+123")], 10
    )
    assert [issue["kind"] for issue in parsed["issues"]] == [
        "orphan_peak",
        "base_interval_without_peak",
        "orphan_peak",
    ]
    valid = audit.episodes(
        [event(0, 18, "(ST0+"), event(0, 18, "AST0+123"), event(2, 18, "AST0+456"), event(2, 18, "ST0+)")], 10
    )
    assert valid["issues"] == []
    assert valid["intervals"][0]["peak_samples"] == [0, 2]


def test_unknown_quality_and_double_updates_channel_local_grid():
    parsed = audit.episodes(
        [
            event(10, 14, subtype=1),
            event(10, 14),
            event(10, 18, "(ST0+"),
            event(15, 18, "AST0+123"),
            event(20, 18, "ST0+)"),
        ],
        40,
    )
    labels, reasons = audit.window_grid(parsed, samples=40, width=10, margin=0)
    np.testing.assert_array_equal(labels, [[0, 1, 2, 2], [0, 2, 2, 2]])
    np.testing.assert_array_equal(reasons, [[1, 0, 0, 0], [1, 0, 0, 0]])
    assert parsed["quality"][0]["state"] == "nc"
    assert len(parsed["quality"]) == 2


def test_partial_second_st_axis_and_or_reasons():
    parsed = audit.episodes(
        [
            event(0, 14),
            event(0, 18, "(ST0+"),
            event(1, 18, "AST0+123"),
            event(5, 18, "(ST0-"),
            event(6, 18, "AST0-123"),
            event(8, 18, "ST0-)"),
            event(10, 22, "(st0+"),
            event(11, 22, "ast0+123"),
            event(15, 22, "st0+)"),
            event(20, 18, "ST0+)"),
        ],
        40,
    )
    labels, reasons = audit.window_grid(parsed, 40, 10, 0)
    np.testing.assert_array_equal(labels[0], [0, 0, 2, 2])
    assert reasons[0, 0] == 8 | 16 | 64
    assert reasons[0, 1] == 32 | 64
    np.testing.assert_array_equal(labels[1], [2, 2, 2, 2])
    assert np.all(reasons[1] == 0)


def test_record_issue_alone_and_unknown_subtype():
    parsed = audit.episodes([event(0, 14, subtype=127), event(1, 18, "(ST0+")], 20)
    labels, reasons = audit.window_grid(parsed, 20, 10, 0)
    assert {item["kind"] for item in parsed["issues"]} == {"unknown_quality_subtype", "unclosed_interval"}
    assert np.all(labels == 0)
    assert np.all(reasons == 128)


def test_exact_case_definition_and_extreme_rejections():
    assert audit.selected_tag(event(0, 18, "(ST0+ "))["error"] == "unsupported_candidate_string"
    assert audit.selected_tag(event(0, 18, "(st0+"))["error"] == "candidate_code_case_or_family_mismatch"
    assert audit.selected_tag(event(0, 19, "(T0+-"))["error"] == "unsupported_sign_or_extreme_family"
    parsed = audit.episodes(
        [event(0, 22, "## time resolution: 250"), event(1, 22, "(t0++"), event(2, 22, "t0++)")], 10
    )
    assert parsed["definitions"]["indices"] == [0]
    assert parsed["issues"] == [{"kind": "extreme_without_containing_base", "index": 1}]
    bad = audit.episodes([event(0, 22, "## time resolution: 500")], 10)
    assert bad["issues"][0]["kind"] == "definition_time_resolution_differs"


def test_grouping_roles_no_component_leakage_and_missing_member():
    import json
    from pathlib import Path

    config = json.loads(Path("configs/ssl_synthetic_edb_annotation_qualification_v1.json").read_text())
    records = [f"e{index:04d}" for index in range(100, 190)]
    assignments = audit.group_roles(records, config)
    assert len({item["component_id"] for item in assignments}) == 78
    for group in config["grouping"]["explicit_groups"]:
        selected = [item for item in assignments if item["record_id"] in group]
        assert {item["component_id"] for item in selected} == {min(group)}
        assert len({item["role"] for item in selected}) == 1
    with pytest.raises(ValueError, match="linkage member absent"):
        audit.group_roles(
            records[:-1] + ["e9999"]
            if records[-1] == "e0163"
            else [name if name != "e0163" else "e9999" for name in records],
            config,
        )


def test_negative_skip_diagnostic_does_not_repair_or_gate_unrelated_events():
    parsed = audit.episodes([event(5, 1), event(2, 1), event(3, 14)], 20)
    assert parsed["issues"] == []
    assert parsed["diagnostics"] == [{"kind": "decreasing_event_sample", "index": 1}]
    invalid = audit.episodes([event(5, 14), event(2, 14)], 20)
    assert invalid["issues"] == [{"kind": "decreasing_quality_update", "index": 1}]
    labels, reasons = audit.window_grid(invalid, 20, 10, 0)
    assert np.all(labels == 0)
    assert np.all(reasons == 128)


def test_structural_header_uses_only_three_lines_and_preserves_observed_names():
    lines = [
        "e0101 2 250 1800000",
        "e0101.dat 212 200 12 0 1 2 0 V4",
        "e0101.dat 212 200 12 0 1 2 0 native other lead",
    ]
    result = audit.structural_header(lines, "e0101")
    assert result["source_signal_names"] == ["V4", "native other lead"]
    with pytest.raises(ValueError, match="three structural"):
        audit.structural_header(lines + ["# Never read this clinical comment"], "e0101")
    with pytest.raises(ValueError, match="geometry"):
        audit.structural_header([lines[0].replace("250", "500"), *lines[1:]], "e0101")
    assert audit.selected_tag(event(0, 22, "(ST0+"))["error"] == "candidate_code_case_or_family_mismatch"
    assert audit.selected_tag(event(0, 22, "(st2+"))["error"] == "unsupported_candidate_string"


def test_margin_boundary_contact_clipping_and_positive_precedence():
    parsed = audit.episodes(
        [event(0, 14), event(20, 18, "(ST0+"), event(25, 18, "AST0+123"), event(30, 18, "ST0+)")], 50
    )
    labels, reasons = audit.window_grid(parsed, 50, 10, 10)
    np.testing.assert_array_equal(labels[0], [2, 0, 1, 0, 2])
    np.testing.assert_array_equal(reasons[0], [0, 64, 0, 64, 0])
    assert np.all(labels[1] == 2)
    clipped = audit.window_grid(parsed, 50, 10, 100)
    np.testing.assert_array_equal(clipped[0][0], [0, 0, 1, 0, 0])


def test_quality_reason_union_and_unknown_prefix():
    parsed = audit.episodes([event(2, 14, subtype=1), event(4, 14, subtype=17)], 20)
    labels, reasons = audit.window_grid(parsed, 20, 10, 0)
    np.testing.assert_array_equal(labels, [[0, 0], [0, 2]])
    np.testing.assert_array_equal(reasons, [[1 | 2 | 4, 4], [1, 0]])


def test_exact_custom_standard_restatement_and_changed_description():
    declarations = [
        event(0, 22, "## annotation type definitions"),
        event(0, 22, "18 s ST change"),
        event(0, 22, "## end of definitions"),
    ]
    assert audit.episodes(declarations, 10)["issues"] == []
    declarations[1] = event(0, 22, "18 s ST change ")
    parsed = audit.episodes(declarations, 10)
    assert parsed["issues"] == [{"kind": "unresolved_interpreted_code_definition", "index": 1}]


def test_multiple_containing_valid_bases_retained_without_uniqueness_gate():
    events = [
        event(0, 19, "(T0+"),
        event(1, 19, "AT0+123"),
        event(100, 19, "T0+)"),
        event(10, 19, "(T0+"),
        event(20, 19, "AT0+456"),
        event(90, 19, "T0+)"),
        event(30, 19, "(T0++"),
        event(40, 19, "T0++)"),
    ]
    parsed = audit.episodes(events, 200)
    assert parsed["issues"] == []
    assert parsed["intervals"][-1]["parent_begin_indices"] == [0, 3]
    assert parsed["intervals"][-1]["parent_count"] == 2
    assert parsed["diagnostics"][-1]["kind"] == "multiple_containing_bases"
