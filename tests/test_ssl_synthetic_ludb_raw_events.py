import struct

from ecg_experiment.ssl_synthetic_ludb_raw_events import decode


def event(code, delta):
    return struct.pack("<H", (code << 10) | delta)


def skip(increment):
    value = increment % 2**32
    return event(59, 0) + struct.pack("<HH", value >> 16, value & 65535)


def test_signed_skip_retains_original_nonmonotone_order():
    result = decode(event(39, 10) + event(1, 20) + skip(-25) + event(40, 5) + b"\0\0")
    assert [row["sample"] for row in result["events"]] == [10, 30, 10]
    assert [row["symbol"] for row in result["events"]] == ["(", "N", ")"]
    assert result["metadata_counts"]["SKIP"] == 1
    assert result["errors"] == []


def test_channel_inherits_but_auxiliary_text_is_never_exposed():
    payload = event(24, 12) + event(62, 3) + event(63, 3) + b"ABC\0" + event(40, 4) + b"\0\0"
    result = decode(payload)
    assert [row["channel"] for row in result["events"]] == [3, 3]
    assert result["metadata_counts"]["AUX"] == 1
    assert "ABC" not in str(result)
    assert result["errors"] == []


def test_unknown_event_is_retained_with_raw_type_not_discarded():
    result = decode(event(17, 42) + b"\0\0")
    assert result["events"][0]["type_code"] == 17
    assert result["events"][0]["symbol"] == "unknown_code_17"
    assert result["events"][0]["sample"] == 42
    assert result["errors"] == []


def test_truncated_auxiliary_retains_prior_events_and_explicit_errors():
    result = decode(event(1, 12) + event(63, 5) + b"A")
    assert len(result["events"]) == 1
    assert result["EOF"] is False
    assert [row["kind"] for row in result["errors"]] == ["truncated_AUX", "missing_EOF", "undecoded_tail"]


def test_padded_or_missing_eof_is_visible():
    assert decode(b"\0\0\0\0")["errors"][0]["kind"] == "bytes_after_EOF"
    assert decode(event(1, 10))["errors"][0]["kind"] == "missing_EOF"


def test_invalid_channel_is_retained_and_flagged():
    result = decode(event(1, 10) + event(62, 19) + b"\0\0")
    assert result["events"][0]["channel"] == 19
    assert result["errors"][0]["kind"] == "channel_outside_12_leads"


def test_numeric_num_inherits_and_subtype_resets_between_events():
    payload = event(1, 10) + event(60, 255) + event(61, 254) + event(1, 5) + b"\0\0"
    result = decode(payload)
    assert [row["num"] for row in result["events"]] == [-1, -1]
    assert [row["subtype"] for row in result["events"]] == [-2, 0]
    assert result["errors"] == []
