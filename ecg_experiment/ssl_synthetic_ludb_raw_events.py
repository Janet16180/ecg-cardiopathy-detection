"""Independently decode opaque WFDB events for one prospectively bounded annotation diagnosis."""

import hashlib
import struct
from typing import Any

SYMBOLS = {1: "N", 24: "p", 27: "t", 39: "(", 40: ")"}
METADATA = {59: "SKIP", 60: "NUM", 61: "SUB", 62: "CHN", 63: "AUX"}


def skip(payload: bytes, position: int, value: int) -> tuple[int, int]:
    """Independently decode the signed high-word-first long increment, retaining negative times."""
    if value != 0 or position + 4 > len(payload):
        raise ValueError("invalid_or_truncated_SKIP")
    high, low = struct.unpack_from("<HH", payload, position)
    increment = (high << 16) | low
    if increment >= 2**31:
        increment -= 2**32
    return increment, position + 4


def extra(
    payload: bytes, position: int, code: int, value: int, state: dict[str, int], events: list[dict[str, Any]]
) -> int:
    """Apply numeric metadata to current/subsequent events, exposing only auxiliary byte counts."""
    if not events:
        raise ValueError("metadata_without_annotation")
    current = events[-1]
    if code == 60:
        state["num"] = (value & 255) - 256 if value & 128 else value & 255
        current["num"] = state["num"]
    elif code == 61:
        current["subtype"] = (value & 255) - 256 if value & 128 else value & 255
    elif code == 62:
        state["channel"] = value
        current["channel"] = value
        if not 0 <= value < 12:
            raise ValueError("channel_outside_12_leads")
    elif code == 63:
        current["aux_length"] = value
        end = position + value + value % 2
        if end > len(payload):
            raise ValueError("truncated_AUX")
        return end
    return position


def decode(payload: bytes) -> dict[str, Any]:
    """Retain every event, unknown code and structural error without sorting, repair or text decoding.

    Parameters
    ----------
    payload : bytes
        One locally admitted MIT-format annotation file, never a waveform or header.

    Returns
    -------
    dict
        Original ordered events, numeric metadata, EOF state, byte identity and explicit errors.
        Auxiliary bytes are skipped opaquely; no text, diagnosis or patient fields are returned.
    """
    events: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    state = {"sample": 0, "channel": 0, "num": 0}
    position, eof = 0, False
    while position + 2 <= len(payload):
        start = position
        word = struct.unpack_from("<H", payload, position)[0]
        position += 2
        if word == 0:
            eof = True
            if position != len(payload):
                errors.append(
                    {"kind": "bytes_after_EOF", "byte_offset": position, "bytes": len(payload) - position}
                )
            break
        code, value = word >> 10, word & 1023
        try:
            if code == 59:
                increment, position = skip(payload, position, value)
                state["sample"] += increment
                metadata.append({"byte_offset": start, "type": "SKIP", "increment": increment})
            elif code in (60, 61, 62, 63):
                metadata.append({"byte_offset": start, "type": METADATA[code], "value": value})
                position = extra(payload, position, code, value, state, events)
            else:
                state["sample"] += value
                events.append(
                    {
                        "event_index": len(events),
                        "byte_offset": start,
                        "sample": state["sample"],
                        "type_code": code,
                        "symbol": SYMBOLS.get(code, f"unknown_code_{code}"),
                        "channel": state["channel"],
                        "num": state["num"],
                        "subtype": 0,
                        "aux_length": 0,
                    }
                )
        except ValueError as error:
            errors.append({"kind": str(error), "byte_offset": start})
            break
    if not eof:
        errors.append({"kind": "missing_EOF", "byte_offset": position})
        if position < len(payload):
            errors.append(
                {"kind": "undecoded_tail", "byte_offset": position, "bytes": len(payload) - position}
            )
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "events": events,
        "metadata": metadata,
        "metadata_counts": {
            name: sum(field["type"] == name for field in metadata) for name in METADATA.values()
        },
        "EOF": eof,
        "errors": errors,
    }
