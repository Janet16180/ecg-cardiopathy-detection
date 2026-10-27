"""Qualify public EDB episode annotations without waveforms or neural predictions."""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import wfdb
from wfdb.io import annotation as wfdb_annotation

CONFIG = "configs/ssl_synthetic_edb_annotation_qualification_v1.json"
CONTRACT = "configs/ssl_synthetic_edb_annotation_contract_v1.json"
EVENT_FIELDS = ("sample", "type_code", "subtype", "channel", "num", "aux_length", "aux_hex")
REASON_BITS = {
    "quality_unknown": 1,
    "quality_noisy": 2,
    "quality_unreadable": 4,
    "partial_ST_intersection": 8,
    "multiple_ST_intersections": 16,
    "axis_intersection": 32,
    "expanded_episode_margin": 64,
    "record_issue": 128,
}
SOURCES = (
    CONFIG,
    CONTRACT,
    "docs/ssl-synthetic-localization-edb-annotation-qualification-v1.md",
    "docs/ssl-synthetic-localization-edb-annotation-contract-v1.md",
    "ecg_experiment/ssl_synthetic_edb_qualification.py",
    "tests/test_ssl_synthetic_edb_qualification.py",
)


def event_digest(events: list[dict[str, Any]]) -> str:
    """Hash complete canonical event tuples without exposing unrelated AUX text.

    Parameters
    ----------
    events : list
        Ordered events with the seven prospectively frozen canonical fields.

    Returns
    -------
    str
        SHA-256 of compact, ASCII JSON canonical tuples.
    """
    rows = [[event[key] for key in EVENT_FIELDS] for event in events]
    text = json.dumps(rows, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _canonical_arrays(arrays: tuple) -> list[dict[str, Any]]:
    """Retain exact integer fields and WFDB's logical Latin1 AUX byte mapping."""
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("WFDB numeric and AUX arrays have inconsistent lengths")
    events = []
    for index, values in enumerate(zip(*arrays, strict=True)):
        sample, code, subtype, channel, number, aux = values
        if not isinstance(aux, str):
            raise ValueError("WFDB AUX is not a logical byte string")
        payload = aux.encode("latin-1")
        events.append(
            {
                "event_index": index,
                "sample": int(sample),
                "type_code": int(code),
                "subtype": int(subtype),
                "channel": int(channel),
                "num": int(number),
                "aux_length": len(payload),
                "aux_hex": payload.hex(),
            }
        )
    return events


def _structural_header(record: Path, scratch: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read precisely three structural lines and never interpret clinical comments."""
    geometry = config["geometry"]
    with record.with_suffix(".hea").open() as stream:
        lines = [stream.readline().rstrip("\n") for _ in range(geometry["structural_header_lines"])]
    fields = lines[0].split()
    expected = [str(geometry["signals"]), str(geometry["fs"]), str(geometry["samples"])]
    if len(fields) < 4 or fields[0] != record.name or fields[1:4] != expected:
        raise ValueError("Header record identity or declared EDB geometry differs")
    if any(not line or line.startswith("#") for line in lines):
        raise ValueError("Missing structural signal metadata")
    scratch.mkdir(parents=True, exist_ok=False)
    sanitized = scratch / record.name
    sanitized.with_suffix(".hea").write_text("\n".join(lines) + "\n")
    sanitized.with_suffix(".atr").symlink_to(os.path.abspath(record.with_suffix(".atr")))
    header = wfdb.rdheader(str(sanitized))
    if header.n_sig != 2 or header.fs != 250 or header.sig_len != 1800000:
        raise ValueError("WFDB structural geometry disagrees")
    return {
        "signals": header.n_sig,
        "fs": header.fs,
        "samples": header.sig_len,
        "source_signal_names": list(header.sig_name),
        "sanitized_header_lines": len(lines),
    }


def decode_record(record: Path, scratch: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read structural headers and the complete source-bound WFDB annotation view.

    Parameters
    ----------
    record : Path
        Admitted annotation/header base path; no waveform is required or opened.
    scratch : Path
        Exclusive sanitized-header output directory.
    config : dict
        Frozen EDB geometry and annotation contract.

    Returns
    -------
    dict
        Transient complete canonical events and persistable metadata/digests.
        The caller must not serialize unrelated free-text AUX payloads.
    """
    metadata = _structural_header(record, scratch, config)
    payload = record.with_suffix(".atr").read_bytes()
    if len(payload) < 2 or len(payload) % 2 or payload[-2:] != b"\0\0":
        raise ValueError("MIT annotation byte count or final EOF marker is invalid")
    pairs = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 2)
    arrays = wfdb_annotation.proc_ann_bytes(pairs, None)
    events = _canonical_arrays(arrays)
    sample, codes, _, _, _, aux = arrays
    definitions, removed = wfdb_annotation.get_special_inds(np.asarray(sample), np.asarray(codes), aux)
    filtered = [event for event in events if event["event_index"] not in removed]
    effective, decoder_issues = _effective_view(scratch / record.name, events, definitions, filtered, config)
    metadata.update(
        record=record.name,
        annotation_bytes=len(payload),
        annotation_sha256=hashlib.sha256(payload).hexdigest(),
        event_count=len(events),
        canonical_event_sha256=event_digest(events),
        wfdb_filtered_event_count=len(filtered),
        wfdb_filtered_event_sha256=event_digest(filtered),
        wfdb_rdann_event_count=None if effective is None else len(effective),
        wfdb_rdann_event_sha256=None if effective is None else event_digest(effective),
        wfdb_effective_view_verified=not decoder_issues,
        wfdb_removed_indices=sorted(map(int, removed)),
        wfdb_potential_definition_indices=sorted(map(int, definitions)),
        complete_raw_target=True,
        independent_decoder_verified=False,
        decoder_issues=decoder_issues,
    )
    return {"events": events, "metadata": metadata}


def _definition_block_end(events: list[dict[str, Any]], index: int) -> int | None:
    """Bound the installed WFDB custom-definition scan before calling its unchecked loop."""
    while index < len(events):
        text = _aux(events[index])
        if text == "## end of definitions":
            return index + 1
        if not wfdb_annotation.rx_custom_label.findall(text):
            return None
        index += 1
    return None


def _wfdb_definition_safe(events: list[dict[str, Any]], definition_count: int) -> bool:
    """Detect installed WFDB's nonadvancing metadata loop using its actual prefix indexing."""
    index, fs = 0, None
    while index < definition_count:
        text = _aux(events[index])
        if not text.startswith("## "):
            index += 1
            continue
        matches = wfdb_annotation.rx_fs.findall(text)
        if not fs and matches:
            fs = float(matches[0])
            index += 1
            continue
        if text != "## annotation type definitions":
            return False
        end = _definition_block_end(events, index + 1)
        if end is None:
            return False
        index = end
    return True


def _effective_view(
    record: Path, events: list[dict[str, Any]], definitions: set, filtered: list, config: dict[str, Any]
) -> tuple[list | None, list]:
    """Compare real sanitized-path rdann while reporting known unsafe library metadata scans."""
    if not _wfdb_definition_safe(events, len(definitions)):
        return None, [{"kind": "WFDB_unsafe_definition_scan", "rdann_comparison_run": False}]
    annotation = wfdb.rdann(str(record), "atr", return_label_elements=["label_store"])
    arrays = (
        annotation.sample,
        annotation.label_store,
        annotation.subtype,
        annotation.chan,
        annotation.num,
        annotation.aux_note,
    )
    effective, issues = _canonical_arrays(arrays), []
    if event_digest(effective) != event_digest(filtered):
        issues.append({"kind": "WFDB_effective_view_differs_from_declared_filter"})
    if annotation.fs != config["geometry"]["fs"]:
        issues.append({"kind": "WFDB_annotation_time_resolution_differs"})
    return effective, issues


def _aux(event: dict[str, Any]) -> str:
    """Recover exact logical AUX characters without trimming or case folding."""
    payload = bytes.fromhex(event["aux_hex"])
    if len(payload) != event["aux_length"]:
        raise ValueError("Canonical AUX length disagrees with logical bytes")
    return payload.decode("latin-1")


def _issue(kind: str, event: dict[str, Any] | None = None, **details: Any) -> dict[str, Any]:
    """Describe an unresolved issue without emitting unrelated AUX contents."""
    result = {"kind": kind, **details}
    if event is not None:
        result.update(
            event_index=event["event_index"],
            sample=event["sample"],
            type_code=event["type_code"],
            aux_sha256=hashlib.sha256(bytes.fromhex(event["aux_hex"])).hexdigest(),
        )
    return result


def _definition(event: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    """Interpret only source-syntax definitions, retaining other sample-zero NOTE targets."""
    if event["sample"] != 0 or event["type_code"] != 22:
        return None
    text = _aux(event)
    if not text.startswith("##"):
        return None
    match = re.fullmatch(r"## time resolution: ([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return {
            "kind": "time_resolution",
            "event_index": event["event_index"],
            "value": float(match[1]),
            "valid": float(match[1]) == config["geometry"]["fs"],
        }
    if text in {"## annotation type definitions", "## end of definitions"}:
        return {
            "kind": "label_definition_boundary",
            "event_index": event["event_index"],
            "boundary": "begin" if text == "## annotation type definitions" else "end",
            "valid": True,
        }
    return {"kind": "unrecognized_definition", "event_index": event["event_index"], "valid": False}


def _custom_definition(event: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Retain definition identities and require exact known interpreted-code declarations."""
    match = re.fullmatch(r"([0-9]+) (\S+) (.+)", _aux(event), flags=re.ASCII)
    if match is None or event["sample"] != 0 or event["type_code"] != 22:
        return {"kind": "malformed_custom_label", "event_index": event["event_index"], "valid": False}
    code, symbol, description = int(match[1]), match[2], match[3]
    valid = True
    if code in config["annotation_codes"].values():
        standard = wfdb_annotation.ann_label_table.loc[code]
        valid = symbol == standard["symbol"] and description == standard["description"]
    return {
        "kind": "custom_label",
        "event_index": event["event_index"],
        "label_store": code,
        "symbol": symbol,
        "description_sha256": hashlib.sha256(description.encode("latin-1")).hexdigest(),
        "valid": bool(valid),
    }


def _definitions(events: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict, list]:
    """Track complete definition blocks without silently accepting orphan or changed entries."""
    definitions: dict[int, dict[str, Any]] = {}
    issues, active = [], False
    for event in events:
        definition = _definition(event, config)
        if definition is not None:
            boundary = definition.get("boundary")
            if boundary in {"begin", "end"}:
                if active == (boundary == "begin"):
                    issues.append(_issue("unbalanced_definition_boundary", event))
                active = boundary == "begin"
        elif active and event["type_code"] != 0:
            definition = _custom_definition(event, config)
        if definition is not None:
            definition["aux_sha256"] = hashlib.sha256(bytes.fromhex(event["aux_hex"])).hexdigest()
            definitions[event["event_index"]] = definition
            if not definition["valid"]:
                issues.append(_issue("unresolved_annotation_definition", event))
    if active:
        issues.append({"kind": "unclosed_definition_block"})
    return definitions, issues


def _candidate(event: dict[str, Any]) -> bool:
    """Identify narrow episode-like NOTE syntax without interpreting general comments."""
    return re.match(r"^(?:\(|[aA])?(?:[sS][tT]|[tT])(?:[0-9+\-)]|$)", _aux(event)) is not None


def _marker(event: dict[str, Any]) -> dict[str, Any] | None:
    """Match exact uppercase target or lowercase nuisance syntax on the proper code."""
    code, text = event["type_code"], _aux(event)
    if code not in {18, 19, 22} or (code == 22 and not _candidate(event)):
        return None
    axis = code == 22
    prefix = r"(?P<prefix>\(|a)?" if axis else r"(?P<prefix>\(|A)?"
    family = r"(?P<family>st|t)" if axis else r"(?P<family>ST|T)"
    pattern = (
        prefix + family + r"(?P<channel>[01])(?P<sign>\+\+|--|[+-])(?P<magnitude>[0-9]{3,4})?(?P<end>\))?"
    )
    match = re.fullmatch(pattern, text, flags=re.ASCII)
    if match is None:
        raise ValueError("unsupported_episode_candidate")
    values = match.groupdict()
    normalized_family = "ST" if values["family"] in {"ST", "st"} else "T"
    if not axis and code != (18 if normalized_family == "ST" else 19):
        raise ValueError("episode_family_type_mismatch")
    extreme = len(values["sign"]) == 2
    if extreme and normalized_family != "T":
        raise ValueError("double_sign_ST_not_allowed")
    kind = _marker_kind(values, extreme)
    return {
        "event_index": event["event_index"],
        "sample": event["sample"],
        "category": "axis" if axis else "real",
        "family": normalized_family,
        "channel": int(values["channel"]),
        "sign": values["sign"][0],
        "extreme": extreme,
        "kind": kind,
        "magnitude_microvolts": int(values["magnitude"]) if values["magnitude"] else None,
        "aux": text,
    }


def _marker_kind(values: dict[str, Any], extreme: bool) -> str:
    """Validate complete begin, peak or end forms without optional-field repair."""
    prefix, magnitude, end = values["prefix"], values["magnitude"], values["end"]
    if prefix == "(" and magnitude is None and end is None:
        return "begin"
    if prefix in {"A", "a"} and magnitude is not None and end is None and not extreme:
        return "peak"
    if prefix is None and magnitude is None and end == ")":
        return "end"
    raise ValueError("unsupported_episode_candidate")


def _key(marker: dict[str, Any]) -> tuple:
    """Keep category, family, channel, polarity and extreme state separate."""
    return tuple(marker[field] for field in ("category", "family", "channel", "sign", "extreme"))


def _close(begin: dict[str, Any], end: dict[str, Any], peaks: list[dict[str, Any]]) -> tuple:
    """Retain a paired interval and all its peaks; report unresolved admission failures."""
    start, stop = begin["sample"], end["sample"]
    issues = []
    if start >= stop:
        issues.append({"kind": "nonpositive_episode_duration", "event_index": end["event_index"]})
    if not begin["extreme"] and not peaks:
        issues.append({"kind": "missing_episode_peak", "event_index": begin["event_index"]})
    if any(not start <= peak["sample"] <= stop for peak in peaks):
        issues.append({"kind": "peak_outside_boundaries", "event_index": end["event_index"]})
    interval = {key: begin[key] for key in ("category", "family", "channel", "sign", "extreme")}
    interval.update(
        start=start,
        end=stop,
        begin_index=begin["event_index"],
        end_index=end["event_index"],
        peak_indices=[p["event_index"] for p in peaks],
        peak_samples=[p["sample"] for p in peaks],
        peak_microvolts=[p["magnitude_microvolts"] for p in peaks],
        valid=not issues,
    )
    return interval, issues


def _pair(markers: list[dict[str, Any]]) -> tuple[list, list]:
    """Pair per-key markers in original order, retaining every unresolved orphan."""
    active: dict[tuple, tuple] = {}
    intervals, issues = [], []
    for marker in markers:
        key = _key(marker)
        if marker["kind"] == "begin":
            if key in active:
                issues.append({"kind": "nested_same_key_begin", "event_index": marker["event_index"]})
            else:
                active[key] = (marker, [])
        elif key not in active:
            issues.append({"kind": "orphan_" + marker["kind"], "event_index": marker["event_index"]})
        elif marker["kind"] == "peak":
            active[key][1].append(marker)
        else:
            begin, peaks = active.pop(key)
            interval, found = _close(begin, marker, peaks)
            intervals.append(interval)
            issues.extend(found)
    issues.extend(
        {"kind": "unclosed_episode", "event_index": begin["event_index"]} for begin, _ in active.values()
    )
    return intervals, issues


def _extreme_parents(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require extreme T subinterval containment without inventing a missing parent."""
    issues = []
    for interval in intervals:
        if not interval["extreme"]:
            continue
        parents = [
            parent
            for parent in intervals
            if not parent["extreme"]
            and parent["valid"]
            and interval["start"] < interval["end"]
            and all(parent[key] == interval[key] for key in ("category", "family", "channel", "sign"))
            and parent["start"] <= interval["start"]
            and interval["end"] <= parent["end"]
        ]
        interval["parent_begin_indices"] = [parent["begin_index"] for parent in parents]
        interval["parent_count"] = len(parents)
        if not parents:
            issues.append(
                {
                    "kind": "extreme_T_parent_count",
                    "event_index": interval["begin_index"],
                    "parent_count": len(parents),
                }
            )
    return issues


def _quality(events: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list, list]:
    """Preserve whole two-channel quality states, including an UNKNOWN prefix."""
    spans, issues = [], []
    previous, state = 0, ["unknown", "unknown"]
    for event in events:
        if event["type_code"] != config["annotation_codes"]["NOISE"]:
            continue
        sample = event["sample"]
        if sample < previous:
            issues.append(_issue("decreasing_quality_sample", event))
        if sample > previous:
            spans.append({"start": previous, "end": sample, "states": list(state)})
        subtype = str(event["subtype"])
        if subtype not in config["quality"]["subtypes"]:
            state = ["unknown", "unknown"]
            issues.append(_issue("unknown_quality_subtype", event, subtype=event["subtype"]))
        else:
            state = list(config["quality"]["subtypes"][subtype])
        previous = sample
    if previous < config["geometry"]["samples"]:
        spans.append({"start": previous, "end": config["geometry"]["samples"], "states": state})
    return spans, issues


def _intersects(starts: np.ndarray, stops: np.ndarray, left: int, right: int) -> np.ndarray:
    """Use positive-length half-open overlap, preserving exact boundary contact."""
    return np.maximum(starts, left) < np.minimum(stops, right)


def _channel_windows(channel: int, intervals: list, quality: list, config: dict[str, Any]) -> tuple:
    """Compute one observed channel's fixed label/reason grid without waveform access."""
    grid, geometry = config["availability_grid"], config["geometry"]
    starts = np.arange(grid["windows_per_channel"]) * grid["stride_seconds"] * geometry["fs"]
    stops = starts + grid["window_seconds"] * geometry["fs"]
    reasons = np.zeros(len(starts), dtype=np.uint16)
    counts = np.zeros(len(starts), dtype=np.int32)
    contains = np.zeros(len(starts), dtype=np.int32)
    axis = np.zeros(len(starts), dtype=bool)
    margin = np.zeros(len(starts), dtype=bool)
    for span in quality:
        state = span["states"][channel]
        bit = {"unknown": 1, "n": 2, "u": 4, "c": 0}[state]
        reasons[_intersects(starts, stops, span["start"], span["end"])] |= bit
    for interval in [entry for entry in intervals if entry["channel"] == channel]:
        overlap = _intersects(starts, stops, interval["start"], interval["end"])
        if interval["category"] == "real" and interval["family"] == "ST":
            full = (interval["start"] <= starts) & (stops <= interval["end"])
            counts += overlap
            contains += full
            reasons[overlap & ~full] |= 8
        if interval["category"] == "axis":
            axis |= overlap
        amount = grid["reference_margin_seconds"] * geometry["fs"]
        margin |= _intersects(
            starts,
            stops,
            max(0, interval["start"] - amount),
            min(geometry["samples"], interval["end"] + amount),
        )
    clean = (reasons & 7) == 0
    positive = clean & (counts == 1) & (contains == 1) & ~axis
    reference = clean & ~margin
    reasons[counts > 1] |= 16
    reasons[axis] |= 32
    reasons[margin] |= 64
    labels = np.zeros(len(starts), dtype=np.uint8)
    labels[positive] = 1
    labels[reference & ~positive] = 2
    reasons[labels != 0] = 0
    return labels, reasons


def _validate(events: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Require complete ordered numeric event fields without dropping or coercing events."""
    issues = []
    for index, event in enumerate(events):
        fields = ("event_index", "sample", "type_code", "subtype", "channel", "num", "aux_length")
        if any(type(event[key]) is not int for key in fields) or event["event_index"] != index:
            raise ValueError("Canonical numeric event fields or event index are invalid")
        if not 0 <= event["sample"] < config["geometry"]["samples"]:
            issues.append(_issue("event_out_of_bounds", event))
        if not 0 <= event["channel"] <= 255:
            issues.append(_issue("invalid_WFDB_channel_metadata", event))
        _aux(event)
    return issues


def _selected(event: dict[str, Any], marker: dict[str, Any] | None) -> dict[str, Any]:
    """Retain selected numeric fields and only allowlisted episode AUX strings."""
    result = {
        key: event[key]
        for key in ("event_index", "sample", "type_code", "subtype", "channel", "num", "aux_length")
    }
    result["aux_sha256"] = hashlib.sha256(bytes.fromhex(event["aux_hex"])).hexdigest()
    if marker is not None:
        result["interpretation"] = marker
    return result


def _markers(events: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list, list, list, list]:
    """Select source-syntax targets/quality while retaining unresolved candidate identities."""
    selected, markers = [], []
    definitions, issues = _definitions(events, config)
    for event in events:
        if event["event_index"] in definitions:
            continue
        try:
            marker = _marker(event)
        except ValueError as error:
            issues.append(_issue(str(error), event))
            selected.append(_selected(event, None))
            continue
        if marker is not None:
            markers.append(marker)
        if marker is not None or event["type_code"] == config["annotation_codes"]["NOISE"]:
            selected.append(_selected(event, marker))
    return selected, markers, issues, list(definitions.values())


def qualify_events(events: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Apply frozen episode, quality and channel-local window rules to one record.

    Parameters
    ----------
    events : list
        Complete source-ordered canonical events, including raw WFDB-removed records.
    config : dict
        Committed base contract; the separate grid clarification is source-bound.

    Returns
    -------
    dict
        Selected events, original paired episodes/peaks, quality spans, issues,
        uint8 labels and uint16 reasons. A record issue makes every cell UNKNOWN128.
    """
    issues = _validate(events, config)
    contract = config["contract"]
    if contract["unknown_reason_bits"] != REASON_BITS or contract["shape"][1:] != [2, 720]:
        raise ValueError("Grid clarification differs from the frozen source contract")
    diagnostics = [
        {"kind": "global_sample_decrease", "event_index": right["event_index"]}
        for left, right in zip(events[:-1], events[1:], strict=True)
        if right["sample"] < left["sample"]
    ]
    selected, markers, found, definitions = _markers(events, config)
    issues.extend(found)
    intervals, found = _pair(markers)
    issues.extend(found)
    issues.extend(_extreme_parents(intervals))
    diagnostics.extend(
        {
            "kind": "multiple_containing_extreme_T_parents",
            "event_index": interval["begin_index"],
            "parent_begin_indices": interval["parent_begin_indices"],
            "parent_count": interval["parent_count"],
        }
        for interval in intervals
        if interval["extreme"] and interval["parent_count"] > 1
    )
    quality, found = _quality(events, config)
    issues.extend(found)
    shape = (config["geometry"]["signals"], config["availability_grid"]["windows_per_channel"])
    labels = np.zeros(shape, dtype=np.uint8)
    reasons = np.full(shape, 128, dtype=np.uint16)
    if not issues:
        for channel in range(shape[0]):
            labels[channel], reasons[channel] = _channel_windows(channel, intervals, quality, config)
    counts = {
        "events": len(events),
        "selected_events": len(selected),
        "issues": len(issues),
        "episodes": len(intervals),
        "base_episodes": sum(not e["extreme"] for e in intervals),
        "extreme_episodes": sum(e["extreme"] for e in intervals),
        "axis_episodes": sum(e["category"] == "axis" for e in intervals),
        "ST_positive_windows": int((labels == 1).sum()),
        "reference_windows": int((labels == 2).sum()),
        "unknown_windows": int((labels == 0).sum()),
    }
    overlaps = [
        {
            "interval_indices": [left, right],
            "cross_channel": intervals[left]["channel"] != intervals[right]["channel"],
            "cross_category": intervals[left]["category"] != intervals[right]["category"],
        }
        for left in range(len(intervals))
        for right in range(left + 1, len(intervals))
        if max(intervals[left]["start"], intervals[right]["start"])
        < min(intervals[left]["end"], intervals[right]["end"])
    ]
    return {
        "selected_events": selected,
        "episodes": intervals,
        "quality_spans": quality,
        "definitions": definitions,
        "issues": issues,
        "counts": counts,
        "diagnostics": diagnostics,
        "overlap_pairs": overlaps,
        "labels": labels,
        "reasons": reasons,
        "engineering_passed": not issues,
    }
