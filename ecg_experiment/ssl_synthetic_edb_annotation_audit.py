"""Independent lossless MIT events and observed-channel annotation qualification."""

import hashlib
import json
from bisect import bisect_right
from typing import Any

from ecg_experiment import ssl_synthetic_ludb_raw_events as raw_words

STANDARD_INTERPRETED_LABELS = {
    14: ("~", "Signal quality change"),
    18: ("s", "ST change"),
    19: ("T", "T-wave change"),
    22: ('"', "Comment annotation"),
}

CANONICAL_FIELDS = ("sample", "type_code", "subtype", "channel", "num", "aux_length", "aux_hex")


def decode(payload: bytes) -> dict[str, Any]:
    """Decode numeric MIT words independently of WFDB and retain exact logical AUX bytes.

    Parameters
    ----------
    payload : bytes
        One source-admitted annotation stream, never a waveform or clinical header.

    Returns
    -------
    dict
        Original events and metadata, complete byte identity, padding and explicit
        structural errors. No sorting, event deletion or boundary repair occurs.
    """
    original = raw_words.decode(payload)
    events = [
        {
            **{key: event[key] for key in CANONICAL_FIELDS[:-1]},
            "aux_hex": "",
            "byte_offset": event["byte_offset"],
            "raw_index": event["event_index"],
        }
        for event in original["events"]
    ]
    errors = list(original["errors"])
    offsets = [event["byte_offset"] for event in events]
    segments: dict[int, list[dict[str, Any]]] = {}
    for token in original["metadata"]:
        if token["type"] == "SKIP":
            continue
        if token["value"] > 255:
            errors.append({"kind": "metadata_value_exceeds_byte", "byte_offset": token["byte_offset"]})
        if token["type"] != "AUX":
            continue
        index = bisect_right(offsets, token["byte_offset"]) - 1
        if index < 0:
            continue
        first = token["byte_offset"] + 2
        last = first + token["value"]
        data = payload[first:last]
        padding = payload[last : last + token["value"] % 2]
        part = {
            "byte_offset": token["byte_offset"],
            "declared_bytes": token["value"],
            "logical_hex": data.hex(),
            "padding_hex": padding.hex(),
        }
        segments.setdefault(index, []).append(part)
        events[index]["aux_hex"] = data.hex()
        if len(data) != token["value"] or len(padding) != token["value"] % 2:
            errors.append({"kind": "incomplete_auxiliary_bytes", "byte_offset": token["byte_offset"]})
    for index, parts in segments.items():
        if len(parts) > 1:
            errors.append({"kind": "multiple_auxiliary_fields", "byte_offset": offsets[index]})
    return {
        "sha256": original["sha256"],
        "bytes": original["bytes"],
        "events": events,
        "metadata": original["metadata"],
        "metadata_counts": original["metadata_counts"],
        "auxiliary_segments": segments,
        "EOF": original["EOF"],
        "errors": errors,
    }


def canonical_events(events: list[dict[str, Any]]) -> list[list[Any]]:
    """Represent every event in original order using exact shared numeric and AUX fields.

    Parameters
    ----------
    events : list
        Full raw or explicitly filtered event view.

    Returns
    -------
    list
        Integer fields plus logical AUX hexadecimal text; order/multiplicity retained.
    """
    return [[event[key] for key in CANONICAL_FIELDS] for event in events]


def event_digest(events: list[dict[str, Any]]) -> str:
    """Hash the canonical event sequence without writing unrelated auxiliary contents.

    Parameters
    ----------
    events : list
        Complete ordered local event records.

    Returns
    -------
    str
        SHA256 of compact ASCII JSON canonical event tuples.
    """
    encoded = json.dumps(canonical_events(events), separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def decoder_views(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate the documented WFDB effective filter while keeping all raw target candidates.

    Parameters
    ----------
    events : list
        Every original MIT event, including sample-zero NOTE and code-zero records.

    Returns
    -------
    dict
        Exact removed indices and complete raw/effective digests; removed is not
        synonymous with harmless metadata or an instruction to discard targets.
    """
    removed = [
        index
        for index, event in enumerate(events)
        if event["type_code"] == 0 or event["sample"] == 0 and event["type_code"] == 22
    ]
    indices = set(removed)
    effective = [event for index, event in enumerate(events) if index not in indices]
    return {
        "full_raw_events": len(events),
        "wfdb_effective_events": len(effective),
        "removed_raw_indices": removed,
        "full_raw_digest": event_digest(events),
        "wfdb_effective_digest": event_digest(effective),
    }


def group_roles(records: list[str], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Construct document-derived linkage units and deterministic development roles.

    Parameters
    ----------
    records : list
        Unique official record roster, before annotation access.
    config : dict
        Frozen grouping and role rules.

    Returns
    -------
    list
        One assignment per original record, preserving roster order.
    """
    if len(records) != 90 or len(set(records)) != 90:
        raise ValueError("Require ninety unique official records")
    parent = {record: record for record in records}
    for group in config["grouping"]["explicit_groups"]:
        if not set(group) <= set(records):
            raise ValueError("Published linkage member absent")
        members = {parent[record] for record in group}
        representative = min(members)
        parent = {record: representative if unit in members else unit for record, unit in parent.items()}
    if len(set(parent.values())) != config["grouping"]["expected_document_components"]:
        raise ValueError("Document-derived component count differs")
    rule = config["roles"]
    result = []
    for record in records:
        unit = parent[record]
        key = f"{rule['namespace']}:{rule['seed']}:{unit}"
        bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % rule["bucket_count"]
        role = rule["names"][
            0 if bucket < rule["train_stop"] else 1 if bucket < rule["validation_stop"] else 2
        ]
        result.append({"record_id": record, "component_id": unit, "bucket": bucket, "role": role})
    return result


def selected_tag(event: dict[str, Any]) -> dict[str, Any] | None:
    """Interpret only exact allowed ASCII ST/T/axis grammar with AUX-defined channel.

    Parameters
    ----------
    event : dict
        One raw event with hexadecimal logical AUX bytes.

    Returns
    -------
    dict or None
        Parsed target marker, explicit invalid candidate, or unrelated event.
    """
    import re

    code = event["type_code"]
    if code not in (18, 19, 22):
        return None
    payload = bytes.fromhex(event["aux_hex"])
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError:
        return {"error": "non_ASCII_candidate"} if code in (18, 19) else None
    expected = {18: ("real", "ST"), 19: ("real", "T")}
    forms = (
        ("begin", r"\((ST|T|st|t)([01])([+-]{1,2})"),
        ("peak", r"([Aa])(ST|T|st|t)([01])([+-])([0-9]{3,4})"),
        ("end", r"(ST|T|st|t)([01])([+-]{1,2})\)"),
    )
    for kind, pattern in forms:
        match = re.fullmatch(pattern, text)
        if match is None:
            continue
        groups = match.groups()
        if kind == "peak":
            prefix, family, channel, sign, magnitude = groups
            if prefix != ("A" if family.isupper() else "a"):
                return {"error": "wrong_peak_case"}
        else:
            family, channel, sign = groups
            magnitude = None
        category = "real" if family.isupper() else "axis"
        if code == 22 and category != "axis" or code in expected and expected[code] != (category, family):
            return {"error": "candidate_code_case_or_family_mismatch"}
        if sign not in ("+", "-", "++", "--") or len(sign) == 2 and family.upper() != "T":
            return {"error": "unsupported_sign_or_extreme_family"}
        return {
            "category": category,
            "family": family.upper(),
            "channel": int(channel),
            "sign": sign[0],
            "extreme": len(sign) == 2,
            "kind": kind,
            "magnitude": int(magnitude) if magnitude is not None else None,
            "text": text,
        }
    if code in (18, 19) or re.match(r"(?:\(|[aA])?(?:[sS][tT]|[tT])(?:[0-9+\-)]|$)", text):
        return {"error": "unsupported_candidate_string"}
    return None


def definition_entry(text: str, index: int, block: bool) -> tuple[Any, list, bool]:
    """Classify one definition candidate without interpreting unrelated NOTE text.

    Parameters
    ----------
    text : str
        Lossless Latin1 representation of logical AUX bytes.
    index : int
        Original event index.
    block : bool
        Whether a recognized custom-definition block is active.

    Returns
    -------
    tuple
        Recognized declaration, explicit issues and updated block state.
    """
    import re

    fs = re.fullmatch(r"## time resolution: ([0-9]+(?:\.[0-9]+)?)", text)
    if fs:
        issues = (
            [] if float(fs[1]) == 250 else [{"kind": "definition_time_resolution_differs", "index": index}]
        )
        return {"kind": "time_resolution", "value": float(fs[1])}, issues, block
    if text in ("## annotation type definitions", "## end of definitions"):
        opening = text == "## annotation type definitions"
        issues = (
            [{"kind": "nested_definition_block" if opening else "orphan_definition_end", "index": index}]
            if opening == block
            else []
        )
        return {"kind": "definition_boundary"}, issues, opening
    if block:
        match = re.fullmatch(r"([0-9]+) (\S+) (.+)", text)
        if match is None:
            return None, [{"kind": "invalid_definition_entry", "index": index}], block
        code = int(match[1])
        issues = (
            [{"kind": "unresolved_interpreted_code_definition", "index": index}]
            if code in STANDARD_INTERPRETED_LABELS
            and (match[2], match[3]) != STANDARD_INTERPRETED_LABELS[code]
            else []
        )
        return (
            {
                "kind": "custom_label",
                "code": code,
                "content_sha256": hashlib.sha256(text.encode("latin1")).hexdigest(),
            },
            issues,
            block,
        )
    return (
        None,
        ([{"kind": "unsupported_definition_syntax", "index": index}] if text.startswith("##") else []),
        block,
    )


def classify_definitions(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Recognize actual sample-zero definitions without discarding arbitrary NOTE markers.

    Parameters
    ----------
    events : list
        Complete ordered raw stream.

    Returns
    -------
    dict
        Recognized metadata indices, declarations and explicit unsupported definitions.
    """
    indices, declarations, issues = [], [], []
    block = False
    for index, event in enumerate(events):
        if event["sample"] != 0 or event["type_code"] != 22:
            continue
        text = bytes.fromhex(event["aux_hex"]).decode("latin1")
        declaration, errors, block = definition_entry(text, index, block)
        issues.extend(errors)
        if declaration is not None:
            indices.append(index)
            declarations.append({"index": index, **declaration})
    if block:
        issues.append({"kind": "unclosed_definition_block"})
    return {"indices": indices, "declarations": declarations, "issues": issues}


def append_marker(marker: dict[str, Any], active: dict, intervals: list, issues: list) -> None:
    """Advance one keyed state in original event order without boundary repair.

    Parameters
    ----------
    marker : dict
        Parsed target marker including original index and sample.
    active : dict
        Open intervals by category/family/channel/polarity/extreme key.
    intervals : list
        Destination for complete original interval records.
    issues : list
        Destination for every unresolved boundary or peak issue.
    """
    index, sample = marker["index"], marker["sample"]
    key = tuple(marker[field] for field in ("category", "family", "channel", "sign", "extreme"))
    if marker["kind"] == "begin":
        if key in active:
            issues.append({"kind": "nested_begin", "index": index})
        else:
            active[key] = {
                **{field: marker[field] for field in ("category", "family", "channel", "sign", "extreme")},
                "start": sample,
                "begin_index": index,
                "peak_indices": [],
                "peak_samples": [],
                "peak_microvolts": [],
            }
    elif key not in active:
        issues.append({"kind": "orphan_" + marker["kind"], "index": index})
    elif marker["kind"] == "peak":
        active[key]["peak_indices"].append(index)
        active[key]["peak_samples"].append(sample)
        active[key]["peak_microvolts"].append(marker["magnitude"])
    else:
        previous_issues = len(issues)
        interval = active.pop(key)
        interval.update({"end": sample, "end_index": index})
        intervals.append(interval)
        if sample <= interval["start"]:
            issues.append({"kind": "nonpositive_interval", "index": index})
        if not marker["extreme"] and not interval["peak_indices"]:
            issues.append({"kind": "base_interval_without_peak", "index": index})
        if any(not interval["start"] <= peak <= sample for peak in interval["peak_samples"]):
            issues.append({"kind": "peak_outside_interval", "index": index})
        interval["valid"] = len(issues) == previous_issues


def interval_relationships(intervals: list, issues: list, diagnostics: list) -> list:
    """Preserve overlap descriptors and reject unparented extreme subintervals.

    Parameters
    ----------
    intervals : list
        Complete paired intervals in closure order.
    issues : list
        Mutable destination for unresolved extreme-parent issues.
    diagnostics : list
        Destination for multiple valid parent matches, without another eligibility gate.

    Returns
    -------
    list
        All positive-length pairwise overlaps with channel/category indicators.
    """
    for interval in intervals:
        if not interval["extreme"]:
            continue
        parents = [
            base["begin_index"]
            for base in intervals
            if base["valid"]
            and not base["extreme"]
            and all(interval[field] == base[field] for field in ("category", "family", "channel", "sign"))
            and base["start"] <= interval["start"] < interval["end"] <= base["end"]
        ]
        interval["parent_begin_indices"] = parents
        interval["parent_count"] = len(parents)
        if not parents:
            issues.append({"kind": "extreme_without_containing_base", "index": interval["begin_index"]})
        if len(parents) > 1:
            diagnostics.append(
                {
                    "kind": "multiple_containing_bases",
                    "index": interval["begin_index"],
                    "parent_begin_indices": parents,
                }
            )
    overlaps = []
    for first, left in enumerate(intervals):
        for second in range(first + 1, len(intervals)):
            right = intervals[second]
            if max(left["start"], right["start"]) < min(left["end"], right["end"]):
                overlaps.append(
                    {
                        "interval_indices": [first, second],
                        "cross_channel": left["channel"] != right["channel"],
                        "cross_category": left["category"] != right["category"],
                    }
                )
    return overlaps


def append_quality(event: dict, index: int, quality: list, issues: list) -> None:
    """Retain each quality update and reject invalid state or chronological partition.

    Parameters
    ----------
    event : dict
        One original numeric event.
    index : int
        Its preserved raw index.
    quality : list
        Destination for all updates, including coincident ones.
    issues : list
        Destination for unknown-state or decreasing-update issues.
    """
    if event["type_code"] != 14:
        return
    lookup = {0: "cc", 1: "nc", 2: "cn", 3: "nn", 17: "uc", 18: "un", 32: "cu", 33: "nu", 51: "uu"}
    sample = event["sample"]
    if quality and sample < quality[-1]["sample"]:
        issues.append({"kind": "decreasing_quality_update", "index": index})
    state = lookup.get(event["subtype"], "??")
    quality.append({"index": index, "sample": sample, "subtype": event["subtype"], "state": state})
    if state == "??":
        issues.append({"kind": "unknown_quality_subtype", "index": index})


def episodes(events: list[dict[str, Any]], samples: int = 1800000) -> dict[str, Any]:
    """Pair independent keyed marker streams without repairing boundaries or event order.

    Parameters
    ----------
    events : list
        Complete raw event stream, including WFDB-filtered sample-zero NOTE markers.
    samples : int
        Declared recording length.

    Returns
    -------
    dict
        Selected markers, paired intervals, overlaps, quality events and all issues.
    """
    definitions = classify_definitions(events)
    excluded = set(definitions["indices"])
    issues = list(definitions["issues"])
    selected, intervals, quality = [], [], []
    active: dict[tuple[Any, ...], dict[str, Any]] = {}
    previous = -1
    diagnostics = []
    for index, event in enumerate(events):
        sample = event["sample"]
        if not 0 <= sample < samples:
            issues.append({"kind": "event_outside_record", "index": index})
        if sample < previous:
            diagnostics.append({"kind": "decreasing_event_sample", "index": index})
        previous = sample
        append_quality(event, index, quality, issues)
        if index in excluded:
            continue
        tag = selected_tag(event)
        if tag is None:
            continue
        if "error" in tag:
            issues.append({"kind": tag["error"], "index": index})
            selected.append({"index": index, "sample": sample, **tag})
            continue
        marker = {"index": index, "sample": sample, **tag}
        selected.append(marker)
        append_marker(marker, active, intervals, issues)
    for interval in active.values():
        issues.append({"kind": "unclosed_interval", "index": interval["begin_index"]})
    overlaps = interval_relationships(intervals, issues, diagnostics)
    return {
        "selected": selected,
        "intervals": intervals,
        "overlaps": overlaps,
        "quality": quality,
        "diagnostics": diagnostics,
        "definitions": definitions,
        "issues": issues,
    }


def quality_spans(quality: list[dict[str, Any]], samples: int) -> list[dict[str, Any]]:
    """Retain exact quality states on positive-duration spans including unknown prefix.

    Parameters
    ----------
    quality : list
        Original-order quality updates, preserving coincident-event order.
    samples : int
        Recording length.

    Returns
    -------
    list
        Nonoverlapping complete record partition; zero-duration updates remain in input.
    """
    result = []
    start, state = 0, "??"
    for event in quality:
        if event["sample"] > start:
            result.append({"start": start, "end": event["sample"], "state": state})
        start, state = event["sample"], event["state"]
    if start < samples:
        result.append({"start": start, "end": samples, "state": state})
    return result


def window_grid(
    parsed: dict[str, Any], samples: int = 1800000, width: int = 2500, margin: int = 7500
) -> tuple[Any, Any]:
    """Independently label each observed channel/window using exact exclusion reason bits.

    Parameters
    ----------
    parsed : dict
        Independently paired intervals, quality updates and record-level issues.
    samples : int
        Complete record duration in samples.
    width : int
        Nonoverlapping window duration.
    margin : int
        Reference exclusion expansion in samples.

    Returns
    -------
    tuple
        uint8 labels and uint16 reasons, both channel by complete window index.
    """
    import numpy as np

    if samples % width:
        raise ValueError("Require complete fixed windows")
    labels = np.zeros((2, samples // width), dtype=np.uint8)
    reasons = np.zeros(labels.shape, dtype=np.uint16)
    if parsed["issues"]:
        reasons.fill(128)
        return labels, reasons
    spans = quality_spans(parsed["quality"], samples)
    for channel in range(2):
        intervals = [item for item in parsed["intervals"] if item["channel"] == channel]
        for column, start in enumerate(range(0, samples, width)):
            end = start + width
            bad_quality = 0
            for span in spans:
                if max(start, span["start"]) < min(end, span["end"]):
                    bad_quality |= {"?": 1, "n": 2, "u": 4, "c": 0}[span["state"][channel]]
            st = [
                item
                for item in intervals
                if item["category"] == "real"
                and item["family"] == "ST"
                and max(start, item["start"]) < min(end, item["end"])
            ]
            axis = any(
                item["category"] == "axis" and max(start, item["start"]) < min(end, item["end"])
                for item in intervals
            )
            partial = any(not item["start"] <= start < end <= item["end"] for item in st)
            expanded = any(
                max(start, max(0, item["start"] - margin)) < min(end, min(samples, item["end"] + margin))
                for item in intervals
            )
            positive = len(st) == 1 and not partial and not axis and not bad_quality
            reference = not expanded and not bad_quality
            if positive or reference:
                labels[channel, column] = 1 if positive else 2
            else:
                reasons[channel, column] = (
                    bad_quality
                    | (8 if partial else 0)
                    | (16 if len(st) > 1 else 0)
                    | (32 if axis else 0)
                    | (64 if expanded else 0)
                )
    return labels, reasons


def structural_header(lines: list[str], record: str) -> dict[str, Any]:
    """Check only three structural header lines independently of WFDB.

    Parameters
    ----------
    lines : list
        Exactly the first three source lines; no clinical or demographic comments.
    record : str
        Admitted official record name.

    Returns
    -------
    dict
        Exact two-channel geometry and literal signal descriptions.
    """
    if len(lines) != 3 or any(not line.strip() or line.startswith("#") for line in lines):
        raise ValueError("Require exactly three structural header lines")
    fields = lines[0].split()
    if fields[:4] != [record, "2", "250", "1800000"]:
        raise ValueError("Structural recording geometry or identity differs")
    signals = [line.split() for line in lines[1:]]
    if any(len(tokens) < 9 for tokens in signals):
        raise ValueError("Signal structural fields are incomplete")
    return {
        "signals": 2,
        "fs": 250,
        "samples": 1800000,
        "source_signal_names": [" ".join(tokens[8:]) for tokens in signals],
        "sanitized_header_lines": 3,
    }
