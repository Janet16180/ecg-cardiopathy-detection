"""Qualify an explicit episode text view while preserving raw EDB event identities."""

import hashlib
from typing import Any

from ecg_experiment import ssl_synthetic_edb_qualification as frozen

SEMANTIC_RECOVERY = {
    "id": "ssl_synthetic_edb_semantic_recovery_v2",
    "episode_terminal_nul": "remove_exactly_one_terminal_NUL_only_if_remaining_bytes_contain_no_NUL",
    "definitions": "raw_unchanged",
    "quality_policy": "two_channel_WFDB_bits_mask_0x33_unreadable_precedes_noisy",
}
QUALITY_SUBTYPES = {
    "0": "cc",
    "1": "nc",
    "2": "cn",
    "3": "nn",
    "16": "uc",
    "17": "uc",
    "18": "un",
    "19": "un",
    "32": "cu",
    "33": "nu",
    "34": "cu",
    "35": "nu",
    "48": "uu",
    "49": "uu",
    "50": "uu",
    "51": "uu",
}
SOURCES = (
    *frozen.SOURCES,
    "ecg_experiment/ssl_synthetic_edb_qualification_v2.py",
    "tests/test_ssl_synthetic_edb_qualification_v2.py",
)


def _operating_config(config: dict[str, Any]) -> dict[str, Any]:
    """Require the explicit successor policy and copy its source-derived quality map."""
    if config.get("semantic_recovery") != SEMANTIC_RECOVERY:
        raise ValueError("Require the exact prospective EDB semantic recovery v2 policy")
    quality = {**config["quality"], "subtypes": dict(QUALITY_SUBTYPES)}
    return {**config, "quality": quality}


def _episode_candidate(event: dict[str, Any], definitions: dict[int, dict[str, Any]]) -> bool:
    """Restrict the text view to episode candidates outside raw definition records."""
    if event["event_index"] in definitions:
        return False
    return event["type_code"] in {18, 19} or (event["type_code"] == 22 and frozen._candidate(event))


def _grammar_view(events: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list, dict, list]:
    """Remove exactly one terminal NUL from candidate AUX without changing raw events."""
    definitions, _ = frozen._definitions(events, config)
    view, candidates, issues = [], {}, []
    for event in events:
        value = dict(event)
        if _episode_candidate(event, definitions):
            payload = bytes.fromhex(event["aux_hex"])
            removed = payload.endswith(b"\0") and payload.count(b"\0") == 1
            candidates[event["event_index"]] = removed
            if b"\0" in payload and not removed:
                issues.append(frozen._issue("invalid_episode_AUX_NUL", event))
            if removed:
                value.update(aux_length=len(payload) - 1, aux_hex=payload[:-1].hex())
        view.append(value)
    return view, candidates, issues


def _restore_selected(selected: list[dict[str, Any]], events: list, candidates: dict) -> None:
    """Restore raw AUX identity and retain candidate bytes alongside grammar interpretation."""
    for item in selected:
        event = events[item["event_index"]]
        payload = bytes.fromhex(event["aux_hex"])
        item["aux_length"] = event["aux_length"]
        item["aux_sha256"] = hashlib.sha256(payload).hexdigest()
        if item["event_index"] in candidates:
            item["raw_aux_hex"] = event["aux_hex"]
            item["grammar_aux_terminal_nul_removed"] = candidates[item["event_index"]]


def _record_issues(result: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """Make malformed text-view payloads fail the complete original record."""
    if not issues:
        return
    result["issues"].extend(issues)
    result["labels"].fill(0)
    result["reasons"].fill(frozen.REASON_BITS["record_issue"])
    result["engineering_passed"] = False
    result["counts"].update(
        issues=len(result["issues"]),
        ST_positive_windows=0,
        reference_windows=0,
        unknown_windows=result["labels"].size,
    )


def _restore_issue_identities(issues: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    """Bind any issue payload digest to the original raw AUX rather than its text view."""
    for issue in issues:
        if "aux_sha256" in issue:
            event = events[issue["event_index"]]
            issue["aux_sha256"] = hashlib.sha256(bytes.fromhex(event["aux_hex"])).hexdigest()


def qualify_events(events: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Apply frozen qualification to a separate, narrowly defined episode AUX text view.

    Parameters
    ----------
    events : list
        Complete source-ordered canonical raw events. No input event is mutated,
        removed or renumbered; raw decoder and WFDB comparison views stay separate.
    config : dict
        Source-bound frozen configuration with the exact ``semantic_recovery``
        successor policy. Quality subtypes are expanded in a separate config copy.

    Returns
    -------
    dict
        Frozen qualification results with raw selected AUX identities, explicit
        candidate text-view metadata and the complete original event digest.
        Definitions and unrelated comments are never normalized.
    """
    config = _operating_config(config)
    frozen._validate(events, config)
    view, candidates, issues = _grammar_view(events, config)
    result = frozen.qualify_events(view, config)
    _restore_selected(result["selected_events"], events, candidates)
    _record_issues(result, issues)
    _restore_issue_identities(result["issues"], events)
    result["grammar_view"] = {
        "episode_candidates": len(candidates),
        "single_terminal_nul_removed": sum(candidates.values()),
        "invalid_nul_candidates": len(issues),
        "raw_event_count": len(events),
        "raw_canonical_event_sha256": frozen.event_digest(events),
        "raw_events_changed": False,
        "metadata_aux_changed": False,
    }
    return result
