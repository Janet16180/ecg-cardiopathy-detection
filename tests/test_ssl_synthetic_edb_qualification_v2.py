"""Artificial event contracts for the separate EDB grammar and quality views."""

import copy
import hashlib
import json

import numpy as np
import pytest

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_edb_qualification as frozen
from ecg_experiment import ssl_synthetic_edb_qualification_v2 as core


def config():
    result = json.loads((ROOT / frozen.CONFIG).read_text())
    result["contract"] = json.loads((ROOT / frozen.CONTRACT).read_text())
    result["semantic_recovery"] = dict(core.SEMANTIC_RECOVERY)
    return result


def events(*rows):
    result = []
    for index, row in enumerate(rows):
        sample, code, aux, *subtype = row
        payload = aux.encode("latin1")
        result.append(
            {
                "event_index": index,
                "sample": sample,
                "type_code": code,
                "subtype": subtype[0] if subtype else 0,
                "channel": 1,
                "num": -2,
                "aux_length": len(payload),
                "aux_hex": payload.hex(),
            }
        )
    return result


def episode_rows(family="ST", channel=0, axis=False, nul=""):
    code = 22 if axis else (18 if family == "ST" else 19)
    family, peak = (family.lower(), "a") if axis else (family, "A")
    return [
        (0, code, f"({family}{channel}+{nul}"),
        (2500, code, f"{peak}{family}{channel}+100{nul}"),
        (10000, code, f"{family}{channel}+){nul}"),
    ]


def assert_record_issue(result, kind):
    assert kind in {issue["kind"] for issue in result["issues"]}
    assert not result["engineering_passed"]
    assert np.all(result["labels"] == 0)
    assert np.all(result["reasons"] == 128)
    assert result["counts"]["issues"] == len(result["issues"])
    assert result["counts"]["unknown_windows"] == 1440
    assert result["counts"]["ST_positive_windows"] == 0
    assert result["counts"]["reference_windows"] == 0


@pytest.mark.parametrize(("family", "axis"), [("ST", False), ("T", False), ("ST", True), ("T", True)])
def test_terminal_nul_changes_only_grammar_view_and_selected_raw_metadata(family, axis):
    raw = events((0, 14, ""), *episode_rows(family, axis=axis, nul="\0"))
    original = copy.deepcopy(raw)
    policy = config()
    original_policy = copy.deepcopy(policy)
    expected = frozen.qualify_events(events((0, 14, ""), *episode_rows(family, axis=axis)), policy)
    result = core.qualify_events(raw, policy)
    assert result["engineering_passed"]
    assert raw == original
    assert policy == original_policy
    assert result["counts"] == expected["counts"]
    assert result["episodes"] == expected["episodes"]
    assert result["definitions"] == expected["definitions"]
    np.testing.assert_array_equal(result["labels"], expected["labels"])
    np.testing.assert_array_equal(result["reasons"], expected["reasons"])
    assert result["grammar_view"] == {
        "episode_candidates": 3,
        "single_terminal_nul_removed": 3,
        "invalid_nul_candidates": 0,
        "raw_event_count": 4,
        "raw_canonical_event_sha256": frozen.event_digest(raw),
        "raw_events_changed": False,
        "metadata_aux_changed": False,
    }
    for selected in result["selected_events"][1:]:
        source = raw[selected["event_index"]]
        assert selected["aux_length"] == source["aux_length"]
        assert selected["aux_sha256"] == hashlib.sha256(bytes.fromhex(source["aux_hex"])).hexdigest()
        assert selected["raw_aux_hex"] == source["aux_hex"]
        assert selected["grammar_aux_terminal_nul_removed"]
        assert "\0" not in selected["interpretation"]["aux"]
    assert frozen.event_digest(raw) == frozen.event_digest(original)


def test_exact_legacy_episode_qualifies_without_normalization():
    raw = events((0, 14, ""), *episode_rows())
    result = core.qualify_events(raw, config())
    assert result["engineering_passed"]
    assert result["grammar_view"]["single_terminal_nul_removed"] == 0
    assert all(not item.get("grammar_aux_terminal_nul_removed") for item in result["selected_events"])


@pytest.mark.parametrize("aux", ["(ST0+\0\0", "(ST0\0+", "\0(ST0+", "(ST0+\0x\0"])
@pytest.mark.parametrize("code", [18, 19])
def test_malformed_target_nul_fails_record_with_original_selected_identity(aux, code):
    raw = events((0, 14, ""), (100, code, aux))
    result = core.qualify_events(raw, config())
    assert_record_issue(result, "invalid_episode_AUX_NUL")
    assert result["grammar_view"]["single_terminal_nul_removed"] == 0
    assert result["selected_events"][-1]["raw_aux_hex"] == raw[-1]["aux_hex"]


@pytest.mark.parametrize("aux", ["(st0+\0\0", "(st0\0+", "(st0+\0x\0"])
def test_malformed_axis_candidate_nul_fails_record(aux):
    result = core.qualify_events(events((0, 22, aux)), config())
    assert_record_issue(result, "invalid_episode_AUX_NUL")


@pytest.mark.parametrize("suffix", [" ", "\n", "\r", "\xff", "\0 ", " \0"])
def test_non_nul_suffix_is_never_repaired_or_trimmed(suffix):
    raw = events((0, 14, ""), (100, 18, "(ST0+" + suffix))
    result = core.qualify_events(raw, config())
    assert_record_issue(result, "unsupported_episode_candidate")
    assert not result["episodes"]
    assert result["selected_events"][-1]["raw_aux_hex"] == raw[-1]["aux_hex"]


def test_definitions_and_unrelated_aux_remain_exact_raw_and_unexposed():
    secret = "Unrelated free text\0"
    raw = events(
        (0, 22, "## time resolution: 250\0"),
        (0, 22, "## annotation type definitions"),
        (0, 22, "18 s ST change"),
        (0, 22, "## end of definitions"),
        (10, 1, secret),
        (20, 22, secret),
        (30, 14, secret),
    )
    result = core.qualify_events(raw, config())
    definitions, issues = frozen._definitions(raw, config())
    assert_record_issue(result, "unresolved_annotation_definition")
    assert result["definitions"] == list(definitions.values())
    assert all(issue in result["issues"] for issue in issues)
    assert result["grammar_view"]["episode_candidates"] == 0
    assert result["grammar_view"]["single_terminal_nul_removed"] == 0
    saved = {key: value for key, value in result.items() if key not in {"labels", "reasons"}}
    assert secret not in json.dumps(saved)
    assert secret.encode("latin1").hex() not in json.dumps(saved)


def test_episode_like_custom_definition_is_not_normalized():
    raw = events(
        (0, 22, "## annotation type definitions"),
        (0, 22, "(st0+\0"),
        (0, 22, "## end of definitions"),
    )
    result = core.qualify_events(raw, config())
    assert_record_issue(result, "unresolved_annotation_definition")
    assert result["grammar_view"]["episode_candidates"] == 0
    assert result["selected_events"] == []


@pytest.mark.parametrize("subtype", range(256))
def test_quality_domain_exactly_matches_two_channel_mask_with_unreadable_priority(subtype):
    policy = config()
    original = copy.deepcopy(policy)
    raw = events((0, 14, "", subtype))
    result = core.qualify_events(raw, policy)
    assert policy == original
    if subtype & ~0x33:
        assert_record_issue(result, "unknown_quality_subtype")
        assert result["quality_spans"][0]["states"] == ["unknown", "unknown"]
        return
    expected = [
        "u" if subtype & (0x10 << channel) else "n" if subtype & (1 << channel) else "c" for channel in (0, 1)
    ]
    assert result["engineering_passed"]
    assert result["quality_spans"] == [{"start": 0, "end": 1800000, "states": expected}]
    for channel, state in enumerate(expected):
        assert np.all(result["labels"][channel] == (2 if state == "c" else 0))
        assert np.all(result["reasons"][channel] == {"c": 0, "n": 2, "u": 4}[state])


@pytest.mark.parametrize("subtype", [-129, -1, 256, 65536])
def test_quality_signed_or_extra_channel_values_are_never_masked_into_domain(subtype):
    result = core.qualify_events(events((0, 14, "", subtype)), config())
    assert_record_issue(result, "unknown_quality_subtype")


def test_quality_unknown_prefix_and_original_same_sample_event_order_are_preserved():
    result = core.qualify_events(events((2500, 14, "", 0), (2500, 14, "", 16)), config())
    assert result["engineering_passed"]
    assert result["quality_spans"] == [
        {"start": 0, "end": 2500, "states": ["unknown", "unknown"]},
        {"start": 2500, "end": 1800000, "states": ["u", "c"]},
    ]
    assert np.all(result["reasons"][:, 0] == 1)
    assert np.all(result["reasons"][0, 1:] == 4)
    assert np.all(result["labels"][1, 1:] == 2)


@pytest.mark.parametrize("key", list(core.SEMANTIC_RECOVERY))
def test_successor_policy_must_be_exactly_declared(key):
    policy = config()
    policy["semantic_recovery"][key] = "drift"
    with pytest.raises(ValueError, match="exact prospective"):
        core.qualify_events([], policy)


def test_successor_policy_missing_or_additional_fields_are_rejected():
    policy = config()
    del policy["semantic_recovery"]
    with pytest.raises(ValueError, match="exact prospective"):
        core.qualify_events([], policy)
    policy = config()
    policy["semantic_recovery"]["trim_whitespace"] = True
    with pytest.raises(ValueError, match="exact prospective"):
        core.qualify_events([], policy)


def test_canonical_validation_runs_on_original_events_before_text_view():
    raw = events((100, 18, "(ST0+\0"))
    raw[0]["aux_length"] -= 1
    with pytest.raises(ValueError, match="AUX length"):
        core.qualify_events(raw, config())


def test_extreme_t_parent_semantics_survive_terminal_nul_view():
    raw = events(
        (0, 14, ""),
        *episode_rows("T", nul="\0"),
        *episode_rows("T", nul="\0"),
        (2500, 19, "(T0++\0"),
        (5000, 19, "T0++)\0"),
    )
    result = core.qualify_events(raw, config())
    assert result["engineering_passed"]
    assert result["episodes"][-1]["parent_count"] == 2
    assert result["episodes"][-1]["parent_begin_indices"] == [1, 4]
    assert result["counts"]["events"] == len(raw)


def test_sample_zero_axis_and_code_zero_raw_events_retain_original_indices():
    raw = events((0, 14, ""), (0, 0, ""), *episode_rows(axis=True, nul="\0"), (10001, 1, ""))
    result = core.qualify_events(raw, config())
    assert result["engineering_passed"]
    assert result["counts"]["events"] == 6
    assert result["episodes"][0]["begin_index"] == 2
    assert result["episodes"][0]["peak_indices"] == [3]
    assert result["episodes"][0]["end_index"] == 4
