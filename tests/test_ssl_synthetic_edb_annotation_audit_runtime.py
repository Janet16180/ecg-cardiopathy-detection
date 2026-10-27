"""Artificial metadata liveness and complete-region independent reconstruction checks."""

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_edb_annotation_audit_run as audit


def event(sample: int, code: int, text: str = "") -> dict:
    """Construct one original-order artificial event without source asset access."""
    return {
        "sample": sample,
        "type_code": code,
        "subtype": 0,
        "channel": 0,
        "num": 0,
        "aux_length": len(text),
        "aux_hex": text.encode().hex(),
    }


@pytest.mark.parametrize(
    ("strings", "safe"),
    [
        (["ordinary comment"], True),
        (["## unknown definition"], False),
        (["## end of definitions"], False),
        (["## time resolution: 250", "## time resolution: 250"], False),
        (["## annotation type definitions", "18 s ST change", "## end of definitions"], True),
        (["## annotation type definitions", "bad", "## end of definitions"], False),
    ],
)
def test_installed_prefix_loop_liveness_is_proven_without_decoder(strings: list, safe: bool) -> None:
    """Known nonadvancing or unbounded cases cannot be called a successful comparison."""
    assert audit.safe_effective_decode([event(0, 22, text) for text in strings]) is safe


def test_library_liveness_uses_prefix_indices_not_note_positions() -> None:
    """A negative-SKIP ordering case follows the installed loop's actual argument indexing."""
    values = [event(100, 1, "## unknown definition"), event(0, 22, "ordinary comment")]
    assert not audit.safe_effective_decode(values)
    values[0] = event(100, 1, "ordinary")
    values[1] = event(0, 22, "## unknown definition")
    assert audit.safe_effective_decode(values)


def test_malformed_custom_region_never_becomes_an_interpreted_episode() -> None:
    """Keep malformed entries as issues, excluding them from target pairing without repair."""
    values = [
        event(0, 22, "## annotation type definitions"),
        event(0, 18, "(ST0+"),
        event(0, 22, "## end of definitions"),
    ]
    result = audit.reconstruct(values, [])
    assert result["intervals"] == []
    assert result["selected"] == []
    assert result["definitions"]["indices"] == [0, 1, 2]
    assert any(item["kind"] == "nonmetadata_custom_definition_entry" for item in result["issues"])
    assert np.all(result["labels"] == 0)
    assert np.all(result["reasons"] == 128)


def test_binary_failure_and_library_comparison_failure_remain_distinct() -> None:
    """No source decoder failure may be hidden by otherwise eligible clean coverage."""
    values = [event(0, 14), event(0, 22, "## unknown definition")]
    result = audit.reconstruct(values, [{"kind": "truncated_AUX"}])
    assert {item["kind"] for item in result["issues"]} >= {
        "binary_structure",
        "unsupported_definition_syntax",
    }
    assert np.all(result["reasons"] == 128)
    assert result["effective_decode_safe"] is True


def test_exact_standard_table_identity_matches_source_bound_library() -> None:
    """The independently stated symbol/description contract cannot drift with a dependency."""
    from wfdb.io.annotation import ann_label_table

    for code, expected in audit.independent.STANDARD_INTERPRETED_LABELS.items():
        actual = ann_label_table.loc[code]
        assert (actual["symbol"], actual["description"]) == expected


def test_released_roster_projection_correction_is_explicit_not_an_old_file_rewrite() -> None:
    """A missing integer inside a published range cannot fabricate a released record."""
    import copy
    import json
    from pathlib import Path

    base = json.loads(Path("configs/ssl_synthetic_edb_annotation_qualification_v1.json").read_text())
    records = [f"e{number:04d}" for number in range(100, 190) if number != 120] + ["e9999"]
    with pytest.raises(ValueError, match="linkage member absent"):
        audit.independent.group_roles(records, base)
    operating = copy.deepcopy(base)
    operating["grouping"]["explicit_groups"][0].remove("e0120")
    operating["grouping"]["expected_document_components"] = 79
    rows = audit.independent.group_roles(records, operating)
    assert len(rows) == 90
    assert len({row["component_id"] for row in rows}) == 79
    assert "e0120" not in {row["record_id"] for row in rows}
    assert base["grouping"]["expected_document_components"] == 78


def test_persisted_selected_rows_keep_raw_channel_separate_from_aux_channel() -> None:
    """Raw metadata channel must never replace the target's observed AUX channel."""
    values = [event(0, 14), event(1, 18, "(ST0+"), event(2, 18, "AST0+123"), event(3, 18, "ST0+)")]
    values[1]["channel"] = 255
    parsed = audit.reconstruct(values, [])
    rows = audit.selected_records(values, parsed)
    assert len(rows) == 4
    assert "interpretation" not in rows[0]
    assert rows[1]["channel"] == 255
    assert rows[1]["interpretation"]["channel"] == 0
    assert rows[2]["interpretation"]["magnitude_microvolts"] == 123
    assert parsed["intervals"][0]["peak_microvolts"] == [123]
    assert audit.complete_quality_spans(parsed) == [{"start": 0, "end": 1800000, "states": ["c", "c"]}]


def test_persisted_definition_records_retain_exact_contract_and_no_unknown_text() -> None:
    """Declaration hashes retain non-target contents without serializing arbitrary prose."""
    values = [
        event(0, 22, "## annotation type definitions"),
        event(0, 22, "18 s ST change"),
        event(0, 22, "## end of definitions"),
    ]
    parsed = audit.reconstruct(values, [])
    rows = audit.definition_records(values, parsed)
    assert rows[0]["kind"] == "label_definition_boundary"
    assert rows[1]["kind"] == "custom_label"
    assert rows[1]["label_store"] == 18
    assert rows[1]["symbol"] == "s"
    assert rows[1]["valid"] is True
    assert "description" not in rows[1]
    assert rows[2]["boundary"] == "end"
    assert parsed["selected"] == []


def clean_fixture() -> tuple:
    """State a complete clean artificial record independently of main reconstruction code."""
    import hashlib

    payload = bytes((0, 56, 0, 0))
    decoded = audit.independent.decode(payload)
    header = {
        "signals": 2,
        "fs": 250,
        "samples": 1800000,
        "source_signal_names": ["I", "II"],
        "sanitized_header_lines": 3,
    }
    role = {"record": "artificial", "component": "artificial", "bucket": 1, "role": "train_candidate"}
    digest = audit.independent.event_digest(decoded["events"])
    count = {
        "events": 1,
        "selected_events": 1,
        "issues": 0,
        "episodes": 0,
        "base_episodes": 0,
        "extreme_episodes": 0,
        "axis_episodes": 0,
        "ST_positive_windows": 0,
        "reference_windows": 1440,
        "unknown_windows": 0,
    }
    saved = {
        **role,
        "metadata": {
            **header,
            "record": "artificial",
            "annotation_bytes": 4,
            "annotation_sha256": hashlib.sha256(payload).hexdigest(),
            "event_count": 1,
            "canonical_event_sha256": digest,
            "wfdb_filtered_event_count": 1,
            "wfdb_filtered_event_sha256": digest,
            "wfdb_rdann_event_count": 1,
            "wfdb_rdann_event_sha256": digest,
            "wfdb_effective_view_verified": True,
            "wfdb_removed_indices": [],
            "wfdb_potential_definition_indices": [],
            "complete_raw_target": True,
            "independent_decoder_verified": False,
            "decoder_issues": [],
        },
        "selected_events": [
            {
                "event_index": 0,
                "sample": 0,
                "type_code": 14,
                "subtype": 0,
                "channel": 0,
                "num": 0,
                "aux_length": 0,
                "aux_sha256": hashlib.sha256(b"").hexdigest(),
            }
        ],
        "episodes": [],
        "quality_spans": [{"start": 0, "end": 1800000, "states": ["c", "c"]}],
        "definitions": [],
        "diagnostics": [],
        "overlap_pairs": [],
        "issues": [],
        "counts": count,
        "engineering_passed": True,
        "descriptives": {
            "unknown_reason_bit_cells": {str(1 << power): 0 for power in range(8)},
            "episode_class_counts": {},
            "raw_quality_seconds_by_channel": [
                {"unknown": 0, "clean": 7200.0, "noisy": 0, "unreadable": 0},
                {"unknown": 0, "clean": 7200.0, "noisy": 0, "unreadable": 0},
            ],
            "quality_seconds_are_not_qualified_label_coverage": True,
        },
    }
    return (
        saved,
        decoded,
        header,
        role,
        np.full((2, 720), 2, dtype=np.uint8),
        np.zeros((2, 720), dtype=np.uint16),
    )


def test_full_record_comparison_and_single_cell_mutations() -> None:
    """A clean known reference grid passes; one label or one reason corruption is counted exactly."""
    saved, decoded, header, role, labels, reasons = clean_fixture()
    result = audit.compare_record(saved, decoded, header, role, labels, reasons)
    assert result["mismatch_count"] == 0
    assert result["label_cells_compared"] == 1440
    labels[1, 719] = 1
    reasons[0, 0] = 128
    result = audit.compare_record(saved, decoded, header, role, labels, reasons)
    assert result["mismatch_count"] == 2
    assert result["label_cell_mismatches"] == result["reason_cell_mismatches"] == 1
    saved["metadata"]["canonical_event_sha256"] = "0" * 64
    assert audit.compare_record(saved, decoded, header, role, labels, reasons)["mismatch_count"] == 3


def test_issue_comparison_preserves_multiplicity_and_source_identity() -> None:
    """Order differences are harmless, but duplicate loss or changed source identity is a mismatch."""
    import hashlib

    values = [event(10, 18, "unsupported")]
    parsed = audit.reconstruct(values, [])
    expected = [
        {
            "kind": "unsupported_episode_candidate",
            "event_index": 0,
            "sample": 10,
            "type_code": 18,
            "aux_sha256": hashlib.sha256(b"unsupported").hexdigest(),
        }
    ]
    own = audit.issue_multiset(parsed["issues"], values, parsed, independent_view=True)
    assert own == audit.issue_multiset(expected, values, parsed)
    assert own != audit.issue_multiset(expected * 2, values, parsed)
    expected[0]["sample"] = 11
    assert own != audit.issue_multiset(expected, values, parsed)


def test_array_preflight_rejects_payload_length_dtype_and_shape(tmp_path) -> None:
    """Header-only setup rejects truncation and wrong scientific array contracts."""
    path = tmp_path / "array.npy"
    np.save(path, np.zeros((2, 720), dtype=np.uint8))
    audit.array_header(path, (2, 720), "uint8")
    with pytest.raises(ValueError, match="contract differs"):
        audit.array_header(path, (2, 720), "uint16")
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="contract differs"):
        audit.array_header(path, (2, 720), "uint8")


def test_failed_partial_attempt_keeps_actual_cost_and_prior_once(tmp_path) -> None:
    """A failed audit preserves a terminal receipt and appends only its own CPU cost."""
    import json
    import time

    setup = {
        "config": {
            "resources": {
                "independent_verification": {
                    "cpu_seconds": 120,
                    "wall_seconds": 300,
                    "rss_bytes": 2**40,
                    "output_bytes": 2**24,
                }
            }
        },
        "prior": {
            "current_machine_stages": 44,
            "current_machine_cumulative_cpu_seconds": 1284.0,
            "current_machine_cumulative_gpu_wall_seconds": 564.0,
        },
        "summary": {"availability": {}, "engineering_passed_before_independent_audit": False},
        "obligations": {},
    }
    admission = {"source_sha256": {}, "main_summary_sha256": "a" * 64, "prior_ledger": {"sha256": "b" * 64}}
    audit.write_new(tmp_path / "admission_snapshot.json", admission)
    summary = audit.terminal(
        tmp_path, setup, admission, [], "artificial failure", time.process_time(), time.monotonic()
    )
    ledger = json.loads((tmp_path / "prospective_ledger.json").read_text())
    assert not summary["independent_audit_passed"]
    assert not summary["combined_annotation_qualification"]
    assert ledger["current_machine_stages"] == 45
    assert ledger["current_machine_cumulative_cpu_seconds"] == 1284.0 + ledger["cpu_seconds"]
    assert ledger["current_machine_cumulative_gpu_wall_seconds"] == 564.0
    assert not ledger["technical_passed"]
    assert audit.sha256_file(tmp_path / "summary.json") == ledger["sha256"]


@pytest.mark.parametrize(
    ("code", "text", "kind"),
    [
        (22, "(ST0+", "unsupported_episode_candidate"),
        (18, "(T0+", "episode_family_type_mismatch"),
        (18, "(st0+", "unsupported_episode_candidate"),
        (18, "(ST0+-", "unsupported_episode_candidate"),
        (18, "(ST0++", "double_sign_ST_not_allowed"),
    ],
)
def test_issue_contract_translation_preserves_distinct_grammar_failures(code, text, kind) -> None:
    """Independent naming maps to declared code/case/family failures without forgiving syntax."""
    values = [event(10, code, text)]
    parsed = audit.reconstruct(values, [])
    assert audit.issue_identity(parsed["issues"][0], values, parsed) == (kind, 0)


def test_zero_mismatch_independent_pass_does_not_override_main_engineering_failure(tmp_path) -> None:
    """A complete faithful audit confirms a scientific engineering failure without upgrading it."""
    import time

    rows = []
    for index in range(90):
        rows.append(
            {
                "record": f"artificial{index}",
                "component": f"artificial{index}",
                "role": "train_candidate",
                "counts": {"ST_positive_windows": 0, "reference_windows": 0},
                "mismatch_count": 0,
                "label_cells_compared": 1440,
                "reason_cells_compared": 1440,
                "label_cell_mismatches": 0,
                "reason_cell_mismatches": 0,
            }
        )
    setup = {
        "config": {
            "resources": {
                "independent_verification": {
                    "cpu_seconds": 120,
                    "wall_seconds": 300,
                    "rss_bytes": 2**40,
                    "output_bytes": 2**24,
                }
            }
        },
        "prior": {
            "current_machine_stages": 44,
            "current_machine_cumulative_cpu_seconds": 1284.0,
            "current_machine_cumulative_gpu_wall_seconds": 564.0,
        },
        "summary": {
            "availability": audit.availability(rows),
            "engineering_passed_before_independent_audit": False,
        },
        "obligations": {},
    }
    admission = {"source_sha256": {}, "main_summary_sha256": "a" * 64, "prior_ledger": {"sha256": "b" * 64}}
    audit.write_new(tmp_path / "admission_snapshot.json", admission)
    result = audit.terminal(tmp_path, setup, admission, rows, None, time.process_time(), time.monotonic())
    assert result["independent_audit_passed"]
    assert result["mismatch_count"] == 0
    assert result["label_cells_compared"] == result["reason_cells_compared"] == 129600
    assert not result["main_engineering_passed"]
    assert not result["combined_annotation_qualification"]
