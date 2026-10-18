"""Artificial tests for patient coordinates, predecessor integrity and cumulative budgets."""

import copy

import pytest

from ecg_experiment import ssl_synthetic_paired_background_common as common


def case_plan() -> list[dict]:
    """Build identity-only patient sets; no real metadata or waveform is read."""
    rows = []
    for role, count, first, profile in (
        ("train", 500, 0, "A"),
        ("calibration", 64, 500, "A"),
        ("evaluation", 128, 564, "B"),
    ):
        for local in range(count):
            subject = first + local
            for variant, name in enumerate(("focal", "repeated", "sham", "nuisance")):
                rows.append(
                    {
                        "case_index": 4 * subject + variant,
                        "role": role,
                        "cache_index": subject,
                        "subject_index": subject,
                        "role_subject_index": local,
                        "profile": profile,
                        "patient_id": str(subject),
                        "variant": name,
                        "target": variant < 2,
                        "kind": "positive" if variant < 2 else name,
                        "mode": name if variant < 2 else "none",
                        "nuisance": variant == 3,
                        "baseline_sha256": f"control:{subject}",
                        "edited_sha256": f"control:{subject}",
                        "nuisance_parameters": {},
                    }
                )
    return rows


def test_role_map_keeps_global_masks_and_local_probability_rows_distinct() -> None:
    """Calibration/evaluation map rows begin at global case 2000, not zero."""
    roles = common.validate_case_plan(case_plan())
    assert roles["train"] == list(range(2000))
    assert roles["calibration"] == list(range(2000, 2256))
    assert roles["evaluation"] == list(range(2256, 2768))
    assert roles["evaluation"][0] - 2000 == 256


@pytest.mark.parametrize(
    ("index", "field", "value"),
    [
        (2000, "cache_index", 0),
        (2256, "subject_index", 0),
        (2256, "role_subject_index", 564),
        (0, "profile", "B"),
        (2000, "patient_id", "0"),
        (2256, "baseline_sha256", "changed"),
        (2257, "target", False),
        (2257, "variant", "focal"),
    ],
)
def test_corrupt_patient_or_counterpart_coordinates_stop(index: int, field: str, value: object) -> None:
    """Patient leakage, misplaced masks and altered counterparts cannot be silently accepted."""
    cases = case_plan()
    cases[index][field] = value
    with pytest.raises(ValueError, match="Patient|patient|coordinate|Target|Positive"):
        common.validate_case_plan(cases)


def test_predecessor_reproduction_requires_full_values_and_failed_original() -> None:
    """A successful audit flag alone cannot admit changed scores or a changed original verdict."""
    audit = {
        "combined_complete_pass": True,
        "all_passed": True,
        "mismatch_count": 0,
        "independently_recomputed_complete_summary": {"methods": {"S": 0.289397}},
    }
    transfer = {
        "evaluation_completed": True,
        "original_advancement_passed": False,
        "methods": {"S": 0.289397},
    }
    original = {"decision": {"engineering_advancement_passed": False}}
    common.predecessor_integrity(audit, transfer, original)
    altered = copy.deepcopy(transfer)
    altered["methods"]["S"] += 0.0001
    with pytest.raises(ValueError, match="numerical"):
        common.predecessor_integrity(audit, altered, original)
    original["decision"]["engineering_advancement_passed"] = True
    with pytest.raises(ValueError, match="advancement"):
        common.predecessor_integrity(audit, transfer, original)


def test_nested_comparison_rejects_schema_omission_and_boolean_coercion() -> None:
    """Extra fields and bool/int substitution remain visible at fixed numerical tolerance."""
    assert common.same_values({"a": [1.0, False]}, {"a": [1.0 + 1e-13, False]})
    assert not common.same_values({"a": 1}, {"a": 1, "b": 2})
    assert not common.same_values(True, 1)


@pytest.mark.parametrize(("prior_cpu", "prior_wall", "reserve"), [(14395, 0, 1), (0, 7195, 1), (0, 0, 1795)])
def test_live_guard_charges_cumulative_costs_and_reserve(
    monkeypatch: pytest.MonkeyPatch, prior_cpu: float, prior_wall: float, reserve: float
) -> None:
    """A stage cannot borrow spent global time or the reserved evaluation allowance."""
    monkeypatch.setattr(common.time, "perf_counter", lambda: 10.0)
    monkeypatch.setattr(common.time, "process_time", lambda: 10.0)
    context = {
        "prior_cpu_seconds": prior_cpu,
        "prior_gpu_wall_seconds": prior_wall,
        "config": {
            "gpu_seconds_stage_ceiling": 900,
            "cpu_seconds_stage_ceiling": 1800,
            "gpu_seconds_ceiling": 7200,
            "cpu_seconds_ceiling": 14400,
        },
    }
    with pytest.raises(TimeoutError):
        common.check_budget(context, {"wall": 0.0, "cpu": 0.0}, reserve)
