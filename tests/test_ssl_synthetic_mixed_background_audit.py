"""Leakage, pairing and strict-boundary checks for mixed-background verification."""

import copy
import hashlib

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_mixed_background_audit as audit


def payload(records: list[tuple[int, int, int]]) -> bytes:
    """Serialize artificial identity-only rows with an uninterpreted diagnostic column."""
    lines = ["ecg_id,patient_id,strat_fold,filename_hr,diagnostic"]
    lines.extend(
        f"{ecg},{patient}.0,{fold},records500/00000/{ecg:05d}_hr,uninterpreted"
        for ecg, patient, fold in records
    )
    return ("\n".join(lines) + "\n").encode()


def train_patients() -> list[int]:
    """Return artificial patients with a known original TRAIN role."""
    return [
        p
        for p in range(1, 200)
        if int(hashlib.sha256(f"ssl_synthetic_partition_v1:20261018:{p}".encode()).hexdigest(), 16)
        % 10000
        < 8000
    ]


def fixture_rows() -> list[dict]:
    """Create heterogeneous complete fixed-fit rows with analytically paired offsets."""
    rows = []
    for training in audit.TRAINING_ARMS:
        for source in audit.transfer.SOURCES:
            for slot in range(64):
                for fit in audit.transfer.FITS:
                    offset = 0.02 if training == "mixed" else 0.0
                    positives = {
                        name: dict.fromkeys(
                            audit.METRICS, 0.1 + 0.0001 * slot + 0.001 * fit + 0.002 * position + offset
                        )
                        for position, name in enumerate(audit.transfer.POSITIVES)
                    }
                    controls = {
                        name: {
                            metric: float(slot % 3 == 0)
                            if metric == "maximum_gt_0p5"
                            else 0.01 + slot * 0.00001 + fit * 0.0001
                            for metric in audit.CONTROLS
                        }
                        for name in ("sham", "nuisance")
                    }
                    rows.append(
                        {
                            "training_arm": training,
                            "source": source,
                            "slot": slot,
                            "fit_id": fit,
                            "arm": "B",
                            "positives": positives,
                            "controls": controls,
                        }
                    )
    return rows


def test_repeat_in_calibration_fold_quarantines_entire_patient() -> None:
    """A single fold8 record excludes the same patient's otherwise eligible fold1 row."""
    first, second = train_patients()[:2]
    rows, counts = audit.cohort_candidates(payload([(1, first, 1), (2, first, 8), (3, second, 2)]))
    assert [r["patient_id"] for r in rows] == [str(second)]
    assert counts["fold8_9_10_quarantined_patients"] == 1


def test_cohort_lowest_record_and_digest_order_survive_metadata_reordering() -> None:
    """Repeat records cannot multiply patient weight or change the prospective ordering."""
    patients = train_patients()[:8]
    records = [(2 * i + 1, patient, 1) for i, patient in enumerate(patients)]
    records += [(2 * i + 2, patient, 2) for i, patient in enumerate(patients)]
    first, _ = audit.cohort_candidates(payload(records))
    second, _ = audit.cohort_candidates(payload(records[::-1]))
    assert first == second
    assert all(int(r["ecg_id"]) % 2 == 1 for r in first)
    assert len({r["patient_id"] for r in first}) == 8


def test_original_nontrain_patient_and_bad_identity_cannot_enter() -> None:
    """Unspent protected roles stay excluded even when the patient's official fold is permitted."""
    patient = next(p for p in range(1, 100) if p not in train_patients())
    rows, _ = audit.cohort_candidates(payload([(1, patient, 1)]))
    assert rows == []
    with pytest.raises(ValueError, match="Invalid metadata"):
        audit.cohort_candidates(payload([(1, 1, 1), (1, 2, 1)]))
    with pytest.raises(ValueError, match="finite integer"):
        audit.identifier("1.5")


def test_complete_shared_draws_keep_paired_constant_treatment_contrast() -> None:
    """All fitted states stay fixed and every conditional treatment draw retains a known offset."""
    rows = fixture_rows()
    report, draws = audit.summarize(rows, {"bootstrap": {"draws": 2000, "seed": 2026106701}})
    assert draws.shape == (2000, 348)
    assert len(report["per_fit_mode"]) == 48
    assert len(report["scientific_gates"]) == 14
    name = "mixed_minus_generated__real__biphasic__pooled__edited_ap"
    np.testing.assert_allclose(draws[:, report["endpoint_order"].index(name)], 0.02, atol=1e-15, rtol=0)
    assert report["scientific_passed"]
    vectors, _ = audit.endpoint_vectors(rows)
    primary = vectors["generated__real__biphasic__pooled__edited_ap"]
    np.testing.assert_allclose(primary, 0.107 + np.arange(64) * 0.0001, atol=1e-15, rtol=0)
    for malformed in (rows[:-1], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError, match="all768"):
            audit.endpoint_vectors(malformed)


def test_fourteen_gate_strict_and_inclusive_boundaries_are_distinct() -> None:
    """Equality passes the minimum gain but fails strict uncertainty/preservation limits."""
    report, _ = audit.summarize(fixture_rows(), {"bootstrap": {"draws": 2000, "seed": 2026106701}})
    endpoints = copy.deepcopy(report["endpoints"])
    primary = endpoints["mixed_minus_generated__real__biphasic__pooled__edited_ap"]
    primary.update(mean=0.01, ci_low=0.0)
    endpoints["mixed_minus_generated__real__biphasic__focal__edited_ap"]["ci_low"] = -0.01
    endpoints["mixed_minus_generated__real__sham__top600_mean"]["ci_high"] = 0.02
    gates = audit.decision(endpoints)
    assert gates["primary_AP_gain_at_least_0p01"]
    assert not gates["primary_AP_gain_lower_positive"]
    assert not gates["real_biphasic_focal_noninferiority"]
    assert not gates["real_sham_top600_preservation"]
