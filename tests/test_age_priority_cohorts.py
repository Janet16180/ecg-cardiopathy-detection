"""Scientific selection boundaries for the proposed age-priority cohorts."""

import hashlib

import pandas as pd
import pytest

from ecg_experiment.age_priority_cohorts import select_cohort


def candidates() -> pd.DataFrame:
    """Return a small pool with independent age and label-evidence priorities."""
    rows = pd.DataFrame({
        "record_id": ["older_verified", "young_verified", "young_less_verified", "minor", "unknown"],
        "age": [70, 24, 19, 12, float("nan")],
        "label_evidence_rank": [0, 0, 1, 0, 0],
        "target": [1, 0, 1, 0, 1],
    })
    rows["signal_sha256"] = rows.record_id.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    return rows.assign(source="synthetic", split="train", quality_passed=True,
                       label_evidence="synthetic evidence, not clinical data")


def test_label_evidence_precedes_youth_and_unknown_age_is_not_young() -> None:
    """A younger record must not displace a better-supported label."""
    selected = select_cohort(candidates(), 10, 42, True)
    assert selected.record_id.tolist() == ["young_verified", "older_verified", "young_less_verified"]
    assert selected.target.tolist() == [0, 1, 1]


def test_comparison_keeps_full_age_pool_and_order_is_reproducible() -> None:
    """The all-age selection retains minors and unknown ages without favoring them."""
    pool = candidates()
    one = select_cohort(pool, 10, 42, False)
    two = select_cohort(pool.sample(frac=1, random_state=8), 10, 42, False)
    pd.testing.assert_frame_equal(one, two)
    assert set(one.record_id) == set(pool.record_id)
    assert one.iloc[-1].record_id == "young_less_verified"


@pytest.mark.parametrize(("column", "value"), [
    ("split", "test"), ("quality_passed", False), ("target", float("nan")),
    ("label_evidence_rank", 3), ("label_evidence", ""), ("age", 300),
])
def test_rejects_unadmitted_candidates(column: str, value: object) -> None:
    """Selection cannot silently admit held-out, unclean or ambiguously labeled ECGs."""
    pool = candidates()
    pool.loc[0, column] = value
    with pytest.raises(ValueError, match="Reserved|quality|binary|evidence|provenance|Age"):
        select_cohort(pool, 25_000, 42, True)


def test_rejects_duplicate_waveform_across_sources() -> None:
    """A source name change cannot disguise the same canonical signal."""
    pool = candidates()
    pool.loc[1, "signal_sha256"] = pool.loc[0, "signal_sha256"]
    pool.loc[1, "source"] = "other_source"
    with pytest.raises(ValueError, match="deduplicated"):
        select_cohort(pool, 25_000, 42, True)
