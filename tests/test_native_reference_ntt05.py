"""Native reference contracts: geometry, patient weighting, abstention and redundancy."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_experiment.native_reference_ntt05 import (
    NativeTokens,
    ReferenceInstance,
    SignalRegion,
    candidate_regions,
    lead_patch_tokens,
    match_region,
    match_vector,
    pool_region,
    reference_bank,
    soft_decay,
    unit_vector,
)


def grid() -> NativeTokens:
    """Return contextual synthetic lead-major patch features, without a diagnostic label."""
    return lead_patch_tokens(
        np.array([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]]),
        ("V1", "V6"),
        250,
        50,
        100,
        "synthetic",
        "all leads and times",
        "target-record",
    )


def bank():
    """Return independent synthetic reference patients."""
    return reference_bank(
        [
            ReferenceInstance("a", "first", np.array([1, 0]), "synthetic", "synthetic"),
            ReferenceInstance("b", "second", np.array([0, 1]), "synthetic", "synthetic"),
        ],
        ("first", "second"),
    )


def test_lead_patch_geometry_is_explicit_and_physical() -> None:
    """Flatten lead-major vectors with physical 200 ms anchors, not inferred geometry."""
    tokens = grid()
    np.testing.assert_allclose(tokens.intervals, [[0, 0.2], [0.2, 0.4], [0, 0.2], [0.2, 0.4]])
    assert tokens.token_leads == (("V1",), ("V1",), ("V6",), ("V6",))
    assert tokens.duration_seconds == 0.4
    assert tokens.context_scope == "all leads and times"
    np.testing.assert_array_equal(tokens.values, [[1, 0], [0, 1], [-1, 0], [0, -1]])


def test_partial_token_occupancy_and_lead_selection() -> None:
    """A half-covered first cell and complete second cell receive 1:2 weight."""
    vector = pool_region(grid(), SignalRegion(0.1, 0.4, ("V1",)))
    np.testing.assert_allclose(vector, unit_vector(np.array([1, 2])))
    np.testing.assert_allclose(pool_region(grid(), SignalRegion(0, 0.2, ("V6",))), [-1, 0])


def test_mixed_lead_features_cannot_be_sliced() -> None:
    """All-lead temporal tokens cannot fabricate single-lead vectors."""
    tokens = NativeTokens(
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0.2], [0.2, 0.4]]),
        (("V1", "V6"),) * 2,
        0.4,
        "synthetic mixed leads",
        "full recording",
        "target-record",
    )
    with pytest.raises(ValueError, match="mixed-lead"):
        pool_region(tokens, SignalRegion(0, 0.2, ("V1",)))
    np.testing.assert_allclose(pool_region(tokens, SignalRegion(0, 0.2, ("V1", "V6"))), [1, 0])


@pytest.mark.parametrize("region", [SignalRegion(0, 0.5, ("V1",)), SignalRegion(0, 0.2, ("II",))])
def test_missing_anchors_and_outside_regions_raise(region: SignalRegion) -> None:
    """Do not replace unsupported regions by global pooling."""
    with pytest.raises(ValueError, match="outside|coverage"):
        pool_region(grid(), region)


def test_gaps_in_anchor_coverage_raise() -> None:
    """Partial temporal coverage cannot silently represent a full requested interval."""
    tokens = NativeTokens(
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 0.1], [0.2, 0.4]]),
        (("V1",),) * 2,
        0.4,
        "synthetic gap",
        "local",
        "target-record",
    )
    with pytest.raises(ValueError, match="gap"):
        pool_region(tokens, SignalRegion(0, 0.4, ("V1",)))


@pytest.mark.parametrize("leads", [("V13",), ("V1", "V1"), ()])
def test_invalid_lead_scopes_raise(leads: tuple[str, ...]) -> None:
    """Unknown, duplicate and absent lead assignments are not accepted."""
    with pytest.raises(ValueError, match="canonical"):
        SignalRegion(0, 0.2, leads)


@pytest.mark.parametrize("values", [np.array([0, 0]), np.array([np.nan, 1]), np.array([[1, 0]])])
def test_invalid_vectors_raise(values: np.ndarray) -> None:
    """Empty-direction and nonfinite features cannot become a prediction."""
    with pytest.raises(ValueError, match="nonzero"):
        unit_vector(values)


def test_patch_padding_and_implicit_context_rejected() -> None:
    """A nominal stride cannot hide dropped samples or undeclared context."""
    with pytest.raises(ValueError, match="padding"):
        lead_patch_tokens(np.ones((1, 2, 3)), ("V1",), 250, 50, 101, "synthetic", "full", "target-record")
    with pytest.raises(ValueError, match="context"):
        lead_patch_tokens(np.ones((1, 2, 3)), ("V1",), 250, 50, 100, "synthetic", "", "target-record")


def test_patient_weight_does_not_follow_instance_count() -> None:
    """Ten repeated regions of one patient retain one patient weight."""
    repeated = [ReferenceInstance("a", "first", np.array([1, 0]), "synthetic", "synthetic")] * 10
    others = [
        ReferenceInstance("c", "first", np.array([0, 1]), "synthetic", "synthetic"),
        ReferenceInstance("b", "second", np.array([-1, 0]), "synthetic", "synthetic"),
    ]
    result = reference_bank(repeated + others, ("first", "second"))
    np.testing.assert_allclose(result.prototypes[0], unit_vector(np.array([1, 1])))
    assert result.patients == frozenset(("a", "b", "c"))


def test_target_patient_is_rejected_across_all_reference_findings() -> None:
    """A patient in another reference finding is still a support patient."""
    with pytest.raises(ValueError, match="reference bank"):
        match_vector(
            np.array([1, 0]),
            SignalRegion(0, 0.2, ("V1",)),
            bank(),
            "b",
            -1,
            0,
            "full",
            "synthetic",
            "target-record",
        )


def test_missing_reference_finding_raises() -> None:
    """A missing reference class is not inferred from target labels."""
    with pytest.raises(ValueError, match="no reference"):
        reference_bank(
            [ReferenceInstance("a", "first", np.array([1, 0]), "synthetic", "synthetic")], ("first", "second")
        )


def test_abstention_thresholds_and_fixed_tie_order() -> None:
    """Score and margin rejection keep unknown separate from an explicit finding."""
    region = SignalRegion(0, 0.2, ("V1",))
    ref = bank()
    matched = match_vector(
        np.array([1, 1]), region, ref, "target", -1, 0, "full", "synthetic", "target-record"
    )
    assert matched.finding == "first"
    assert (
        match_vector(
            np.array([1, 1]), region, ref, "target", -1, 0.1, "full", "synthetic", "target-record"
        ).finding
        is None
    )
    assert (
        match_vector(
            np.array([-1, -1]), region, ref, "target", 0, 0, "full", "synthetic", "target-record"
        ).finding
        is None
    )
    assert (
        match_vector(
            np.array([1, 0]), region, ref, "target", 1, 1, "full", "synthetic", "target-record"
        ).finding
        == "first"
    )


def test_region_matching_retains_context_and_scores() -> None:
    """A local anchor selection retains its full-context declaration."""
    result = match_region(grid(), SignalRegion(0, 0.2, ("V1",)), bank(), "target", 0, 0)
    assert result.finding == "first"
    assert result.context_scope == "all leads and times"
    np.testing.assert_array_equal(result.scores, [1, 0])


def test_multiscale_windows_include_tail_without_duplicates() -> None:
    """Generate label-free intervals deterministically, including an off-grid final start."""
    regions = candidate_regions(1, (("V1",),), (0.3, 0.3, 2), 0.4)
    assert len(regions) == 3
    np.testing.assert_allclose([(r.start, r.stop) for r in regions], [[0, 0.3], [0.4, 0.7], [0.7, 1]])
    assert regions == candidate_regions(1, (("V1",),), (0.3, 0.3, 2), 0.4)


@pytest.mark.parametrize(
    ("duration", "widths", "stride"), [(1, (0.2,), 0), (1, (), 0.1), (1, (np.nan,), 0.1)]
)
def test_invalid_candidate_geometry_raises(duration: float, widths: tuple, stride: float) -> None:
    """Invalid sampling proposals cannot silently create candidates."""
    with pytest.raises(ValueError, match="positive|width"):
        candidate_regions(duration, (("V1",),), widths, stride)


def test_soft_decay_only_suppresses_redundant_same_finding_leads() -> None:
    """Identical nested evidence decays; different leads and recurring beats remain distinct."""
    ref = bank()
    regions = [
        SignalRegion(0, 0.4, ("V1",)),
        SignalRegion(0.1, 0.3, ("V1",)),
        SignalRegion(0.1, 0.3, ("V6",)),
        SignalRegion(0.6, 0.8, ("V1",)),
    ]
    matches = [
        match_vector(np.array([1, 0]), r, ref, "target", 0, 0, "full", "synthetic", "target-record")
        for r in regions
    ]
    np.testing.assert_allclose([score for _, score in soft_decay(matches)], [1, 0, 1, 1])
    assert len(soft_decay(matches)) == len(matches)


def test_abstained_and_negative_scores_are_not_boosted_or_suppressors() -> None:
    """Signed cosine scores cannot increase through multiplicative suppression."""
    ref = reference_bank(
        [
            ReferenceInstance("a", "first", np.array([1, 0]), "synthetic", "synthetic"),
            ReferenceInstance("b", "second", np.array([1, 1]), "synthetic", "synthetic"),
        ],
        ("first", "second"),
    )
    region = SignalRegion(0, 0.2, ("V1",))
    negative = match_vector(
        np.array([-1, -1]), region, ref, "target", -1, 0, "full", "synthetic", "target-record"
    )
    assert negative.similarity < 0
    assert all(score == negative.similarity for _, score in soft_decay([negative, negative]))
    unknown = match_vector(
        np.array([1, 0]), region, ref, "target", 1, 2, "full", "synthetic", "target-record"
    )
    assert unknown.finding is None
    assert soft_decay([unknown, unknown])[1][1] == unknown.similarity


def test_embedding_space_identity_must_match_even_for_equal_dimensions() -> None:
    """A different encoder/checkpoint/preprocessing cannot silently share a memory bank."""
    instances = [
        ReferenceInstance("a", "first", np.array([1, 0]), "reviewed source A", "encoder-A"),
        ReferenceInstance("b", "second", np.array([0, 1]), "reviewed source B", "encoder-B"),
    ]
    with pytest.raises(ValueError, match="embedding-space"):
        reference_bank(instances, ("first", "second"))
    with pytest.raises(ValueError, match="embedding-space"):
        match_vector(
            np.array([1, 0]),
            SignalRegion(0, 0.2, ("V1",)),
            bank(),
            "target",
            0,
            0,
            "full",
            "different checkpoint",
            "target-record",
        )
    assert bank().provenances == ("synthetic",)
    assert bank().embedding_space == "synthetic"


def test_soft_decay_rejects_different_record_coordinate_frames() -> None:
    """Matching timestamps in different ECGs cannot suppress each other."""
    region = SignalRegion(0, 0.2, ("V1",))
    first = match_vector(np.array([1, 0]), region, bank(), "target", 0, 0, "full", "synthetic", "record-a")
    second = match_vector(np.array([1, 0]), region, bank(), "target", 0, 0, "full", "synthetic", "record-b")
    with pytest.raises(ValueError, match="coordinate frames"):
        soft_decay([first, second])
