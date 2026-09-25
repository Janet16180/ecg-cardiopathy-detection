"""Synthetic boundary checks for the frozen v13 remaining-work gate."""

from __future__ import annotations

import pytest

from ecg_experiment.xecg_encoder_motion_v13_cost import (
    CEILING_SECONDS,
    H_SECONDS,
    P_SECONDS,
    Q_INITIAL_SECONDS,
    R_INITIAL_SECONDS,
    GateState,
    checkpoint_fraction,
    first_checkpoint_admissible_seconds,
)


def test_initial_and_first_checkpoint_boundaries() -> None:
    """The first block is credited once and its incomplete work is forecast."""
    initial = GateState(0, 2, (), Q_INITIAL_SECONDS, R_INITIAL_SECONDS).projection()
    assert initial["projected_seconds"] == pytest.approx(7096.0748268175375)
    assert initial["go"] is True
    limit = first_checkpoint_admissible_seconds()
    assert limit == pytest.approx(117.50435937177716)
    for elapsed, expected in ((limit - 0.001, True), (limit + 0.001, False)):
        row = GateState(
            elapsed + Q_INITIAL_SECONDS,
            1,
            (),
            0,
            R_INITIAL_SECONDS,
            active_elapsed_seconds=elapsed,
            active_fraction=checkpoint_fraction(40),
            completed_block_paces=(12 * elapsed,),
        ).projection()
        assert row["go"] is expected
    assert checkpoint_fraction(240) == 0.5
    with pytest.raises(ValueError, match="mandatory"):
        checkpoint_fraction(39)


def test_incomplete_arm_is_not_forecast_as_a_second_full_arm() -> None:
    """M elapsed time enters E once, while unstarted F retains full Pstar."""
    active = GateState(
        Q_INITIAL_SECONDS + P_SECONDS / 4,
        1,
        (),
        0,
        R_INITIAL_SECONDS,
        active_elapsed_seconds=P_SECONDS / 4,
        active_fraction=0.25,
    ).projection()
    assert active["active_remaining_seconds"] == pytest.approx(0.75 * P_SECONDS)
    assert active["forecast_seconds"] == pytest.approx(1.75 * P_SECONDS + R_INITIAL_SECONDS)
    assert active["projected_seconds"] < CEILING_SECONDS


def test_completed_m_atomically_updates_pstar() -> None:
    """A measured slow M enters E and raises the full F forecast."""
    completed = GateState(Q_INITIAL_SECONDS + 1200, 1, (1200,), 0, R_INITIAL_SECONDS).projection()
    assert completed["pstar_seconds"] == 1200
    assert completed["active_remaining_seconds"] == 0
    assert completed["projected_seconds"] == pytest.approx(
        H_SECONDS + Q_INITIAL_SECONDS + 1200 + 1.25 * (1200 + R_INITIAL_SECONDS)
    )


def test_ledger_rejects_omission_and_double_count() -> None:
    """Every M/F slot and active elapsed interval must be accounted for."""
    with pytest.raises(ValueError, match="missing"):
        GateState(5, 1, (), 0, 300, active_elapsed_seconds=6).projection()
    with pytest.raises(ValueError, match="accounting"):
        GateState(100, 2, (50,), 0, 300).projection()
    with pytest.raises(ValueError, match="atomically"):
        GateState(100, 1, (), 0, 300, active_elapsed_seconds=100, active_fraction=1).projection()
