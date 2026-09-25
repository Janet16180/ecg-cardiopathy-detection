"""Prospective cost arithmetic for the frozen xECG seed-47 v13 design."""

from __future__ import annotations

from dataclasses import dataclass

H_SECONDS = 2693.7965717150364
P_SECONDS = 1098.4113020410005
Q_INITIAL_SECONDS = 1025.0
R_INITIAL_SECONDS = 300.0
CEILING_SECONDS = 7200.0
MARGIN = 1.25
CHECKPOINTS = (40, 80, 120, 160, 200, 240)


@dataclass(frozen=True)
class GateState:
    """Disjoint elapsed time and the remaining work at one safe gate."""

    elapsed_seconds: float
    unstarted_arms: int
    completed_pipeline_seconds: tuple[float, ...]
    q_remaining_seconds: float
    r_remaining_seconds: float
    active_elapsed_seconds: float = 0.0
    active_fraction: float = 0.0
    completed_block_paces: tuple[float, ...] = ()

    def projection(self) -> dict[str, float | bool]:
        """Apply v13's frozen remaining-work rule without double counting."""
        if self.unstarted_arms not in (0, 1, 2):
            raise ValueError("Invalid unstarted-arm count")
        if any(
            value < 0
            for value in (
                self.elapsed_seconds,
                self.q_remaining_seconds,
                self.r_remaining_seconds,
                self.active_elapsed_seconds,
            )
        ):
            raise ValueError("Negative cost interval or reserve")
        if not 0 <= self.active_fraction <= 1:
            raise ValueError("Invalid active scientific fraction")
        if self.active_fraction == 1:
            raise ValueError("Completed arm must transition atomically")
        if self.active_elapsed_seconds and not self.active_fraction:
            # Initialization can consume time at f=0. It is already in E.
            pass
        if self.active_elapsed_seconds > self.elapsed_seconds:
            raise ValueError("Active elapsed time is missing from the ledger")
        if any(value < 0 for value in (*self.completed_pipeline_seconds, *self.completed_block_paces)):
            raise ValueError("Invalid measured pace")
        active_count = 2 - len(self.completed_pipeline_seconds) - self.unstarted_arms
        inactive_with_work = not active_count and (self.active_elapsed_seconds or self.active_fraction)
        if active_count not in (0, 1) or inactive_with_work:
            raise ValueError("Incomplete or duplicated M/F arm accounting")
        pstar = max((P_SECONDS, *self.completed_pipeline_seconds))
        pace = max((pstar, *self.completed_block_paces))
        if self.active_fraction:
            pace = max(pace, self.active_elapsed_seconds / self.active_fraction)
        active_remaining = (1 - self.active_fraction) * pace if active_count else 0.0
        forecast = (
            self.unstarted_arms * pstar
            + active_remaining
            + self.q_remaining_seconds
            + self.r_remaining_seconds
        )
        projected = H_SECONDS + self.elapsed_seconds + MARGIN * forecast
        return {
            "pstar_seconds": pstar,
            "active_pace_seconds": pace,
            "active_remaining_seconds": active_remaining,
            "forecast_seconds": forecast,
            "projected_seconds": projected,
            "go": projected <= CEILING_SECONDS,
        }


def checkpoint_fraction(updates: int) -> float:
    """Credit only complete scheduled scientific updates through checkpoint 240."""
    if updates not in CHECKPOINTS:
        raise ValueError("Only mandatory complete checkpoints earn this credit")
    return updates / 480


def first_checkpoint_admissible_seconds(q_spent_seconds: float = Q_INITIAL_SECONDS) -> float:
    """Solve the frozen first-block gate with the first block as pace floor."""
    if q_spent_seconds < 0:
        raise ValueError("Negative preparation time")
    # f=1/12; once a > P/12, Tpace=12*a and Aremaining=11*a.
    threshold = (CEILING_SECONDS - H_SECONDS - q_spent_seconds - MARGIN * (P_SECONDS + R_INITIAL_SECONDS)) / (
        1 + MARGIN * 11
    )
    if threshold < P_SECONDS / 12:
        raise ValueError("No admissible first block even at historical pace")
    return threshold
