"""Prospective incremental complete-path gate for Experiment 016 v14."""

from __future__ import annotations

import math

CEILING = 7200.0
P_HISTORICAL = 1098.4113020410005
MARGIN = 1.25
INITIAL_Q = 2350.0
REPORT_RESERVE = 300.0
STOP_RESERVE = 60.0
HISTORICAL_LOWER_BOUND = 6406.508688546029


def projection(
    elapsed: float,
    unstarted: int,
    completed: tuple[float, ...],
    *,
    active_update: int | None = None,
    slowest_block_seconds_per_update: float | None = None,
    active_tail_complete: bool = False,
    observed_tail_seconds: float = 0.0,
    q_remaining: float = 0.0,
    report_remaining: float = REPORT_RESERVE,
    stop_reserve: float = STOP_RESERVE,
) -> dict:
    """Forecast every unfinished production path with the frozen 25% margin."""
    numbers = (elapsed, *completed, q_remaining, report_remaining, stop_reserve, observed_tail_seconds)
    if (
        any(not math.isfinite(value) or value < 0 for value in numbers)
        or unstarted not in (0, 1, 2)
        or len(completed) + unstarted + (active_update is not None) != 2
        or stop_reserve < STOP_RESERVE and unstarted + (active_update is not None) > 0
    ):
        raise ValueError("Malformed v14 cost state")
    if active_update is not None and not 0 <= active_update <= 240:
        raise ValueError("Malformed active update count")
    if active_update is not None and active_update >= 40 and (
        slowest_block_seconds_per_update is None
        or not math.isfinite(slowest_block_seconds_per_update)
        or slowest_block_seconds_per_update <= 0
    ):
        raise ValueError("A completed block needs its measured pace")
    p_star = max((P_HISTORICAL, *completed))
    active_remaining = 0.0
    if active_update is not None:
        if active_tail_complete:
            active_remaining = 0.0
        elif active_update < 40:
            active_remaining = p_star
        else:
            active_remaining = (
                (240 - active_update) * slowest_block_seconds_per_update
                + max(p_star, observed_tail_seconds)
            )
    projected = elapsed + MARGIN * (
        unstarted * p_star + active_remaining + q_remaining + report_remaining + stop_reserve
    )
    return {
        "elapsed_new_v14_seconds": elapsed,
        "historical_v9_through_v13_lower_approximate_seconds_separate": HISTORICAL_LOWER_BOUND,
        "completed_pipeline_seconds": list(completed),
        "p_star_seconds": p_star,
        "unstarted_arms": unstarted,
        "active_update": active_update,
        "slowest_block_seconds_per_update": slowest_block_seconds_per_update,
        "active_remaining_seconds": active_remaining,
        "q_remaining_seconds": q_remaining,
        "report_remaining_seconds": report_remaining,
        "stop_reserve_seconds": stop_reserve,
        "projected_new_v14_seconds": projected,
        "ceiling_seconds": CEILING,
        "passed": projected <= CEILING and elapsed <= CEILING,
    }
