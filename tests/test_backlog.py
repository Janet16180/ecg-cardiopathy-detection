"""Checks of the experiment backlog ranking."""

import pytest

from ecg_experiment.backlog import markdown_table, rank, score


def candidate(identifier: str, value: int = 3, clarity: float = 1.0, build_hours: float = 4.0,
              blocked_by: tuple[str, ...] = (), status: str = "open") -> dict:
    return {"id": identifier, "title": identifier, "value": value, "clarity": clarity, "gpu_hours": 0.0,
            "cpu_hours": 0.0, "build_hours": build_hours, "blocked_by": list(blocked_by), "status": status}


def test_score_is_value_times_clarity_per_root_hour():
    assert score(candidate("a", value=4, clarity=0.5, build_hours=4.0), dependents=0) == pytest.approx(1.0)
    assert score(candidate("a", value=4, clarity=0.5, build_hours=4.0), dependents=2) == pytest.approx(1.4)


def test_unblocked_candidates_rank_before_blocked_ones():
    ranked = rank([candidate("cheap_blocked", value=5, build_hours=1.0, blocked_by=("base",)),
                   candidate("base", value=2, build_hours=9.0)])
    assert [item["id"] for item in ranked] == ["base", "cheap_blocked"]
    assert ranked[0]["dependents"] == 1
    assert ranked[1]["blocked"] == ["base"]


def test_finished_blockers_no_longer_block_and_done_items_are_dropped():
    ranked = rank([candidate("next", blocked_by=("base",)), candidate("base", status="done")])
    assert [item["id"] for item in ranked] == ["next"]
    assert ranked[0]["blocked"] == []


def test_table_shows_kind_with_experiment_as_default():
    table = markdown_table(rank([candidate("a"), {**candidate("b"), "kind": "wild"}]))
    assert "| a | experiment |" in table
    assert "| b | wild |" in table


def test_external_conditions_block_until_removed():
    ranked = rank([candidate("wait", blocked_by=("@ningbo_download",))])
    assert ranked[0]["blocked"] == ["@ningbo_download"]
