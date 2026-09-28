"""Rank candidate experiments by expected decision value per hour of work.

The backlog is a JSON list of candidates, each with ``value`` (1 to 5: how much either outcome would change
what the project does next), ``clarity`` (0 to 1: chance the result is decisive at the planned sample size),
estimated ``gpu_hours``, ``cpu_hours`` and ``build_hours`` (writing, checking and reviewing code and
protocol), and ``blocked_by`` (IDs of candidates or external conditions that must finish first). The rubric
is in ``docs/experiment-priorities.md``. An optional ``kind`` separates experiments from repository tasks
(``repo``); both are ranked together.
"""

import math
from pathlib import Path
from typing import Any

from .files import read_json

DEPENDENT_BONUS = 0.2


def hours(candidate: dict[str, Any]) -> float:
    """
    Total estimated hours of one candidate.

    Parameters
    ----------
    candidate : dict[str, Any]
        One backlog entry.

    Returns
    -------
    float
        GPU, CPU and build hours added together.
    """
    return candidate["gpu_hours"] + candidate["cpu_hours"] + candidate["build_hours"]


def score(candidate: dict[str, Any], dependents: int) -> float:
    """
    Compute expected decision value per square-root hour, with a bonus for unblocking others.

    The square root keeps long but important studies from being buried under trivial cheap ones.

    Parameters
    ----------
    candidate : dict[str, Any]
        One backlog entry.
    dependents : int
        Number of open candidates that list this one in ``blocked_by``.

    Returns
    -------
    float
        Priority score; higher runs first.
    """
    return candidate["value"] * candidate["clarity"] * (1 + DEPENDENT_BONUS * dependents) / math.sqrt(
        hours(candidate))


def rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Score open candidates and order them: unblocked first, then by score.

    Parameters
    ----------
    candidates : list[dict[str, Any]]
        Backlog entries. Entries with ``status`` ``done`` or ``dropped`` are left out.

    Returns
    -------
    list[dict[str, Any]]
        Open entries with added ``score``, ``hours``, ``dependents`` and ``blocked`` (the open blockers).
    """
    open_ids = {item["id"] for item in candidates if item["status"] not in ("done", "dropped")}
    ranked = []
    for item in candidates:
        if item["id"] not in open_ids:
            continue
        dependents = sum(item["id"] in other["blocked_by"] for other in candidates if other["id"] in open_ids)
        blocked = [name for name in item["blocked_by"] if name in open_ids or name.startswith("@")]
        ranked.append({**item, "score": score(item, dependents), "hours": hours(item),
                       "dependents": dependents, "blocked": blocked})
    return sorted(ranked, key=lambda item: (bool(item["blocked"]), -item["score"]))


def markdown_table(ranked: list[dict[str, Any]]) -> str:
    """
    Format ranked candidates as a Markdown table.

    Parameters
    ----------
    ranked : list[dict[str, Any]]
        Output of ``rank``.

    Returns
    -------
    str
        Table with rank, ID, kind, title, value, clarity, hours, score and blockers.
    """
    lines = ["| Rank | ID | Kind | Candidate | Value | Clarity | Hours | Score | Waiting on |",
             "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for position, item in enumerate(ranked, start=1):
        waiting = ", ".join(item["blocked"]) or "-"
        kind = item.get("kind", "experiment")
        lines.append(f"| {position} | {item['id']} | {kind} | {item['title']} | {item['value']} "
                     f"| {item['clarity']:.1f} | {item['hours']:.1f} | {item['score']:.2f} | {waiting} |")
    return "\n".join(lines)


def load(path: Path) -> list[dict[str, Any]]:
    """
    Read the backlog file.

    Parameters
    ----------
    path : Path
        JSON file with a ``candidates`` list.

    Returns
    -------
    list[dict[str, Any]]
        Backlog entries.
    """
    return read_json(path)["candidates"]
