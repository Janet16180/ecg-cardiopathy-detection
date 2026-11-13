"""Record the visual-check verdicts of one MLC-03 review directory into ``tally.csv``.

The reviewer looks at every flagged sheet and passes one character per flagged panel, in sheet and panel
order: ``y`` (the plot shows the defect the flag describes), ``n`` (it does not) or ``-`` (not viewed).
Notes are optional, ``panel_number=text`` with the 1-based position in that order. Unflagged panels can be
given the same way with ``--unflagged`` (``y`` meaning the plot shows the defect although the flag is not
set).

A flag is reliable when at least 20 of its 24 drawn flagged plots show the defect. When fewer than 24
records carry the flag, all are drawn and the same share applies (at least 20/24 of those drawn).

Example
-------
PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_record_review missing_samples \
    --flagged yyyyyyyyyyyyyyyyyyyyyyyy --note "3=lead dead after the gap"
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ecg_experiment.mlc03_signal_runs import OUTPUT

REVIEW = OUTPUT.parent / "review/signal"


def verdicts(text: str | None, count: int) -> list[str]:
    """
    Expand a verdict string.

    Parameters
    ----------
    text : str | None
        One ``y``, ``n`` or ``-`` (not viewed) per panel, or None when no panel was viewed.
    count : int
        Panels in the group.

    Returns
    -------
    list[str]
        ``yes``, ``no`` or ``not reviewed`` per panel.

    Raises
    ------
    ValueError
        If the string has another length or another character.
    """
    if text is None:
        return ["not reviewed"] * count
    if len(text) != count or set(text) - {"y", "n", "-"}:
        raise ValueError(f"expected {count} characters of y, n or -, got {text!r}")
    return [{"y": "yes", "n": "no", "-": "not reviewed"}[character] for character in text]


def main(argv: list[str] | None = None) -> None:
    """
    Write ``tally.csv`` and ``tally.json`` next to ``sample.csv``.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or None for ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", help="review directory name under review/signal")
    parser.add_argument("--flagged", required=True)
    parser.add_argument("--unflagged")
    parser.add_argument("--note", action="append", default=[], help="flagged panel=text")
    parser.add_argument("--unflagged-note", action="append", default=[], help="unflagged panel=text")
    args = parser.parse_args(argv)
    directory = REVIEW / args.directory
    sample = pd.read_csv(directory / "sample.csv")
    parts = []
    for group, text, notes in (("flagged", args.flagged, args.note), ("unflagged", args.unflagged,
                                                                       args.unflagged_note)):
        rows = sample[sample["group"] == group].reset_index(drop=True)
        rows["order"] = range(1, len(rows) + 1)
        rows["defect_visible"] = verdicts(text, len(rows))
        written = dict(note.split("=", 1) for note in notes)
        rows["note"] = [written.get(str(order), "") for order in rows["order"]]
        parts.append(rows)
    tally = pd.concat(parts, ignore_index=True)
    tally.to_csv(directory / "tally.csv", index=False)
    flagged = tally[tally["group"] == "flagged"]
    unflagged = tally[tally["group"] == "unflagged"]
    summary = {"review": args.directory, "flagged_drawn": len(flagged),
               "flagged_showing_defect": int((flagged["defect_visible"] == "yes").sum()),
               "unflagged_drawn": len(unflagged),
               "unflagged_showing_defect": int((unflagged["defect_visible"] == "yes").sum()),
               "unflagged_viewed": int((unflagged["defect_visible"] != "not reviewed").sum()),
               "reliable": bool(24 * (flagged["defect_visible"] == "yes").sum() >= 20 * len(flagged))}
    (directory / "tally.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
