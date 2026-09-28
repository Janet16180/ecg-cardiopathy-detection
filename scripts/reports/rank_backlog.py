"""Print the experiment backlog ranked by expected decision value per hour."""

import argparse
from pathlib import Path

from ecg_experiment.backlog import load, markdown_table, rank

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Rank the backlog file and print a Markdown table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backlog", type=Path, default=ROOT / "docs/experiment-backlog.json")
    args = parser.parse_args()
    print(markdown_table(rank(load(args.backlog))))


if __name__ == "__main__":
    main()
