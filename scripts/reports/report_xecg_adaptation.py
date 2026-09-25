"""Report all four xECG adaptation arms with paired patient comparisons."""

import argparse
import json
from pathlib import Path
from typing import Any

from ecg_experiment.evaluation import paired_comparison
from ecg_experiment.files import write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"a": "A: continuation", "b": "B: Gram retention",
          "c": "C: uniform local retention", "d": "D: rarity-weighted local retention"}
RELEASE_LABEL = "Released xECG (007)"
BUDGETS = (("full", "All 15,360 training labels"), ("ten_percent", "Fixed 1,518-label subset"))
TABLE_ORDER = ("release", "a", "b", "c", "d")
COMPARISONS = (("d", "c"), ("b", "a"), ("c", "b"), ("a", "release"))
DEFAULT_BOOTSTRAP = 500

INTRODUCTION = [
    "# Experiment 008: local-feature retention during xECG adaptation", "",
    "Four encoders start from the same released xECG and use matched records, masks, adaptation updates, "
    "and downstream training. The released reference receives no adaptation. All adaptation arms finish "
    "before any of their test evaluations.", "",
    "D minus C is the primary weighting comparison. B minus A tests relational retention; C minus B "
    "tests pointwise retention. These are exploratory comparisons from one seed on a previously used "
    "PTB-XL test cohort. Patient bootstrap intervals do not measure retraining variation.", "",
]
CALIBRATION_NOTE = [
    "", "Thresholds and Platt calibration are fitted on calibration patients separately for each "
    "model, then frozen for test evaluation. Lower Brier score is better.", "",
    "| Paired comparison | AUROC difference | Patient-bootstrap 95% interval |",
    "| --- | ---: | --- |",
]
INTERPRETATION = [
    "## Runtime and interpretation", "",
    "GPU profile measurements are in profile_cuda.json; adaptation histories and checkpoints are under "
    "pretrain/, and supervised runtimes and selected epochs are in each transfer's config.json. "
    "The profiles are readiness measurements; completed run histories determine actual training cost.", "",
    "A favorable result does not establish a better encoder architecture or university screening "
    "performance. Rarity weights are an unsupervised feature heuristic and may emphasize artifacts. "
    "Keep negative results and inspect the saved training-only diagnostics. No test-selected winner "
    "is promoted automatically.",
    "",
]


def transfer_paths(output_dir: Path, baseline_dir: Path, budget: str) -> dict[str, Path]:
    """
    Locate the completed transfer runs of every arm and the released reference.

    Parameters
    ----------
    output_dir : Path
        Experiment 008 output directory.
    baseline_dir : Path
        Experiment 007 directory with the released xECG transfers.
    budget : str
        Label budget, ``"full"`` or ``"ten_percent"``.

    Returns
    -------
    dict[str, Path]
        Run directory per arm, plus ``"release"``.

    Raises
    ------
    FileNotFoundError
        If any run lacks ``complete.json``.
    """
    paths = {arm: output_dir / "transfer" / arm / f"xecg_{budget}_seed42" for arm in LABELS}
    paths["release"] = baseline_dir / f"xecg_{budget}_seed42"
    for path in paths.values():
        if not (path / "complete.json").exists():
            raise FileNotFoundError(f"Cannot report an incomplete xECG transfer: {path}")
    return paths


def metric_rows(paths: dict[str, Path]) -> list[str]:
    """
    Format one test-metric table row per encoder.

    Parameters
    ----------
    paths : dict[str, Path]
        Run directory per arm.

    Returns
    -------
    list[str]
        Markdown table rows.
    """
    lines = []
    for arm in TABLE_ORDER:
        metrics = json.loads((paths[arm] / "metrics.json").read_text())["test"]
        label = LABELS.get(arm, RELEASE_LABEL)
        lines.append(f"| {label} | {metrics['auroc']:.4f} | {metrics['average_precision']:.4f} | "
                     f"{metrics['sensitivity']:.4f} | {metrics['specificity']:.4f} | "
                     f"{metrics['brier']:.4f} |")
    return lines


def report(output_dir: Path, bootstrap: int = DEFAULT_BOOTSTRAP,
           baseline_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Write the Experiment 008 report and its paired comparisons.

    Parameters
    ----------
    output_dir : Path
        Experiment 008 output directory; ``report.md`` and
        ``paired_comparisons.json`` are written here.
    bootstrap : int
        Patient bootstrap repeats per comparison.
    baseline_dir : Path | None
        Experiment 007 directory; defaults to ``outputs/experiment007_xecg``.

    Returns
    -------
    dict[str, dict[str, Any]]
        Paired comparisons keyed by ``"{budget}_{left}_minus_{right}"``.

    Raises
    ------
    FileNotFoundError
        If any transfer run is incomplete.
    """
    output_dir = Path(output_dir)
    baseline_dir = Path(baseline_dir) if baseline_dir else ROOT / "outputs/experiment007_xecg"
    comparisons = {}
    lines = list(INTRODUCTION)
    for budget, title in BUDGETS:
        paths = transfer_paths(output_dir, baseline_dir, budget)
        lines += [f"## {title}", "",
                  "| Encoder | AUROC | AP | Sensitivity | Specificity | Brier |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        lines += metric_rows(paths)
        lines += CALIBRATION_NOTE
        for left, right in COMPARISONS:
            result = paired_comparison(paths[left], paths[right], bootstrap)
            comparisons[f"{budget}_{left}_minus_{right}"] = result
            auc = result["auroc"]
            lines.append(f"| {left.upper()} minus {right.upper()} | {auc['difference']:+.4f} | "
                         f"[{auc['ci95'][0]:+.4f}, {auc['ci95'][1]:+.4f}] |")
        lines.append("")
    lines += INTERPRETATION
    write_json_atomic(output_dir / "paired_comparisons.json", comparisons)
    (output_dir / "report.md").write_text("\n".join(lines))
    return comparisons


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiment008_vision_ssl")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Write the Experiment 008 report from the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    report(args.output_dir, args.bootstrap)


if __name__ == "__main__":
    main()
