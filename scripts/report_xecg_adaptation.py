"""Report all four xECG adaptation arms with paired patient comparisons."""

import argparse
import json
from pathlib import Path

from scripts.run_cpc_experiment import paired_comparison
from scripts.run_xecg_finetune import save_atomic_json

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"a": "A: continuation", "b": "B: Gram retention",
          "c": "C: uniform local retention", "d": "D: rarity-weighted local retention"}


def report(output_dir: Path, bootstrap: int = 500, baseline_dir: Path | None = None):
    output_dir = Path(output_dir)
    baseline_dir = Path(baseline_dir) if baseline_dir else ROOT / "outputs/experiment007_xecg"
    comparisons = {}
    lines = ["# Experiment 008: local-feature retention during xECG adaptation", "",
        "Four encoders start from the same released xECG and use matched records, masks, adaptation updates, "
        "and downstream training. The released reference receives no adaptation. All adaptation arms finish "
        "before any of their test evaluations.", "",
        "D minus C is the primary weighting comparison. B minus A tests relational retention; C minus B "
        "tests pointwise retention. These are exploratory comparisons from one seed on a previously used "
        "PTB-XL test cohort. Patient bootstrap intervals do not measure retraining variation.", ""]
    for budget, title in (("full", "All 15,360 training labels"), ("ten_percent", "Fixed 1,518-label subset")):
        paths = {arm: output_dir / "transfer" / arm / f"xecg_{budget}_seed42" for arm in LABELS}
        paths["release"] = baseline_dir / f"xecg_{budget}_seed42"
        for path in paths.values():
            if not (path / "complete.json").exists():
                raise FileNotFoundError(f"Cannot report an incomplete xECG transfer: {path}")
        lines += [f"## {title}", "",
                  "| Encoder | AUROC | AP | Sensitivity | Specificity | Brier |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for arm in ("release", "a", "b", "c", "d"):
            metrics = json.loads((paths[arm] / "metrics.json").read_text())["test"]
            label = LABELS.get(arm, "Released xECG (007)")
            lines.append(f"| {label} | {metrics['auroc']:.4f} | {metrics['average_precision']:.4f} | "
                         f"{metrics['sensitivity']:.4f} | {metrics['specificity']:.4f} | {metrics['brier']:.4f} |")
        lines += ["", "Thresholds and Platt calibration are fitted on calibration patients separately for each "
                  "model, then frozen for test evaluation. Lower Brier score is better.", "",
                  "| Paired comparison | AUROC difference | Patient-bootstrap 95% interval |",
                  "| --- | ---: | --- |"]
        for left, right in (("d", "c"), ("b", "a"), ("c", "b"), ("a", "release")):
            result = paired_comparison(paths[left], paths[right], bootstrap)
            comparisons[f"{budget}_{left}_minus_{right}"] = result
            auc = result["auroc"]
            lines.append(f"| {left.upper()} minus {right.upper()} | {auc['difference']:+.4f} | "
                         f"[{auc['ci95'][0]:+.4f}, {auc['ci95'][1]:+.4f}] |")
        lines.append("")
    lines += ["## Runtime and interpretation", "",
        "GPU profile measurements are in profile_cuda.json; adaptation histories and checkpoints are under "
        "pretrain/, and supervised runtimes and selected epochs are in each transfer's config.json. "
        "The profiles are readiness measurements; completed run histories determine actual training cost.", "",
        "A favorable result does not establish a better encoder architecture or university screening performance. "
        "Rarity weights are an unsupervised feature heuristic and may emphasize artifacts. Keep negative "
        "results and inspect the saved training-only diagnostics. No test-selected winner is promoted automatically.", ""]
    save_atomic_json(output_dir / "paired_comparisons.json", comparisons)
    (output_dir / "report.md").write_text("\n".join(lines))
    return comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiment008_vision_ssl")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    report(args.output_dir, args.bootstrap)
