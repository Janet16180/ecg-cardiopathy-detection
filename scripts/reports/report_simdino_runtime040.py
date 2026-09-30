"""Write the runtime-only diagnosis from immutable timing receipts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ecg_experiment import ROOT
from ecg_experiment.simdino_runtime_analysis040 import NAME, aggregate


def render(result: dict[str, Any], diagnostic_seconds: float) -> str:
    """Render measured evidence separately from estimated future execution.

    Parameters
    ----------
    result : dict
        Complete or partial runtime analysis.
    diagnostic_seconds : float
        Latest charged new work, read separately from the sealed analysis snapshot.

    Returns
    -------
    str
        Standalone development-free runtime report.
    """
    complete = result["status"] == "complete_runtime_diagnostic"
    lines = ["# Experiment 040: runtime diagnosis", "",
             "All 18 full training fits remain unexecuted. This diagnosis measures runtime only; "
             "it does not authorize a new full training schedule.", ""]
    if complete:
        central = result["scenarios"]["central"]["total"]
        conservative = result["scenarios"]["conservative"]["total"]
        lines.extend([
            f"All six 200-update probes completed. Estimated remaining execution is "
            f"**{central / 3600:.3f} hours centrally** and **{conservative / 3600:.3f} hours "
            "in the conservative scenario**, including a separate 900-second reporting allowance.",
            "These are nonprobabilistic extrapolations, not measured full-run durations "
            "or confidence intervals.", ""])
    else:
        lines.extend([f"The timing study is incomplete: {result['packages_completed']}/6 packages "
                      "have receipts. No complete-schedule projection is available.", ""])
    admission = result["original_admission"]
    lines.extend(["## Original rejected projection", "",
                  "| Component | Seconds |", "| --- | ---: |"])
    for name, value in result["original_projection"].items():
        lines.append(f"| {name.replace('_', ' ')} | {value:.6f} |")
    lines.extend(["", f"The immutable admission used {admission['charged_seconds']:.6f} seconds "
                  f"already charged, {admission['remaining_projected_seconds']:.6f} seconds "
                  "remaining and a 3,600-second correction reserve: "
                  f"{admission['projected_combined_seconds']:.6f} seconds against 28,800.", "",
                  "The historical full readout includes feature extraction; adding another extraction "
                  "estimate overlaps costs. The new scenarios subtract the historical extraction proxy "
                  "before adding the measured new path. That subtraction is inferred from old profiles, "
                  "because the old readout did not separately time extraction.", "",
                  "Original checkpoint profiling times serialization and reload. Routine checkpoint "
                  "projections here include CPU capture and writing only. Active all-in training pace "
                  "would already include checkpoint/guard overhead; these projections instead use isolated "
                  "update timers and add checkpoint writing once. The original forecast also omitted "
                  "separate training-stage setup: its matched historical residual is retained here.", "",
                  "The original spent-plus-active accounting does not itself duplicate completed stage "
                  "charges. The 1.5 multiplier is an explicit conservative allowance, "
                  "not a measured error.", ""])
    if complete:
        lines.extend(["## Measured sustained timing", "",
                      "| Package | First 24 s/update | 25–100 | 101–200 selected | "
                      "Full feature pass 1 s | Pass 2 selected s |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for name, p in result["scenarios"]["packages"].items():
            lines.append(f"| {name} | {p['seconds_per_update_first24']:.6f} | "
                         f"{p['seconds_per_update_25_100']:.6f} | "
                         f"{p['seconds_per_update_101_200']:.6f} | "
                         f"{p['full_feature_pass_seconds'][0]:.6f} | "
                         f"{p['full_feature_pass_seconds'][1]:.6f} |")
        lines.extend(["", "The prespecified sustained window uses synchronized block wall times for updates "
                      "101–200, excluding checkpoint I/O. Each full fit uses "
                      "1,953 normal batches plus one measured 16-record update, with 20 checkpoint "
                      "boundaries. The second complete 15,359-record training feature pass is selected "
                      "regardless of whether it is faster. Its per-record rate is extrapolated to the "
                      "1,306 development records; no development waveform was timed or scored.", "",
                      "## Future execution scenarios", "",
                      "| Component | Central seconds | Conservative seconds |",
                      "| --- | ---: | ---: |"])
        scenarios = result["scenarios"]
        for name, value in scenarios["central"].items():
            lines.append(f"| {name.replace('_', ' ')} | {value:.6f} | "
                         f"{scenarios['conservative'][name]:.6f} |")
        lines.extend(["", "Central historical proxies use the mean of three matched 039 patch cells; "
                      "conservative historical proxies use their maximum with the 1.5 multiplier. "
                      "Other variable components are multiplied by 1.5. The 900-second reporting "
                      "allowance and 3,600-second correction reserve are each counted once.", "",
                      "The new all-package source-validation cost is counted for 27 train/readout/audit "
                      "stage gates. The separately measured time gate is counted for nine training starts "
                      "and 360 checkpoint pace checks. Nested validation reproduces the execution wrapper. "
                      "These costs are additional to matched 039 proxies, which lacked the 040 gate.", "",
                      "## Accounting", ""])
        lines.extend(["| Measured diagnostic subdivision | Seconds |", "| --- | ---: |"])
        for name, seconds in result["global_phase_seconds"].items():
            lines.append(f"| {name.replace('_', ' ')} | {seconds:.6f} |")
        for name, seconds in result["package_wall_seconds"].items():
            lines.append(f"| {name} whole package | {seconds:.6f} |")
        lines.extend([f"| Whole probe process body | {result['probe_wall_seconds']:.6f} |", ""])
        for label in ("central", "conservative"):
            projected = (result["parent_spent_seconds"] + diagnostic_seconds
                         + scenarios[label]["total"] + 3600)
            lines.append(f"{label.capitalize()} spent-plus-future scenario: {projected:.6f} seconds "
                         f"({projected / 3600:.3f} hours), including the correction reserve.")
    lines.extend(["", f"Closed parent work: {result['parent_spent_seconds']:.6f} seconds. "
                  f"New diagnostic work charged before this report: {diagnostic_seconds:.6f} seconds. "
                  "Package and phase times are subdivisions of whole-process work and are not added "
                  "again to the ledger. The final report invocation and any later audit are charged "
                  "separately; consult the live diagnostic ledger for the final total.", "",
                  "## Limits and evidence", "",
                  "Cold metadata validation, dataset loading and framework/model initialization are "
                  "outside update timers. Historical setup and CPU head/audit costs are proxies; "
                  "new objective features can change classifier convergence and full-run costs. "
                  "Two hundred updates, one initialization and repeated warm-cache feature reads "
                  "cannot establish three-seed, long-duration throughput. Later time gates also rehash "
                  "completed-cell artifacts, so their cost may exceed this pretraining measurement; "
                  "per-update elapsed-guard JSON overhead was not isolated. Drift is retained in all "
                  "four 50-update windows in the analysis receipt. No fastest-window selection is used.", "",
                  "Original 039/040 receipts and scientific code remain immutable. No head fitting, "
                  "development performance scoring, closed-set access, new label definition or "
                  "Experiment 008 restart is part of this diagnosis. Any repaired execution or full "
                  "training needs a prospective successor identity and user authorization.", "",
                  f"Evidence: `outputs/{NAME}/result.json`, package receipts, `projected_runtime.json` "
                  "and `day_ledger.json`; the analysis records hashes of parent, protocol, source "
                  "and measured inputs, plus the accounting snapshot used for its estimates.", ""])
    return "\n".join(lines)


def report(root: Path) -> Path:
    """Write the results report from saved execution evidence.

    Parameters
    ----------
    root : Path
        Repository containing runtime receipts.

    Returns
    -------
    Path
        Written Markdown report.
    """
    result = aggregate(root)
    ledger = json.loads((root / "outputs" / NAME / "day_ledger.json").read_text())
    path = root / "docs/experiment-040-runtime-diagnostic-results.md"
    path.write_text(render(result, ledger["total_seconds"]))
    return path


def main() -> None:
    """Account for analysis/report work in the new diagnostic ledger only."""
    from ecg_experiment.simdino_runtime_probe040 import charge

    began = time.monotonic()
    status = "failed"
    try:
        print(report(ROOT))
        status = "complete"
    finally:
        charge(ROOT, "analysis_and_report", time.monotonic() - began, status)


if __name__ == "__main__":
    main()
