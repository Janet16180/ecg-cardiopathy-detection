"""Render original complete or incomplete Experiment 040 outcomes from local receipts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ecg_experiment import ROOT
from ecg_experiment import simdino_study040 as study
from ecg_experiment.files import sha256_file, write_text_atomic
from ecg_experiment.simdino_analysis040 import (
    BUDGETS,
    CONTEXTS,
    GROUPS,
    OBJECTIVES,
    OUTPUT,
    SEEDS,
    aggregate,
    cell_path,
)


def _score_table(result: dict[str, Any], budget: str) -> list[str]:
    """Show individual seed scores rather than a probability ensemble."""
    lines = ["| Encoder/objective | Context | 39042 | 39043 | 39044 | Mean AUROC ± seed SD | Mean AP |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for group in GROUPS:
        label = "039 patch/CPC" if group == "patch" else f"Transformer/{group}"
        for context in CONTEXTS:
            item = result["scores"][f"{group}_{budget}_{context}"]
            seeds = " | ".join(f"{value:.5f}" for value in item["auroc"])
            lines.append(f"| {label} | {context} | {seeds} | "
                         f"{item['mean_auroc']:.5f} ± {item['sd_auroc']:.5f} | "
                         f"{item['mean_average_precision']:.5f} |")
    return lines


def _complete_results(result: dict[str, Any]) -> list[str]:
    """Render the six-comparison decision and explicitly exploratory secondary analyses."""
    inference = result["inference"]
    promising = [name for name, item in inference["contrasts"].items() if item["promising"]]
    combinations = [context for context, decision in inference["combined_benefit"].items() if decision]
    combination = ", ".join(combinations) if combinations else "neither context"
    lines = ["All 18 original fits completed and have audited development-only readouts.",
             f"The frozen six-comparison rule identified {len(promising)} promising primary comparisons.",
             f"The hybrid met the rule against both component objectives under {combination}.", "",
             "## Primary limited-label results", "", *_score_table(result, "limited"), "",
             "## Paired primary comparisons", "",
             "| Comparison | Mean ΔAUROC | Paired patient 95% interval | Simultaneous 95% band | Promising |",
             "| --- | ---: | --- | --- | --- |"]
    for name, item in inference["contrasts"].items():
        if item["primary"]:
            lines.append(f"| {name} | {item['difference']:+.5f} | "
                         f"[{item['ci_low']:+.5f}, {item['ci_high']:+.5f}] | "
                         f"[{item['simultaneous_low']:+.5f}, {item['simultaneous_high']:+.5f}] | "
                         f"{'yes' if item['promising'] else 'no'} |")
    lines += ["", "The cpc-minus-patch contrasts test the frontend; hybrid-minus-component contrasts test",
              "the objective within the identical Transformer/context package. A promising combination",
              "requires both component comparisons to pass under the same context.", "",
              f"Intervals use {inference['draws']:,} shared whole-patient draws, seed {inference['seed']}; "
              f"{inference['skipped_draws']} single-class draws were skipped. The six-contrast approximate",
              f"simultaneous band has radius {inference['simultaneous_radius']:.5f}. Each draw averages",
              "three per-seed AUROC differences; probabilities are never averaged into an ensemble.",
              "Seed SD is separate from patient uncertainty, which conditions on the three trained seeds.",
              "",
              "## Full-label results (secondary)", "", *_score_table(result, "full"), "",
              "## Secondary comparisons", "",
              "| Comparison | Mean ΔAUROC | Unadjusted paired patient 95% interval |",
              "| --- | ---: | --- |"]
    secondary = {name: item for name, item in inference["contrasts"].items() if not item["primary"]}
    secondary.update(result["interactions"]["interactions"])
    for name, item in secondary.items():
        lines.append(f"| {name} | {item['difference']:+.5f} | "
                     f"[{item['ci_low']:+.5f}, {item['ci_high']:+.5f}] |")
    lines += ["", "Interaction signs describe the first-minus-second objective advantage under mLSTM minus",
              "the corresponding advantage under GRU. Secondary intervals have no multiplicity adjustment.",
              "AP and its paired mean differences are descriptive; no AP interval or selection rule is used.",
              "",
              "| Exploratory AP comparison | Mean per-seed ΔAP |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value:+.5f} |"
                 for name, value in result["average_precision_differences"].items())
    return lines


def _partial_results(result: dict[str, Any]) -> list[str]:
    """Show only audited observations and explain why no complete-family decision exists."""
    lines = [f"Study incomplete: {result['completed_fits']} of 18 fits have audited readouts.",
             "No complete-family superiority or combined-benefit decision is available.", "",
             "| Audited objective | Seed | Context | Label budget | Development AUROC | AP |",
             "| --- | ---: | --- | --- | ---: | ---: |"]
    for cell in result["completed_cells"]:
        for budget in BUDGETS:
            for context in CONTEXTS:
                score = cell["scores"][budget][context]
                lines.append(f"| {cell['objective']} | {cell['seed']} | {context} | {budget} | "
                             f"{score['auroc']:.5f} | {score['average_precision']:.5f} |")
    lines += ["", "## Incomplete cells", "", "| Objective | Seed | Latest executed stage/status |",
              "| --- | ---: | --- |"]
    for cell in result["unfinished_cells"]:
        stages = [f"{name}: {item['status']}" for name, item in cell["latest_stages"].items() if item]
        lines.append(f"| {cell['objective']} | {cell['seed']} | {'; '.join(stages) or 'not executed'} |")
    for name in ("admission_receipt", "status_receipt"):
        if result[name] is not None:
            lines += ["", f"Recorded {name.replace('_', ' ')}:", "", "```json",
                      json.dumps(result[name], indent=2, sort_keys=True), "```"]
    return lines


def _diagnostics(root: Path, result: dict[str, Any]) -> list[str]:
    """Report every available diagnosis and any severe trigger without a completed diagnosis."""
    lines = ["", "## Failures and one-shot diagnoses", ""]
    diagnosed = set()
    for path in sorted((root / OUTPUT / "diagnostics").glob("*/diagnostic.json")):
        receipt = json.loads(path.read_text())
        diagnosed.add((receipt["objective"], receipt["context"]))
        lines.append(f"- {receipt['objective']}/{receipt['context']}: {receipt['conclusion']} "
                     f"Receipt SHA-256 `{sha256_file(path)}`.")
    failures = result.get("severe_failures", [])
    for item in failures:
        state = "diagnosed" if (item["objective"], item["context"]) in diagnosed else "diagnosis outstanding"
        lines.append(f"- {item['objective']}/{item['context']}, seed {item['seed']}: "
                     f"AUROC {item['auroc']:.5f}, matched-control difference "
                     f"{item['difference']:+.5f}; {state}.")
    if not diagnosed and not failures:
        lines.append("No diagnosis receipt is available. Completed-study score triggers, when present, are")
        lines.append("listed here. An incomplete suite cannot exclude numerical or collapse failures.")
    lines += ["", "Original outcomes are retained. Any separately authorized correction has its own protocol",
              "and receipts and cannot replace the original primary family."]
    return lines


def _resources(root: Path) -> list[str]:
    """Render measurements without substituting projections for completed training time."""
    lines = ["", "## Measured resources and provenance", "",
             "| Objective | Seed | Context | Train s | Peak GB | Total params | Student | Active student |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    protocols = set()
    for seed in SEEDS:
        for objective in OBJECTIVES:
            path = study.directory(root, objective, seed)
            manifest = {}
            if (path / "manifest.json").exists():
                manifest = json.loads((path / "manifest.json").read_text())
                protocols.add(manifest["protocol_commit"])
            train_path, profile_path = path / "training.json", path / "profile.json"
            training = json.loads(train_path.read_text()) if train_path.exists() else {}
            profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
            for context in CONTEXTS:
                arm = training.get("arms", {}).get(context, {})
                measured = profile.get("arms", {}).get(context, {})
                seconds = f"{arm['elapsed_seconds']:.2f}" if arm else "incomplete"
                peak = f"{measured['peak_gpu_memory_bytes'] / 1e9:.3f}" if measured else "unprofiled"
                count = str(arm.get("parameter_count", measured.get("parameter_count", "unmeasured")))
                spec = manifest.get("architectures", {}).get(context, {})
                student = spec.get("student_parameters", "unmeasured")
                active = spec.get("active_student_parameters", "unmeasured")
                lines.append(f"| {objective} | {seed} | {context} | {seconds} | {peak} | {count} | "
                             f"{student} | {active} |")
    lines += ["", "Total parameters include the EMA teacher, which is inactive for CPC-only. Active student",
              "counts exclude the unused CPC heads for SimDINOv2-style-only. For comparison, 039 patch/CPC",
              "has no teacher: its total, student and active counts coincide."]
    reference_path = cell_path(root, "patch", SEEDS[0]) / "manifest.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    for context, spec in reference.get("architectures", {}).items():
        lines.append(f"The 039 patch/{context} reference has {spec['total_parameters']:,} active parameters.")
    own_seconds, total_seconds = study.ledger(root)["total_seconds"], study.combined_seconds(root)
    lines += ["", f"Charged 040 execution before this report: {own_seconds:.2f} seconds.",
              f"Combined charged 039+040 execution: {total_seconds:.2f} seconds against 28,800.",
              "The final ledger also charges report generation and final audit after this snapshot.",
              "Historical cache construction is reused and is not charged twice.", "",
              "Executed 040 protocol commit IDs: " + (", ".join(f"`{value}`" for value in sorted(protocols))
                                                      or "no executable manifest yet") + ".",
              "Historical identities are preserved; a metadata-only Git rewrite does not change receipt IDs."]
    return lines


def report(root: Path = ROOT) -> Path:
    """Write results from verified local outputs without making a partial-family decision.

    Parameters
    ----------
    root : Path
        Repository containing original local study receipts.

    Returns
    -------
    Path
        Results report; callers must charge this execution to the shared day ledger.
    """
    result = aggregate(root)
    complete = result["status"] == "complete_original_study_development_only"
    lines = ["# Experiment 040: causal Transformer and SimDINOv2-style results", ""]
    lines += _complete_results(result) if complete else _partial_results(result)
    lines += _diagnostics(root, result) + _resources(root)
    lines += ["", "## Interpretation and scope", "",
              "The 25k v4 cohort is byte-identical to its v3 manifest and contains no EchoNext records.",
              "The compact causal Transformer has a 49-sample frontend support and independent five-second",
              "halves. Its clean CPC path preserves future-target separation. SimDINOv2-style training adds",
              "two raw-masked student views and a clean EMA teacher; its global mean representation has 256",
              "coordinates, while frozen downstream mean/max features have 512. These choices adapt xECG's",
              "objective; they do not reproduce its bidirectional large-model recipe. Mask corruption is not",
              "a guarantee that focal diagnostic information survives consistency training.", "",
              "Completed fits use 250,000 exposures / 1,954 updates, batch 128 (final 16), float32, AdamW",
              "1e-4 / decay 0.01 and clipping at 1. Each readout uses the final student, train-only scaling",
              "and logistic C=0.01 with 1,518 or 15,359 labels. No coefficient or checkpoint is selected by",
              "development scores. The development cohort has 1,306 ECGs from 1,173 patients.", "",
              "Patient intervals condition on three trained seeds. These repeatedly inspected development",
              "patients cannot establish confirmatory clinical benefit. The label is an ECG-annotation",
              "proxy; no clinical referral threshold or age subgroup was introduced. This study uses no",
              "SPH, Challenge calibration/test, EchoNext test or final frozen-test evaluation. Experiment",
              "008 remains deferred. The candidate screening pipeline is unchanged.", "",
              f"Evidence remains locally under `{OUTPUT}/` and the matched 039 patch control directories.",
              "Follow-ups belong in the ranked backlog. Reproduction requires the recorded protocol/source",
              "identity and original artifacts; completed score receipts must not be overwritten."]
    if complete:
        lines += ["", f"Aggregate receipt SHA-256: `{sha256_file(root / OUTPUT / 'aggregate.json')}`."]
    destination = root / "docs/experiment-040-cpc-simdino-results.md"
    write_text_atomic(destination, "\n".join(lines) + "\n")
    return destination


def main() -> None:
    """Generate and charge the report, including aggregation when it has not yet run."""
    started = time.monotonic()
    status = "failed"
    try:
        print(report())
        status = "complete"
    finally:
        study.charge(ROOT, "analysis", 0, "report", time.monotonic() - started, status, cell=False)


if __name__ == "__main__":
    main()
