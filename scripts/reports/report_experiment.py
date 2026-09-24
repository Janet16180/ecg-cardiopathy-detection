"""Build a reviewable report and scientific figure from completed experiment outputs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from ecg_experiment.files import write_json_atomic, write_text_atomic  # noqa: E402

NAMES = {
    "cnn_supervised": "CNN, supervised",
    "transformer_supervised": "Transformer, supervised",
    "mae_finetuned": "Compact MAE + fine-tuning",
    "jepa_finetuned": "Compact JEPA + fine-tuning",
    "hubert-small_linear": "HuBERT-small, frozen",
    "ecg-fm_linear": "ECG-FM, frozen",
    "ecg-jepa_linear": "Published ECG-JEPA, frozen",
    "hubert-small_finetuned": "HuBERT-small, fine-tuned",
    "ecg-fm_finetuned": "ECG-FM, fine-tuned",
    "ecg-fm_adapted_finetuned": "ECG-FM, adapted + fine-tuned",
    "ecg-fm_pooled_adapted_finetuned": "ECG-FM, pooled adaptation",
    "lead_multiscale_supervised": "Lead multiscale, supervised",
    "lead_multiscale_latent": "Lead multiscale, latent SSL",
    "lead_multiscale_innovation": "Lead multiscale, innovation SSL",
}
CUSTOM_ARMS = ("lead_multiscale_supervised", "lead_multiscale_latent", "lead_multiscale_innovation")
PRIMARY_SEED = 42

# The fixed Experiment 001 cohort quoted in the report; results must match it.
TEST_RECORDS = 1896
TEST_ABNORMAL = 1195
TEST_NORMAL = 701
CALIBRATION_RECORDS = 564
TRAINING_LABELS = 1518
FULL_TRAINING_LABELS = 15360

TOP_CURVES = 4
REFERENCE_MODEL = "cnn_supervised"
CALIBRATION_BINS = 8
TARGET_SENSITIVITY = 0.95

SCREENING_SECTION = [
    "", "## Hypothetical low-prevalence screening", "",
    "The following arithmetic assumes 1% prevalence and unchanged sensitivity/specificity. "
    "Neither assumption has been established in students. The 1% prevalence scenario is unrelated "
    "to the original 1% annotation rate. AUPRC, precision, calibration, and referral volume measured "
    "on PTB-XL cannot be transferred directly to that population.", "",
    "| Model (seed 42) | True positives / 1,000 | False positives / 1,000 | Misses / 1,000 "
    "| Positive predictive value |",
    "| --- | --- | --- | --- | --- |",
]
METHODS_SECTION = [
    "", "## Methods and limits", "",
    "- Compact models use all 12 leads at 100 Hz for 10 seconds, with training-only lead RMS "
    "scaling. The transformer has three 96-dimensional layers and 250 ms patches. MAE learns "
    "masked waveform reconstruction; the JEPA-inspired model predicts EMA-teacher latent targets "
    "and applies a variance penalty. Both receive 50 SSL epochs on training waveforms only, then "
    "the same downstream optimizer as the supervised transformer. They are our small experimental "
    "implementations, not reproductions of published ST-MEM or ECG-JEPA.",
    "- Published encoders retain their own preprocessing and representations. HuBERT and ECG-FM "
    "aggregate two five-second views; source revisions, checkpoint hashes, and preprocessing are "
    "stored with embeddings and model configs. The HuBERT and ECG-FM public SSL checkpoints previously "
    "encountered PTB-XL waveforms, so those results are not an unseen-waveform external test. We use "
    "SSL-only checkpoints rather than supervised Cardio-Learning weights.",
    "- Additional ECG-FM adaptation is a bounded pilot using a CMSC-style temporal contrastive "
    "objective, not a reproduction of its complete pretraining loss. Both arms use one pass "
    "through their training pools, chosen from runtime profiling before downstream evaluation; "
    "this is not a test of extensively optimized continued pretraining. Adaptation checkpoints and "
    "their exact waveform sources are recorded under `outputs/experiment001/`. A pooled arm, where "
    "present, adds a verified Georgia pilot subset without using its diagnoses. Georgia was already "
    "represented in the released encoder's historical pretraining data. Matching the two five-second "
    "views can suppress transient findings present in only one half, so this objective is a hypothesis "
    "to evaluate.",
    "- The custom eight-lead multiscale encoder has matched scratch, ordinary latent-prediction, "
    "and added lead-difference-prediction arms. It is a custom hypothesis with unverified novelty. "
    "Its two SSL arms use the same 20-epoch budget, chosen from timing measurements before "
    "downstream evaluation. See [the design note](custom-architecture.md) for masking, target "
    "construction, related work, and the primary ablation.",
    "- Published ECG-JEPA uses the authors' eight-lead, 250 Hz preprocessing and frozen encoder. "
    "Its repository lists Shaoxing/CODE15 pretraining commands; the exact released checkpoint's "
    "complete data provenance has not been independently verified. It is not established here "
    "as a completely unexposed external-test control.",
    "- All pretrained systems have different historical datasets and compute budgets. These "
    "measurements compare practical systems; they do not isolate architecture from pretraining scale.",
    "- Calibration and the illustrative 95% sensitivity target are research choices. The resulting "
    "probabilities concern the proxy label in this evaluation population, not confirmed heart disease "
    "or a validated clinical referral probability. Local expert-reviewed data remain necessary.", "",
    "The original 1% training-label regime has not been tested. The additional fully labeled "
    "public-data runs are reported separately; the study does not establish prospective local "
    "annotation savings.", "",
    "## Artifacts", "",
    "Detailed metrics, patient bootstrap intervals, test probabilities, model weights, configs, "
    "and training histories are under `outputs/experiment001/`. Raw waveforms and predictions "
    "are git-ignored; this summary and figure are shareable repository artifacts.", "",
    "![Model comparison](experiment001-results.png)", "",
]


def display_name(model: str) -> str:
    """
    Return the report label of a model, or its identifier if it has none.

    Parameters
    ----------
    model : str
        Model identifier.

    Returns
    -------
    str
        Human-readable name.
    """
    return NAMES.get(model, model)


def load_records(input_dir: Path) -> list[dict[str, Any]]:
    """
    Read every completed run, with its directory and optional config.

    Parameters
    ----------
    input_dir : Path
        Directory whose subdirectories hold ``metrics.json``.

    Returns
    -------
    list[dict[str, Any]]
        Metric records in directory listing order.

    Raises
    ------
    ValueError
        If no completed run exists.
    """
    records = []
    for path in input_dir.glob("*/metrics.json"):
        record = json.loads(path.read_text())
        record["directory"] = str(path.parent)
        config_path = path.parent / "config.json"
        record["config"] = json.loads(config_path.read_text()) if config_path.exists() else {}
        records.append(record)
    if not records:
        raise ValueError("No completed experiment results")
    return records


def check_cohort(records: list[dict[str, Any]]) -> None:
    """
    Check that every run was evaluated on the cohort quoted in the report.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Metric records.

    Raises
    ------
    ValueError
        If a run's test or calibration counts differ from the report text.
    """
    for record in records:
        test = record["test"]
        if (test["n"] != TEST_RECORDS or test["tp"] + test["fn"] != TEST_ABNORMAL
                or test["tn"] + test["fp"] != TEST_NORMAL):
            raise ValueError(f"Run {record['model']} used a different test cohort than the report")
        if record["calibration"]["records"] != CALIBRATION_RECORDS:
            raise ValueError(f"Run {record['model']} used a different calibration cohort than the report")


def load_paired(input_dir: Path) -> dict[str, Any] | None:
    """
    Read paired adaptation comparisons, appending the pooling comparison if present.

    Parameters
    ----------
    input_dir : Path
        Experiment output directory.

    Returns
    -------
    dict[str, Any] | None
        Method description and comparisons, or ``None`` when none were computed.
    """
    paired_path = input_dir / "paired_adaptation_comparisons.json"
    if not paired_path.exists():
        return None
    paired = json.loads(paired_path.read_text())
    pooling_path = input_dir / "paired_pooling_comparison.json"
    if pooling_path.exists():
        paired["comparisons"].extend(json.loads(pooling_path.read_text())["comparisons"])
    return paired


def introduction_lines() -> list[str]:
    """
    Return the title, data description and primary table header.

    Returns
    -------
    list[str]
        Markdown lines.
    """
    prevalence = 100 * TEST_ABNORMAL / TEST_RECORDS
    return [
        "# Experiment 001 results: 10% labeled training patients", "",
        "These are measured results on the PTB-XL v1.0.3 diagnostic-abnormality proxy. "
        "They do not measure disease detection or referral safety in university students. "
        "This is an exploratory study: later experimental arms were designed after inspecting earlier "
        "results, so the overall comparison was not prospectively locked.", "",
        "## Data and evaluation", "",
        f"The primary run exposes {TRAINING_LABELS:,} training ECG labels from 1,335 patients "
        "(approximately 10% of eligible training patients). All 17,418 training ECGs are available "
        f"to our SSL models. The fixed evaluation set contains {TEST_RECORDS:,} ECGs from 1,653 "
        f"patients, with {TEST_ABNORMAL:,} abnormal and {TEST_NORMAL} normal proxy labels "
        f"({prevalence:.1f}% abnormal prevalence). Ambiguous cases are excluded: 313 from validation "
        f"and 302 from test. Test coverage is 86.3% ({TEST_RECORDS:,}/2,198); the remaining 13.7% "
        "are outside this evaluation. Eligibility uses existing annotations and cannot be assumed "
        "available at inference. This exclusion narrows and can simplify the task. The retained "
        "test cohort has median age 64; only 105 ECGs are from people aged 18–30.", "",
        "The official validation fold is split by patient: 70% for checkpoint/hyperparameter selection "
        "and 30% for logistic probability calibration and threshold selection. The operating threshold "
        "is the highest achieving at least 95% sensitivity on calibration cases. Test sensitivity is "
        "measured at that fixed threshold and need not equal 95%. The primary run uses "
        f"{TRAINING_LABELS:,} training, 1,306 development, {CALIBRATION_RECORDS} calibration, and "
        f"{TEST_RECORDS:,} test labels: 5,284 total. Eligibility also "
        "consults the full annotation tables. This is retrospective label masking conditional on "
        "eligibility, not a prospective 10% annotation budget. All metrics are ECG-level; confidence "
        "intervals resample test patients 500 times, conditional on the fitted model and threshold. "
        "They exclude training and threshold-estimation uncertainty and domain shift.", "",
        f"## Primary run (label-sampling seed {PRIMARY_SEED})", "",
        "Rows are ordered by observed test AUROC; this is descriptive, not a claim of statistical "
        "superiority. Frozen and fine-tuned systems have different optimization and pretraining budgets.", "",
        "| Model | AUROC (95% patient-cluster CI) | Average precision | Sensitivity | Specificity "
        "| Brier score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]


def primary_rows(primary: list[dict[str, Any]]) -> list[str]:
    """
    Format one primary-table row per model.

    Parameters
    ----------
    primary : list[dict[str, Any]]
        Primary-seed records ordered by test AUROC.

    Returns
    -------
    list[str]
        Markdown table rows.
    """
    lines = []
    for record in primary:
        m = record["test"]
        low, high = record["test_ci95_patient_bootstrap"]["auroc"]
        lines.append(f"| {display_name(record['model'])} | {m['auroc']:.3f} "
                     f"({low:.3f}–{high:.3f}) | {m['average_precision']:.3f} | "
                     f"{100*m['sensitivity']:.1f}% | {100*m['specificity']:.1f}% | {m['brier']:.3f} |")
    return lines


def custom_architecture_lines(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    """
    Summarize the custom multiscale arms when all three have results.

    Parameters
    ----------
    grouped : dict[str, list[dict[str, Any]]]
        Records grouped by model.

    Returns
    -------
    list[str]
        Markdown lines, or none if an arm is missing.
    """
    if not all(name in grouped for name in CUSTOM_ARMS):
        return []
    means = {name: np.mean([r["test"]["auroc"] for r in grouped[name]]) for name in CUSTOM_ARMS}
    return [
        "", "### Custom architecture finding", "",
        f"The new encoder averaged AUROC {means['lead_multiscale_supervised']:.4f} from scratch, "
        f"{means['lead_multiscale_latent']:.4f} with ordinary latent SSL, and "
        f"{means['lead_multiscale_innovation']:.4f} with the added lead-difference target. "
        "The added objective did not outperform scratch training in this pilot. Its comparison "
        "with ordinary SSL and the exact per-seed changes are documented in "
        "[the architecture note](custom-architecture.md). This is a reported negative result, "
        "not evidence that all possible multiscale or lead-aware methods are ineffective.",
    ]


def seed_variation_lines(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    """
    Tabulate AUROC variation across label seeds for models with several seeds.

    Parameters
    ----------
    grouped : dict[str, list[dict[str, Any]]]
        Records grouped by model.

    Returns
    -------
    list[str]
        Markdown section.
    """
    lines = [
        "", "## Variation across label samples", "",
        "The same validation/test patients are reused. For compact SSL models, label seeds share "
        "one SSL initialization trained without labels. These repeats measure label-sampling and "
        "downstream optimization variation, not independent external validations. Standard deviations "
        "below are sample standard deviations across completed seeds.", "",
        "| Model | Completed seeds | Mean AUROC ± SD | Mean sensitivity | Mean specificity |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        auc = np.array([r["test"]["auroc"] for r in group])
        sensitivity = np.mean([r["test"]["sensitivity"] for r in group])
        specificity = np.mean([r["test"]["specificity"] for r in group])
        seeds = ", ".join(str(r["label_seed"]) for r in sorted(group, key=lambda r: r["label_seed"]))
        lines.append(f"| {display_name(name)} | {seeds} "
                     f"| {auc.mean():.3f} ± {auc.std(ddof=1):.3f} | {100*sensitivity:.1f}% "
                     f"| {100*specificity:.1f}% |")
    return lines


def paired_lines(paired: dict[str, Any] | None) -> list[str]:
    """
    Tabulate paired adaptation comparisons.

    Parameters
    ----------
    paired : dict[str, Any] | None
        Output of ``load_paired``.

    Returns
    -------
    list[str]
        Markdown section, or none without comparisons.
    """
    if paired is None:
        return []
    lines = ["", "### Paired adaptation comparisons", "", paired["method"], "",
             "| Adapted model minus reference | AUROC difference | 95% patient-cluster interval |",
             "| --- | ---: | --- |"]
    for comparison in paired["comparisons"]:
        name = NAMES.get(comparison["model"].removesuffix("_seed42"), comparison["model"])
        reference = NAMES.get(comparison["reference"].removesuffix("_seed42"), comparison["reference"])
        low, high = comparison["paired_patient_bootstrap_ci95"]
        lines.append(f"| {name} minus {reference} | {comparison['auroc_difference']:+.4f} "
                     f"| [{low:+.4f}, {high:+.4f}] |")
    return lines


def full_label_lines(full_label: list[dict[str, Any]]) -> list[str]:
    """
    Tabulate runs that used all eligible public training labels.

    Parameters
    ----------
    full_label : list[dict[str, Any]]
        Full-label metric records.

    Returns
    -------
    list[str]
        Markdown section, or none without runs.
    """
    if not full_label:
        return []
    lines = [
        "", "## Using all eligible public training labels", "",
        f"Following the expanded project brief, these runs expose all {FULL_TRAINING_LABELS:,} "
        "eligible PTB-XL training ECG labels from 13,352 patients. Development, calibration, test cases, and "
        "evaluation procedures remain the same. This uses public labels fully; it does not "
        "assume that university labels are available. Results are single-seed exploratory comparisons.", "",
        "| Model | Training labels | AUROC | Sensitivity | Specificity |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in sorted(full_label, key=lambda r: r["test"]["auroc"], reverse=True):
        m = record["test"]
        lines.append(f"| {display_name(record['model'])} | {FULL_TRAINING_LABELS:,} | {m['auroc']:.4f} "
                     f"| {100*m['sensitivity']:.1f}% | {100*m['specificity']:.1f}% |")
    return lines


def screening_rows(primary: list[dict[str, Any]]) -> list[str]:
    """
    Format the hypothetical 1% prevalence arithmetic per model.

    Parameters
    ----------
    primary : list[dict[str, Any]]
        Primary-seed records ordered by test AUROC.

    Returns
    -------
    list[str]
        Markdown table rows.
    """
    lines = []
    for record in primary:
        m = record["hypothetical_1pct_prevalence"]
        lines.append(f"| {display_name(record['model'])} | {m['expected_tp_per_1000']:.1f} "
                     f"| {m['expected_fp_per_1000']:.1f} | {m['expected_fn_per_1000']:.1f} "
                     f"| {100*m['ppv']:.1f}% |")
    return lines


def markdown_report(primary: list[dict[str, Any]], grouped: dict[str, list[dict[str, Any]]],
                    full_label: list[dict[str, Any]], paired: dict[str, Any] | None) -> str:
    """
    Assemble the Experiment 001 Markdown report.

    Parameters
    ----------
    primary : list[dict[str, Any]]
        Primary-seed records ordered by test AUROC.
    grouped : dict[str, list[dict[str, Any]]]
        All records grouped by model.
    full_label : list[dict[str, Any]]
        Full-label metric records.
    paired : dict[str, Any] | None
        Paired adaptation comparisons.

    Returns
    -------
    str
        Report text.
    """
    lines = introduction_lines() + primary_rows(primary)
    lines += custom_architecture_lines(grouped)
    lines += seed_variation_lines(grouped)
    lines += paired_lines(paired)
    lines += full_label_lines(full_label)
    lines += SCREENING_SECTION + screening_rows(primary)
    lines += METHODS_SECTION
    return "\n".join(lines)


def read_test_predictions(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read test targets and probabilities of one run.

    Parameters
    ----------
    directory : Path
        Run directory containing ``test_predictions.csv``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Integer targets and probabilities.
    """
    with (directory / "test_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    target = np.array([int(r["target"]) for r in rows])
    probability = np.array([float(r["probability"]) for r in rows])
    return target, probability


def curve_records(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Choose the leading observed AUROC performers, plus the supervised CNN reference.

    Parameters
    ----------
    primary : list[dict[str, Any]]
        Primary-seed records ordered by test AUROC.

    Returns
    -------
    list[dict[str, Any]]
        Records whose ROC and calibration curves are drawn.
    """
    selected = primary[:TOP_CURVES]
    cnn = next((r for r in primary if r["model"] == REFERENCE_MODEL), None)
    if cnn is not None and cnn not in selected:
        selected = selected + [cnn]
    return selected


def plot_auroc_intervals(ax: Any, primary: list[dict[str, Any]], labels: list[str]) -> None:
    """Plot each model's test AUROC with its patient bootstrap interval."""
    auc = np.array([r["test"]["auroc"] for r in primary])
    lo = np.array([r["test_ci95_patient_bootstrap"]["auroc"][0] for r in primary])
    hi = np.array([r["test_ci95_patient_bootstrap"]["auroc"][1] for r in primary])
    y_positions = np.arange(len(primary))
    ax.errorbar(auc, y_positions, xerr=[auc-lo, hi-auc], fmt="o", capsize=3)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test AUROC with 95% patient bootstrap CI")
    ax.set_title("Primary label sample (seed 42)")


def plot_operating_points(ax: Any, primary: list[dict[str, Any]], labels: list[str],
                          colors: dict[str, Any]) -> None:
    """Scatter test sensitivity against specificity at the calibration threshold."""
    for record, label in zip(primary, labels, strict=True):
        ax.scatter(record["test"]["specificity"], record["test"]["sensitivity"],
                   label=label, color=colors[record["model"]])
    ax.axhline(TARGET_SENSITIVITY, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Test specificity", ylabel="Test sensitivity",
           title="Threshold fixed on calibration patients")


def plot_curves(roc_ax: Any, calibration_ax: Any, records: list[dict[str, Any]],
                colors: dict[str, Any]) -> None:
    """Draw test ROC and calibration curves for the selected runs."""
    for record in records:
        target, probability = read_test_predictions(Path(record["directory"]))
        fpr, tpr, _ = roc_curve(target, probability)
        name = display_name(record["model"])
        roc_ax.plot(fpr, tpr, label=name, color=colors[record["model"]])
        observed, predicted = calibration_curve(target, probability, n_bins=CALIBRATION_BINS,
                                                strategy="quantile")
        calibration_ax.plot(predicted, observed, "o-", markersize=3, label=name,
                            color=colors[record["model"]])
    roc_ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    roc_ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Test ROC curves")
    roc_ax.legend(fontsize=8, loc="lower right")
    calibration_ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    calibration_ax.set(xlabel="Mean calibrated probability", ylabel="Observed positive fraction",
                       title="Calibration on the PTB-XL proxy")


def plot_results(primary: list[dict[str, Any]], output_dir: Path) -> None:
    """
    Save the four-panel results figure as PNG and PDF.

    Parameters
    ----------
    primary : list[dict[str, Any]]
        Primary-seed records ordered by test AUROC, with their run directories.
    output_dir : Path
        Destination directory.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    labels = [display_name(r["model"]) for r in primary]
    colors = {record["model"]: plt.get_cmap("tab20")(i) for i, record in enumerate(primary)}
    plot_auroc_intervals(axes[0, 0], primary, labels)
    plot_operating_points(axes[0, 1], primary, labels, colors)
    plot_curves(axes[1, 0], axes[1, 1], curve_records(primary), colors)
    axes[0, 1].legend(fontsize=7, loc="best")
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    fig.suptitle("ECG experiment: 10% labeled training patients\n"
                 "PTB-XL diagnostic proxy; not student screening validation", fontsize=13)
    fig.savefig(output_dir / "experiment001-results.png", dpi=170)
    fig.savefig(output_dir / "experiment001-results.pdf")


def main() -> None:
    """Write the Experiment 001 report, shareable metric tables and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/experiment001"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument("--full-label-dir", type=Path, default=Path("outputs/experiment002_public_labels"))
    args = parser.parse_args()
    records = load_records(args.input_dir)
    full_label = [json.loads(path.read_text()) for path in args.full_label_dir.glob("*/metrics.json")]
    check_cohort(records + full_label)
    primary = sorted((r for r in records if r["label_seed"] == PRIMARY_SEED),
                     key=lambda r: r["test"]["auroc"], reverse=True)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)
    paired = load_paired(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if paired is not None:
        write_json_atomic(args.output_dir / "paired-adaptation-comparisons.json", paired, allow_nan=True)
    write_text_atomic(args.output_dir / "experiment001-results.md",
                      markdown_report(primary, grouped, full_label, paired))
    # Strip paths/configs from the small shareable metric table; it contains no patient records.
    shareable = [{k: v for k, v in r.items() if k not in {"directory", "config"}} for r in records]
    write_json_atomic(args.output_dir / "experiment001-metrics.json", shareable, allow_nan=True)
    write_json_atomic(args.output_dir / "experiment002-metrics.json", full_label, allow_nan=True)
    plot_results(primary, args.output_dir)
    print(f"Wrote report for {len(records)} completed runs")


if __name__ == "__main__":
    main()
