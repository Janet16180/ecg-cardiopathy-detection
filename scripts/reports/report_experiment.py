"""Build a reviewable report and scientific figure from completed experiment outputs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/experiment001"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument("--full-label-dir", type=Path, default=Path("outputs/experiment002_public_labels"))
    args = parser.parse_args()
    records = []
    for path in args.input_dir.glob("*/metrics.json"):
        record = json.loads(path.read_text())
        record["directory"] = str(path.parent)
        config_path = path.parent / "config.json"
        record["config"] = json.loads(config_path.read_text()) if config_path.exists() else {}
        records.append(record)
    if not records:
        raise ValueError("No completed experiment results")
    primary = sorted((r for r in records if r["label_seed"] == 42), key=lambda r: r["test"]["auroc"], reverse=True)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)
    full_label = [json.loads(path.read_text()) for path in args.full_label_dir.glob("*/metrics.json")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Experiment 001 results: 10% labeled training patients", "",
        "These are measured results on the PTB-XL v1.0.3 diagnostic-abnormality proxy. "
        "They do not measure disease detection or referral safety in university students. "
        "This is an exploratory study: later experimental arms were designed after inspecting earlier "
        "results, so the overall comparison was not prospectively locked.", "",
        "## Data and evaluation", "",
        "The primary run exposes 1,518 training ECG labels from 1,335 patients (approximately 10% "
        "of eligible training patients). All 17,418 training ECGs are available to our SSL models. "
        "The fixed evaluation set contains 1,896 ECGs from 1,653 patients, with 1,195 abnormal and "
        "701 normal proxy labels (63.0% abnormal prevalence). Ambiguous cases are excluded: 313 "
        "from validation and 302 from test. Test coverage is 86.3% (1,896/2,198); the remaining "
        "13.7% are outside this evaluation. Eligibility uses existing annotations and cannot be "
        "assumed available at inference. This exclusion narrows and can simplify the task. "
        "The retained test cohort has median age 64; only 105 ECGs are from people aged 18–30.", "",
        "The official validation fold is split by patient: 70% for checkpoint/hyperparameter selection "
        "and 30% for logistic probability calibration and threshold selection. The operating threshold "
        "is the highest achieving at least 95% sensitivity on calibration cases. Test sensitivity is "
        "measured at that fixed threshold and need not equal 95%. The primary run uses 1,518 training, "
        "1,306 development, 564 calibration, and 1,896 test labels: 5,284 total. Eligibility also "
        "consults the full annotation tables. This is retrospective label masking conditional on "
        "eligibility, not a prospective 10% annotation budget. All metrics are ECG-level; confidence "
        "intervals resample test patients 500 times, conditional on the fitted model and threshold. "
        "They exclude training and threshold-estimation uncertainty and domain shift.", "",
        "## Primary run (label-sampling seed 42)", "",
        "Rows are ordered by observed test AUROC; this is descriptive, not a claim of statistical "
        "superiority. Frozen and fine-tuned systems have different optimization and pretraining budgets.", "",
        "| Model | AUROC (95% patient-cluster CI) | Average precision | Sensitivity | Specificity | Brier score |",
        "| --- | --- | --- | --- | --- | --- |"]
    for record in primary:
        m = record["test"]
        low, high = record["test_ci95_patient_bootstrap"]["auroc"]
        lines.append(f"| {NAMES.get(record['model'], record['model'])} | {m['auroc']:.3f} "
                     f"({low:.3f}–{high:.3f}) | {m['average_precision']:.3f} | "
                     f"{100*m['sensitivity']:.1f}% | {100*m['specificity']:.1f}% | {m['brier']:.3f} |")
    if all(name in grouped for name in ("lead_multiscale_supervised", "lead_multiscale_latent", "lead_multiscale_innovation")):
        means = {name: np.mean([r["test"]["auroc"] for r in grouped[name]]) for name in
                 ("lead_multiscale_supervised", "lead_multiscale_latent", "lead_multiscale_innovation")}
        lines.extend(["", "### Custom architecture finding", "",
            f"The new encoder averaged AUROC {means['lead_multiscale_supervised']:.4f} from scratch, "
            f"{means['lead_multiscale_latent']:.4f} with ordinary latent SSL, and "
            f"{means['lead_multiscale_innovation']:.4f} with the added lead-difference target. "
            "The added objective did not outperform scratch training in this pilot. Its comparison "
            "with ordinary SSL and the exact per-seed changes are documented in "
            "[the architecture note](custom-architecture.md). This is a reported negative result, "
            "not evidence that all possible multiscale or lead-aware methods are ineffective."])
    lines.extend(["", "## Variation across label samples", "",
        "The same validation/test patients are reused. For compact SSL models, label seeds share "
        "one SSL initialization trained without labels. These repeats measure label-sampling and "
        "downstream optimization variation, not independent external validations. Standard deviations "
        "below are sample standard deviations across completed seeds.", "",
        "| Model | Completed seeds | Mean AUROC ± SD | Mean sensitivity | Mean specificity |",
        "| --- | --- | --- | --- | --- |"])
    for name, group in sorted(grouped.items()):
        auc = np.array([r["test"]["auroc"] for r in group])
        if len(group) < 2:
            continue
        sensitivity = np.mean([r["test"]["sensitivity"] for r in group])
        specificity = np.mean([r["test"]["specificity"] for r in group])
        lines.append(f"| {NAMES.get(name, name)} | {', '.join(str(r['label_seed']) for r in sorted(group, key=lambda r:r['label_seed']))} "
                     f"| {auc.mean():.3f} ± {auc.std(ddof=1):.3f} | {100*sensitivity:.1f}% | {100*specificity:.1f}% |")
    paired_path = args.input_dir / "paired_adaptation_comparisons.json"
    if paired_path.exists():
        paired = json.loads(paired_path.read_text())
        pooling_path = args.input_dir / "paired_pooling_comparison.json"
        if pooling_path.exists():
            pooling = json.loads(pooling_path.read_text())
            paired["comparisons"].extend(pooling["comparisons"])
        lines.extend(["", "### Paired adaptation comparisons", "", paired["method"], "",
                      "| Adapted model minus reference | AUROC difference | 95% patient-cluster interval |",
                      "| --- | ---: | --- |"])
        for comparison in paired["comparisons"]:
            name = NAMES.get(comparison["model"].removesuffix("_seed42"), comparison["model"])
            reference = NAMES.get(comparison["reference"].removesuffix("_seed42"), comparison["reference"])
            low, high = comparison["paired_patient_bootstrap_ci95"]
            lines.append(f"| {name} minus {reference} | {comparison['auroc_difference']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
        (args.output_dir / "paired-adaptation-comparisons.json").write_text(json.dumps(paired, indent=2) + "\n")
    if full_label:
        lines.extend(["", "## Using all eligible public training labels", "",
            "Following the expanded project brief, these runs expose all 15,360 eligible PTB-XL "
            "training ECG labels from 13,352 patients. Development, calibration, test cases, and "
            "evaluation procedures remain the same. This uses public labels fully; it does not "
            "assume that university labels are available. Results are single-seed exploratory comparisons.", "",
            "| Model | Training labels | AUROC | Sensitivity | Specificity |",
            "| --- | ---: | ---: | ---: | ---: |"])
        for record in sorted(full_label, key=lambda r: r["test"]["auroc"], reverse=True):
            m = record["test"]
            lines.append(f"| {NAMES.get(record['model'], record['model'])} | 15,360 | {m['auroc']:.4f} "
                         f"| {100*m['sensitivity']:.1f}% | {100*m['specificity']:.1f}% |")
    lines.extend(["", "## Hypothetical low-prevalence screening", "",
        "The following arithmetic assumes 1% prevalence and unchanged sensitivity/specificity. "
        "Neither assumption has been established in students. The 1% prevalence scenario is unrelated "
        "to the original 1% annotation rate. AUPRC, precision, calibration, and referral volume measured "
        "on PTB-XL cannot be transferred directly to that population.", "",
        "| Model (seed 42) | True positives / 1,000 | False positives / 1,000 | Misses / 1,000 | Positive predictive value |",
        "| --- | --- | --- | --- | --- |"])
    for record in primary:
        m = record["hypothetical_1pct_prevalence"]
        lines.append(f"| {NAMES.get(record['model'],record['model'])} | {m['expected_tp_per_1000']:.1f} "
                     f"| {m['expected_fp_per_1000']:.1f} | {m['expected_fn_per_1000']:.1f} | {100*m['ppv']:.1f}% |")
    lines.extend(["", "## Methods and limits", "",
        "- Compact models use all 12 leads at 100 Hz for 10 seconds, with training-only lead RMS "
        "scaling. The transformer has three 96-dimensional layers and 250 ms patches. MAE learns "
        "masked waveform reconstruction; the JEPA-inspired model predicts EMA-teacher latent targets "
        "and applies a variance penalty. Both receive 50 SSL epochs on training waveforms only, then "
        "the same downstream optimizer as the supervised transformer. They are our small experimental "
        "implementations, not reproductions of published ST-MEM or ECG-JEPA.",
        "- Published encoders retain their own preprocessing and representations. HuBERT and ECG-FM "
        "aggregate two five-second views; source revisions, checkpoint hashes, and preprocessing are "
        "stored with embeddings and model configs. The HuBERT and ECG-FM public SSL checkpoints previously encountered "
        "PTB-XL waveforms, so those results are not an unseen-waveform external test. We use SSL-only "
        "checkpoints rather than supervised Cardio-Learning weights.",
        "- Additional ECG-FM adaptation is a bounded pilot using a CMSC-style temporal contrastive "
        "objective, not a reproduction of its complete pretraining loss. Both arms use one pass "
        "through their training pools, chosen from runtime profiling before downstream evaluation; "
        "this is not a test of extensively optimized continued pretraining. Adaptation checkpoints and their exact waveform sources are "
        "recorded under `outputs/experiment001/`. A pooled arm, where present, adds a verified Georgia "
        "pilot subset without using its diagnoses. Georgia was already represented in the released "
        "encoder's historical pretraining data. Matching the two five-second views can suppress "
        "transient findings present in only one half, so this objective is a hypothesis to evaluate.",
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
        "![Model comparison](experiment001-results.png)", ""])
    (args.output_dir / "experiment001-results.md").write_text("\n".join(lines))
    # Strip paths/configs from the small shareable metric table; it contains no patient records.
    (args.output_dir / "experiment001-metrics.json").write_text(json.dumps([
        {k: v for k, v in r.items() if k not in {"directory", "config"}} for r in records], indent=2) + "\n")
    (args.output_dir / "experiment002-metrics.json").write_text(json.dumps(full_label, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    labels = [NAMES.get(r["model"], r["model"]) for r in primary]
    colors = {record["model"]: plt.get_cmap("tab20")(i) for i, record in enumerate(primary)}
    auc = np.array([r["test"]["auroc"] for r in primary])
    lo = np.array([r["test_ci95_patient_bootstrap"]["auroc"][0] for r in primary])
    hi = np.array([r["test_ci95_patient_bootstrap"]["auroc"][1] for r in primary])
    y_positions = np.arange(len(primary))
    axes[0, 0].errorbar(auc, y_positions, xerr=[auc-lo, hi-auc], fmt="o", capsize=3)
    axes[0, 0].set_yticks(y_positions, labels)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Test AUROC with 95% patient bootstrap CI")
    axes[0, 0].set_title("Primary label sample (seed 42)")
    for record, label in zip(primary, labels):
        axes[0, 1].scatter(record["test"]["specificity"], record["test"]["sensitivity"],
                           label=label, color=colors[record["model"]])
    axes[0, 1].axhline(0.95, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set(xlabel="Test specificity", ylabel="Test sensitivity",
                   title="Threshold fixed on calibration patients")
    # Show the first four observed AUROC performers, plus the supervised CNN reference.
    selected = primary[:4]
    cnn = next((r for r in primary if r["model"] == "cnn_supervised"), None)
    if cnn is not None and cnn not in selected:
        selected = selected + [cnn]
    for record in selected:
        with (Path(record["directory"]) / "test_predictions.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        target = np.array([int(r["target"]) for r in rows])
        probability = np.array([float(r["probability"]) for r in rows])
        fpr, tpr, _ = roc_curve(target, probability)
        name = NAMES.get(record["model"], record["model"])
        axes[1, 0].plot(fpr, tpr, label=name, color=colors[record["model"]])
        observed, predicted = calibration_curve(target, probability, n_bins=8, strategy="quantile")
        axes[1, 1].plot(predicted, observed, "o-", markersize=3, label=name, color=colors[record["model"]])
    axes[1, 0].plot([0, 1], [0, 1], "k--", linewidth=0.7)
    axes[1, 0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Test ROC curves")
    axes[1, 0].legend(fontsize=8, loc="lower right")
    axes[1, 1].plot([0, 1], [0, 1], "k--", linewidth=0.7)
    axes[1, 1].set(xlabel="Mean calibrated probability", ylabel="Observed positive fraction",
                  title="Calibration on the PTB-XL proxy")
    axes[0, 1].legend(fontsize=7, loc="best")
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    fig.suptitle("ECG experiment: 10% labeled training patients\nPTB-XL diagnostic proxy; not student screening validation", fontsize=13)
    fig.savefig(args.output_dir / "experiment001-results.png", dpi=170)
    fig.savefig(args.output_dir / "experiment001-results.pdf")
    print(f"Wrote report for {len(records)} completed runs")


if __name__ == "__main__":
    main()
