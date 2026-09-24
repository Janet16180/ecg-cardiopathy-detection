"""Smoke tests for the report and tracking commands on synthetic result files."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.reports import compare_adaptation, report_experiment, report_xecg_adaptation
from scripts.tracking import import_mlflow_history

PREDICTION_FIELDS = ("ecg_id", "patient_id", "target", "probability")
MODELS = ("cnn_supervised", "transformer_supervised", "mae_finetuned", "jepa_finetuned",
          "lead_multiscale_supervised", "lead_multiscale_latent", "lead_multiscale_innovation")


def write_predictions(path: Path, seed: int, records: int = 60, shift: float = 1.0) -> None:
    rng = np.random.default_rng(seed)
    target = (np.arange(records) % 3 != 0).astype(int)
    logits = rng.standard_normal(records) + shift * target
    probability = 1 / (1 + np.exp(-logits))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for index in range(records):
            writer.writerow({"ecg_id": str(index), "patient_id": f"p{index // 2}",
                             "target": str(target[index]), "probability": repr(float(probability[index]))})


def cohort_record(model: str, seed: int, auroc: float) -> dict:
    """Return a metrics record with the fixed Experiment 001 test cohort."""
    tp, fn, tn, fp = 1131, 64, 322, 379
    return {
        "model": model, "label_seed": seed, "threshold": 0.3,
        "calibration": {"records": 564},
        "test": {"auroc": auroc, "average_precision": auroc + 0.02, "sensitivity": tp / (tp + fn),
                 "specificity": tn / (tn + fp), "brier": 0.13, "n": 1896, "tp": tp, "fn": fn, "tn": tn,
                 "fp": fp, "prevalence": 1195 / 1896},
        "test_ci95_patient_bootstrap": {"auroc": [auroc - 0.01, auroc + 0.01]},
        "hypothetical_1pct_prevalence": {"expected_tp_per_1000": 9.4, "expected_fp_per_1000": 530.0,
                                         "expected_fn_per_1000": 0.6, "ppv": 0.017},
    }


def write_experiment_inputs(root: Path, records: list[dict], full_label: list[dict],
                            paired: dict | None) -> dict[str, Path]:
    """Write result directories in the layout read by ``report_experiment``."""
    input_dir, full_dir = root / "experiment001", root / "experiment002"
    for index, record in enumerate(records):
        directory = input_dir / f"{record['model']}_seed{record['label_seed']}"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps(record))
        (directory / "config.json").write_text(json.dumps({"seed": record["label_seed"]}))
        write_predictions(directory / "test_predictions.csv", index, records=200)
    for record in full_label:
        directory = full_dir / record["model"]
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps(record))
    if paired is not None:
        (input_dir / "paired_adaptation_comparisons.json").write_text(json.dumps(paired))
    return {"input_dir": input_dir, "full_label_dir": full_dir, "output_dir": root / "docs"}


def run_report(monkeypatch, paths: dict[str, Path]) -> None:
    argv = ["report_experiment", "--input-dir", str(paths["input_dir"]),
            "--output-dir", str(paths["output_dir"]), "--full-label-dir", str(paths["full_label_dir"])]
    monkeypatch.setattr(sys, "argv", argv)
    report_experiment.main()


@pytest.fixture
def experiment_records():
    records = [cohort_record(model, seed, 0.80 + 0.01 * index + 0.001 * (seed - 42))
               for index, model in enumerate(MODELS) for seed in (42, 43)]
    records.append(cohort_record("ecg-fm_linear", 42, 0.85))
    return records


def test_report_experiment_writes_report(tmp_path, monkeypatch, experiment_records):
    paired = {"method": "Paired method.", "comparisons": [
        {"model": "mae_finetuned_seed42", "reference": "cnn_supervised_seed42",
         "auroc_difference": 0.012, "paired_patient_bootstrap_ci95": [-0.01, 0.03]}]}
    full_label = [cohort_record("cnn_supervised", 42, 0.91)]
    paths = write_experiment_inputs(tmp_path, experiment_records, full_label, paired)

    run_report(monkeypatch, paths)

    report = (paths["output_dir"] / "experiment001-results.md").read_text()
    assert "701 normal proxy labels (63.0% abnormal prevalence)" in report
    assert "| ECG-FM, frozen | 0.850 (0.840–0.860) |" in report
    assert "| Compact MAE + fine-tuning minus CNN, supervised | +0.0120 | [-0.0100, +0.0300] |" in report
    assert "| CNN, supervised | 15,360 | 0.9100 |" in report
    assert "### Custom architecture finding" in report
    assert "| ECG-FM, frozen | 42 |" not in report
    metrics = json.loads((paths["output_dir"] / "experiment001-metrics.json").read_text())
    assert len(metrics) == len(experiment_records)
    assert not {"directory", "config"} & set(metrics[0])
    for name in ("experiment001-results.png", "experiment001-results.pdf", "experiment002-metrics.json",
                 "paired-adaptation-comparisons.json"):
        assert (paths["output_dir"] / name).stat().st_size > 0


def test_report_experiment_rejects_a_different_cohort(tmp_path, monkeypatch, experiment_records):
    experiment_records[0]["test"]["n"] = 1000
    paths = write_experiment_inputs(tmp_path, experiment_records, [], None)

    with pytest.raises(ValueError, match="test cohort"):
        run_report(monkeypatch, paths)


def test_report_experiment_requires_results(tmp_path, monkeypatch):
    paths = write_experiment_inputs(tmp_path, [], [], None)
    paths["input_dir"].mkdir()

    with pytest.raises(ValueError, match="No completed experiment results"):
        run_report(monkeypatch, paths)


def test_compare_adaptation_pairs_patients(tmp_path):
    for seed, name in enumerate(("reference", "adapted")):
        write_predictions(tmp_path / name / "test_predictions.csv", seed, shift=1.0 + seed)

    result = compare_adaptation.compare(tmp_path, "reference", ["adapted"], repeats=50, seed=3)

    assert result["test_records"] == 60
    assert result["test_patients"] == 30
    comparison = result["comparisons"][0]
    low, high = comparison["paired_patient_bootstrap_ci95"]
    assert low <= comparison["auroc_difference"] <= high
    assert comparison["valid_bootstrap_resamples"] == 50
    again = compare_adaptation.compare(tmp_path, "reference", ["adapted"], repeats=50, seed=3)
    assert again == result


def test_compare_adaptation_rejects_mismatched_targets(tmp_path):
    write_predictions(tmp_path / "reference" / "test_predictions.csv", 0)
    path = tmp_path / "adapted" / "test_predictions.csv"
    write_predictions(path, 1)
    rows = list(csv.DictReader(path.open()))
    rows[0]["target"] = str(1 - int(rows[0]["target"]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Patient identity or target differs"):
        compare_adaptation.compare(tmp_path, "reference", ["adapted"])


@pytest.mark.parametrize(("change", "message"), [
    ({"probability": "1.5"}, "Invalid probabilities"),
    ({"ecg_id": ""}, "missing identifiers"),
])
def test_compare_adaptation_validates_predictions(tmp_path, change, message):
    path = tmp_path / "reference" / "test_predictions.csv"
    write_predictions(path, 0)
    rows = list(csv.DictReader(path.open()))
    rows[0].update(change)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=message):
        compare_adaptation.compare(tmp_path, "reference", [])


def write_xecg_transfer(directory: Path, seed: int) -> None:
    write_predictions(directory / "test_predictions.csv", seed, shift=1.0 + 0.1 * seed)
    test = {"auroc": 0.8 + 0.01 * seed, "average_precision": 0.85, "sensitivity": 0.95,
            "specificity": 0.4, "brier": 0.15}
    (directory / "metrics.json").write_text(json.dumps({"threshold": 0.4, "test": test}))
    (directory / "complete.json").write_text("{}")


def write_xecg_inputs(root: Path) -> tuple[Path, Path]:
    output_dir, baseline_dir = root / "experiment008", root / "experiment007"
    seed = 0
    for budget in ("full", "ten_percent"):
        write_xecg_transfer(baseline_dir / f"xecg_{budget}_seed42", seed)
        for arm in ("a", "b", "c", "d"):
            seed += 1
            write_xecg_transfer(output_dir / "transfer" / arm / f"xecg_{budget}_seed42", seed)
    return output_dir, baseline_dir


def test_report_xecg_adaptation_writes_all_comparisons(tmp_path):
    output_dir, baseline_dir = write_xecg_inputs(tmp_path)

    comparisons = report_xecg_adaptation.report(output_dir, 20, baseline_dir)

    pairs = ("d_minus_c", "b_minus_a", "c_minus_b", "a_minus_release")
    expected = [f"{budget}_{pair}" for budget in ("full", "ten_percent") for pair in pairs]
    assert sorted(comparisons) == sorted(expected)
    saved = json.loads((output_dir / "paired_comparisons.json").read_text())
    assert saved == json.loads(json.dumps(comparisons))
    report = (output_dir / "report.md").read_text()
    assert "## Fixed 1,518-label subset" in report
    assert "| Released xECG (007) | 0.8000 |" in report


def test_report_xecg_adaptation_requires_complete_transfers(tmp_path):
    output_dir, baseline_dir = write_xecg_inputs(tmp_path)
    (output_dir / "transfer" / "c" / "xecg_full_seed42" / "complete.json").unlink()

    with pytest.raises(FileNotFoundError, match="incomplete xECG transfer"):
        report_xecg_adaptation.report(output_dir, 20, baseline_dir)


def test_import_mlflow_history_dry_run_lists_runs(tmp_path, monkeypatch, capsys):
    class Run:
        stage = "experiment001"
        source_id = "docs/experiment001-metrics.json#0"
        metrics = {"test_auroc": 0.8, "test_n": 1896}

    monkeypatch.setattr(import_mlflow_history, "discover_historical_runs", lambda root: [Run()])
    monkeypatch.setattr(sys, "argv", ["import_mlflow_history", "--root", str(tmp_path), "--dry-run"])

    import_mlflow_history.main()

    assert capsys.readouterr().out == "experiment001       docs/experiment001-metrics.json#0 2 metrics\n"


def test_import_mlflow_history_creates_sqlite_directory(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_import(root, uri, **kwargs):
        calls.append((root, uri, kwargs))
        return 3, 1

    monkeypatch.setattr(import_mlflow_history, "import_historical_runs", fake_import)
    monkeypatch.setattr(sys, "argv", ["import_mlflow_history", "--root", str(tmp_path)])
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    import_mlflow_history.main()

    uri = f"sqlite:///{tmp_path.resolve() / 'outputs' / 'mlflow' / 'mlflow.db'}"
    assert calls == [(tmp_path, uri, {"experiment_name": import_mlflow_history.EXPERIMENT_NAME,
                                      "log_aggregate_artifacts": False})]
    assert (tmp_path / "outputs" / "mlflow").is_dir()
    assert capsys.readouterr().out == f"Imported 3; already present 1. Tracking URI: {uri}\n"
