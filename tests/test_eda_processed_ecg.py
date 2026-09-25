"""Smoke tests for the processed-ECG EDA on small synthetic published views."""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ecg_experiment.files import write_csv_atomic
from scripts.reports import eda_processed_ecg as eda

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
CHALLENGE_FIELDS = (
    "ecg_id", "source", "shard", "shard_index", "signal_sha256", "source_samples", "qc_flags",
    "max_abs_mv", "min_lead_std_mv", "duplicate_label_conflict", "record_annotation_available",
    "endpoint_supervised_eligible", "label_scope", "patient_independent_eval_eligible",
    "patient_identity_known", "overlap_by_record", "overlap_by_signal", "original_label_codes",
    "filename_hr", "window_start", "window_samples", "sampling_rate_hz", "units", "lead_order",
    "patient_id",
)
DIAGNOSES = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "normal_ecg")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signal_digest(signal: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(signal, dtype="<f4").tobytes()).hexdigest()


def synthetic_signal(rng: np.random.Generator, kind: str) -> np.ndarray:
    signal = (rng.standard_normal((12, 5000)) * 0.3).astype(np.float32)
    if kind == "rail":
        signal[2, 100:103] = np.float32(32.767)
        signal[4, 50] = np.float32(-32.768)
    elif kind == "peak":
        signal[5, 2000] = np.float32(12.5)
    elif kind == "flat":
        signal[7] *= np.float32(1e-3)
    return signal


def write_raw_header(raw_dir: Path, stem: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    channels = "".join(f"{stem}.mat 16+24 1000/mV 16 0 0 0 0 {lead}\n" for lead in LEADS)
    (raw_dir / f"{stem}.hea").write_text(f"{stem} 12 500 5000\n{channels}")


def write_challenge_view(directory: Path, prepared: Path, policy: str, specs: list[dict],
                         exclusions: list[dict], signals: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shards: dict[str, list[np.ndarray]] = {}
    rows = []
    for spec in specs:
        signal = signals[spec["ecg_id"]]
        shard = spec["shard"]
        shards.setdefault(shard, []).append(signal)
        rows.append({
            "ecg_id": spec["ecg_id"], "source": spec["source"], "shard": shard,
            "shard_index": str(len(shards[shard]) - 1), "signal_sha256": signal_digest(signal),
            "source_samples": spec.get("source_samples", "5000"), "qc_flags": spec.get("qc_flags", ""),
            "max_abs_mv": repr(float(np.abs(signal).max())),
            "min_lead_std_mv": repr(float(signal.std(axis=1).min())),
            "duplicate_label_conflict": spec.get("conflict", "false"),
            "record_annotation_available": "true", "endpoint_supervised_eligible": "false",
            "label_scope": "ssl_only", "patient_independent_eval_eligible": "false",
            "patient_identity_known": "false", "overlap_by_record": spec.get("overlap", "false"),
            "overlap_by_signal": spec.get("overlap", "false"),
            "original_label_codes": spec.get("codes", "426783006"),
            "filename_hr": f"training/{spec['source']}/{spec['ecg_id']}", "window_start": "0",
            "window_samples": "5000", "sampling_rate_hz": "500", "units": "mV",
            "lead_order": ",".join(LEADS), "patient_id": f"record:{spec['ecg_id']}",
        })
    for name, stack in shards.items():
        np.save(directory / name, np.stack(stack))
    write_csv_atomic(directory / "manifest.csv", rows, CHALLENGE_FIELDS)
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "metadata.json").write_text(json.dumps({"prepared": True}))
    write_csv_atomic(prepared / "exclusions.csv", exclusions, ("ecg_id", "source", "reason"))
    provenance: dict[str, dict[str, int]] = {}
    for item in rows + exclusions:
        entry = provenance.setdefault(item["source"], {"candidate_records": 0})
        entry["candidate_records"] += 1
    metadata = {
        "complete": True, "schema_version": 1, "policy": policy,
        "manifest_sha256": sha256(directory / "manifest.csv"),
        "materialized_records": len(rows), "accepted_records": len(rows),
        "candidate_records": len(rows) + len(exclusions), "excluded_records": len(exclusions),
        "shards": [{"file": name, "records": len(stack)} for name, stack in shards.items()],
        "counts": {"qc_flagged": sum(bool(r["qc_flags"]) for r in rows),
                   "label_conflict_retained": sum(r["duplicate_label_conflict"] == "true" for r in rows)},
        "input_prepared_dir": str(prepared),
        "input_metadata_sha256": sha256(prepared / "metadata.json"),
        "input_exclusions_sha256": sha256(prepared / "exclusions.csv"),
        "official_provenance": provenance,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))


def write_code_view(directory: Path, rng: np.random.Generator) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    traces = (rng.standard_normal((6, 4096, 12)) * 0.2).astype(np.float32)
    edges = [(0, 0), (0, 0), (581, 581), (3, 5), (0, 0), (581, 581)]
    manifest = [{"exam_id": str(100 + i), "patient_id": f"p{min(i, 4)}",
                 "edge_zero_left": str(left), "edge_zero_right": str(right),
                 "signal_sha256": signal_digest(traces[i]), "qc_flags": "flat_lead" if i == 3 else ""}
                for i, (left, right) in enumerate(edges)]
    exclusions = [{"exam_id": "200", "reason": "short"}, {"exam_id": "201", "reason": "nan"}]
    ages = ["25", "64", "", "130", "19", "80"]
    demographics = [{"exam_id": r["exam_id"], "age": age, "is_male": str(i % 2 == 0)}
                    for i, (r, age) in enumerate(zip(manifest, ages, strict=True))]
    labels = [{"exam_id": r["exam_id"], **{name: str((i + j) % 3 == 0) for j, name in enumerate(DIAGNOSES)},
               "diagnosis_label_provenance": "released", "normal_ecg_provenance": "released"}
              for i, r in enumerate(manifest)]
    write_csv_atomic(directory / "manifest.csv", manifest, tuple(manifest[0]))
    write_csv_atomic(directory / "exclusions.csv", exclusions, ("exam_id", "reason"))
    write_csv_atomic(directory / "demographics.csv", demographics, ("exam_id", "age", "is_male"))
    write_csv_atomic(directory / "source_labels.csv", labels, tuple(labels[0]))
    with h5py.File(directory / "exams_part0_native.hdf5", "w") as handle:
        handle["tracings"] = traces
        handle["exam_id"] = np.array([int(r["exam_id"]) for r in manifest])
    names = ("manifest.csv", "exclusions.csv", "demographics.csv", "source_labels.csv")
    metadata = {
        "limit_per_archive": None, "canonical_500hz_10s_eligible": False,
        "output_sha256": {name: sha256(directory / name) for name in names},
        "counts": {"accepted": 6, "audited": 8, "review_flagged": 1},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))


def write_eda_inputs(root: Path) -> dict[str, Path]:
    """
    Write synthetic strict, centered and CODE views under a fake project root.

    Returns the directories to pass to ``run``.
    """
    rng = np.random.default_rng(7)
    kinds = {"S0": "rail", "S1": "peak", "S2": "flat", "S3": "flat", "C2": "plain", "C3": "flat"}
    for name in ("S4", "S5", "S6", "S7"):
        kinds[name] = "plain"
    signals = {name: synthetic_signal(rng, kind) for name, kind in sorted(kinds.items())}
    strict_specs = [
        {"ecg_id": "S0", "source": "georgia", "shard": "shard_000.npy",
         "qc_flags": "amplitude_over_10mV_review"},
        {"ecg_id": "S1", "source": "cpsc_2018", "shard": "shard_000.npy",
         "qc_flags": "amplitude_over_10mV_review"},
        {"ecg_id": "S2", "source": "georgia", "shard": "shard_000.npy", "qc_flags": "near_flat_lead_review"},
        {"ecg_id": "S3", "source": "cpsc_2018_extra", "shard": "shard_000.npy",
         "qc_flags": "near_flat_lead_review"},
        {"ecg_id": "S4", "source": "cpsc_2018", "shard": "shard_001.npy", "overlap": "true"},
        {"ecg_id": "S5", "source": "cpsc_2018_extra", "shard": "shard_001.npy", "overlap": "true"},
        {"ecg_id": "S6", "source": "georgia", "shard": "shard_001.npy", "source_samples": "7500",
         "codes": "164889003, 59118001"},
        {"ecg_id": "S7", "source": "cpsc_2018", "shard": "shard_001.npy", "conflict": "true"},
    ]
    crop_specs = [
        # The centered view reuses two strict records unchanged, as the real views do.
        {"ecg_id": "S4", "source": "cpsc_2018", "shard": "shard_000.npy", "overlap": "true"},
        {"ecg_id": "S5", "source": "cpsc_2018_extra", "shard": "shard_000.npy", "overlap": "true"},
        {"ecg_id": "C2", "source": "cpsc_2018", "shard": "shard_000.npy", "source_samples": "6000"},
        {"ecg_id": "C3", "source": "cpsc_2018", "shard": "shard_000.npy",
         "qc_flags": "near_flat_lead_review"},
    ]
    views = root / "data/processed/challenge_ecg_views"
    paths = {"strict_dir": views / "strict_10s", "crop_dir": views / "cpsc_ssl_center_crop",
             "code_dir": root / "data/processed/code15_quality/part0_native",
             "output_dir": root / "outputs/eda"}
    write_challenge_view(
        paths["strict_dir"], root / "prepared/strict", "strict_10s", strict_specs,
        [{"ecg_id": "X0", "source": "georgia", "reason": "duration_contract"},
         {"ecg_id": "X1", "source": "georgia", "reason": "duration_contract"},
         {"ecg_id": "X2", "source": "cpsc_2018", "reason": "unreadable"}], signals)
    write_challenge_view(
        paths["crop_dir"], root / "prepared/crop", "ssl_center_crop", crop_specs,
        [{"ecg_id": "X3", "source": "cpsc_2018_extra", "reason": "duration_contract"}], signals)
    write_raw_header(root / "data/raw/challenge-2020/1.0.2/training/georgia", "S0")
    write_code_view(paths["code_dir"], rng)
    return paths


@pytest.fixture
def eda_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(eda, "ROOT", tmp_path)
    return write_eda_inputs(tmp_path)


def test_run_writes_consistent_outputs(eda_inputs):
    result = eda.run(**eda_inputs, sample_size=3, seed=42)
    output = eda_inputs["output_dir"]

    assert result["strict"]["accepted_records"] == 8
    assert result["center_crop"]["accepted_records"] == 4
    assert result["strict"]["waveform_sample"] == {"seed": 42, "selected": 3, "hashes_verified": 3}
    assert result["strict"]["exclusions_by_source"] == {
        "cpsc_2018": {"unreadable": 1}, "georgia": {"duration_contract": 2}}
    assert result["challenge_cross_view"]["same_ecg_id"] == 2
    assert result["challenge_cross_view"]["same_canonical_signal"] == 2
    assert result["code_part0"]["age_known"] == 4
    assert result["code_part0"]["edge_zero_other"] == 1
    assert result["curated_ssl_candidate"]["selected_total"] == 9
    assert result["curated_ssl_candidate"]["omitted_by_reason"] == {
        "strict_first_exact_signal_overlap": 2, "verified_signed_16_rail": 1}
    assert [r["ecg_id"] for r in result["strict"]["rail_value_examples"]] == ["S0"]

    overlay = list(csv.DictReader((output / "training_exclusion_overlay.csv").open()))
    assert [(r["view"], r["ecg_id"]) for r in overlay] == [("strict_10s", "S0")]
    receipt = json.loads((output / "challenge_ssl_curated_receipt.json").read_text())
    assert receipt["curated_manifest_sha256"] == sha256(output / "challenge_ssl_curated_manifest.csv")
    assert json.loads((output / "summary.json").read_text()) == json.loads(json.dumps(result))
    for name in ("overview.png", "review_examples.png", "challenge_review_flags.csv",
                 "challenge_ssl_curated_omissions.csv"):
        assert (output / name).stat().st_size > 0


def test_run_rejects_changed_waveform(eda_inputs):
    shard = eda_inputs["strict_dir"] / "shard_000.npy"
    array = np.load(shard)
    array[:, 0, 0] += 1
    np.save(shard, array)

    with pytest.raises(ValueError, match="differs from manifest"):
        eda.run(**eda_inputs, sample_size=8, seed=42)


def test_run_rejects_output_inside_raw_data(eda_inputs):
    eda_inputs["output_dir"] = eda.ROOT / "data/raw/eda"

    with pytest.raises(ValueError, match="inside data/raw"):
        eda.run(**eda_inputs)


def test_quantiles_of_empty_values_are_missing():
    assert set(eda.quantiles([]).values()) == {None}
    assert eda.quantiles([1.0, 3.0])["median"] == 2.0


def test_first_per_source_keeps_order_and_limit():
    rows = [{"source": "a", "id": "1"}, {"source": "a", "id": "2"},
            {"source": "b", "id": "3"}, {"source": "c", "id": "4"}]

    assert [r["id"] for r in eda.first_per_source(rows, 2)] == ["1", "3"]
