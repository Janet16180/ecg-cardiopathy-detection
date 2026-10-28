"""Fresh administrative roles and raw quality from local PTB/SPH releases, without scientific-cache inputs."""

from __future__ import annotations

import json
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from . import ROOT, sph
from .age_cohort_inputs import canonical_integral_id, relative_path, stored
from .cpu_screening041 import role_masks
from .downloads import parse_checksums
from .ecg_quality import EXCLUSION_REASONS, REVIEW_FLAGS, assess
from .eda import ptbxl
from .files import sha256_file, write_json_atomic
from .measurement_extractor042 import read_ecg
from .measurement_screening042 import evaluation_rows, patient_role
from .public_sources import signal_sha256
from .recovery_views import verify_pair
from .waveforms import LEADS, read_record

OUTPUT = Path("outputs/cohort25_recovery_20261028/raw_roles")
PROFILE = OUTPUT.with_name("raw_roles_profile")
PROTOCOL = Path("docs/raw-role-bootstrap-protocol-2026-10-28.md")
PTB_RAW = Path("data/raw/ptb-xl/1.0.3")
SPH_RAW = Path("data/raw/sph/sph")
SOURCE_FILES = (
    "ecg_experiment/raw_role_bootstrap.py", "scripts/data/bootstrap_raw_roles.py",
    "ecg_experiment/eda/ptbxl.py", "ecg_experiment/cpu_screening041.py",
    "ecg_experiment/measurement_screening042.py", "ecg_experiment/measurement_extractor042.py",
    "ecg_experiment/sph.py", "ecg_experiment/ecg_quality.py", "ecg_experiment/public_sources.py",
    "ecg_experiment/recovery_views.py", "ecg_experiment/waveforms.py", "ecg_experiment/files.py",
    "ecg_experiment/age_cohort_inputs.py", "ecg_experiment/paths.py",
)


def configure(input_root: str) -> None:
    """Configure frozen local readers without changing their source or scientific rules."""
    root = Path(input_root)
    ptbxl.PTBXL_DIR = root / PTB_RAW
    sph.EXTRACTED = root / SPH_RAW
    sph.METADATA = sph.EXTRACTED / "17912441/metadata.csv"
    sph.RECORDS = sph.EXTRACTED / "17912444/records"


def metadata(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Read release metadata, reuse frozen labels and assign administrative roles before waveform use."""
    configure(str(input_root))
    ptb = ptbxl.load_metadata().sort_index()
    ptb = ptb.loc[ptb.strat_fold.le(9)].copy()
    classes = ptbxl.diagnostic_classes(ptbxl.load_statements())
    ptb["target"] = ptb.scp_codes.apply(lambda codes: ptbxl.superclass_label(codes, classes))
    ptb["patient_id"] = ptb.patient_id.map(lambda value: int(canonical_integral_id(str(value))))
    ptb["role"] = np.select([ptb.strat_fold.le(7), ptb.strat_fold.eq(8)],
                            ["fit", "calibration"], "development")
    sph.verify_small_files()
    other = sph.table().sort_index().rename(columns={"samples": "source_samples"})
    codes = pd.read_csv(sph.METADATA, dtype={"ECG_ID": str, "AHA_Code": str}).set_index("ECG_ID")
    if len(other) != 25770 or not other.index.is_unique or other.patient_id.isna().any():
        raise ValueError("Invalid SPH release metadata coverage")
    other["aha_codes"] = codes.AHA_Code
    other["role"] = other.patient_id.map(patient_role)
    paths = [PTB_RAW / name for name in ("ptbxl_database.csv", "scp_statements.csv", "SHA256SUMS.txt")]
    paths.append(SPH_RAW / "download_manifest.json")
    manifest = json.loads((input_root / paths[-1]).read_text())
    paths += [SPH_RAW / item["name"] for item in manifest["files"] if not item["name"].endswith(".tar.gz")]
    hashes = {stored(path): sha256_file(input_root / path) for path in paths}
    return ptb, other, hashes


def validate_ptb_header(header: wfdb.Record) -> None:
    """Preserve the original raw decoder's lead order/scaling while accepting official lead-name case."""
    names = tuple(name.upper() for name in header.sig_name)
    if (header.fs != 500 or names != tuple(name.upper() for name in LEADS) or
            header.units != ["mV"] * 12 or header.adc_gain != [1000.0] * 12 or header.baseline != [0] * 12):
        raise ValueError("PTB baseline raw format/scaling contract changed")


def ptb_record(item: tuple[str, int, str, dict[str, str]]) -> dict:
    """Apply the historical baseline's constant-lead exclusion and fingerprint an official PTB pair."""
    input_root, ecg_id, stem, expected = item
    raw = Path(input_root) / PTB_RAW
    verify_pair(raw, stem, expected, "before decoding")
    header = wfdb.rdheader(str(raw / stem))
    validate_ptb_header(header)
    original = read_ecg(raw / stem)
    if original.shape != (5000, 12) or not np.isfinite(original).all():
        raise ValueError("Invalid PTB baseline raw waveform")
    canonical = original.T.astype(np.float32)
    verify_pair(raw, stem, expected, "after decoding")
    return {
        "ecg_id": ecg_id, "source": "ptbxl", "record_id": "ptbxl:" + str(ecg_id),
        "backend": "ptbxl_wfdb", "waveform_path": stored(PTB_RAW / stem),
        "window_start": 0, "samples": 5000, "source_samples": 5000,
        "signal_sha256": signal_sha256(canonical), "flat_leads": int((np.ptp(original, axis=0) == 0).sum()),
        "raw_header_sha256": expected[".hea"], "raw_waveform_sha256": expected[".dat"],
        "worker_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def sph_record(item: tuple[str, str]) -> dict:
    """Fingerprint and assess one SPH first-window record without excluding difficult evaluation examples."""
    input_root, ecg_id = item
    configure(input_root)
    path = SPH_RAW / "17912444/records" / (ecg_id + ".h5")
    before = sha256_file(Path(input_root) / path)
    window = sph.read_window(ecg_id)
    reasons, flags = assess(window)
    if sha256_file(Path(input_root) / path) != before:
        raise ValueError("SPH raw HDF5 changed during decoding")
    return {
        "ecg_id": ecg_id, "source": "sph", "record_id": "sph:" + ecg_id,
        "backend": "sph_h5", "waveform_path": stored(path), "window_start": 0, "samples": 5000,
        "signal_sha256": signal_sha256(window), "raw_waveform_sha256": before, "raw_header_sha256": "",
        "quality_fail": bool(reasons), "exclusion_reasons": ";".join(reasons),
        "review_flags": ";".join(flags),
        **{name: name in reasons for name in EXCLUSION_REASONS},
        **{name: name in flags for name in REVIEW_FLAGS},
        "worker_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def admitted_tables(
    ptb: pd.DataFrame, other: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply historical raw-only admission and exact patient roles without reading old scientific stages."""
    selected = ptb.loc[ptb.target.notna() & ptb.flat_leads.eq(0)].copy()
    selected["target"] = selected.target.astype(int)
    role_masks(selected.reset_index())
    ptb_rows = selected.reset_index()[["ecg_id", "patient_id", "strat_fold", "target", "filename_hr"]]
    frame = other.copy()
    frame["duplicate_status"] = sph.duplicate_status(frame.signal_sha256, frame.aha_codes)
    frame["use_evaluation"] = frame.duplicate_status.isin(["unique", "kept"])
    calibration, evaluation = evaluation_rows(frame)
    admitted = pd.concat([calibration.loc[calibration.target.eq(0)], evaluation])
    columns = ["patient_id", "target", "role", "signal_sha256", "quality_fail"]
    frame["target"] = frame.primary
    return ptb_rows, admitted[columns], frame


def verify_bindings(input_root: Path, inputs: dict[str, str], recipe: dict[str, str]) -> None:
    """Reject changed raw administrative inputs or producer code before stage completion."""
    for name, digest in inputs.items():
        if sha256_file(input_root / name) != digest:
            raise ValueError(f"Raw bootstrap input changed: {name}")
    for name, digest in recipe.items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Raw bootstrap recipe changed: {name}")


def representative_checks(input_root: Path, ptb: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Independently reproduce a small deterministic sample with frozen waveform readers."""
    configure(str(input_root))
    checks = {}
    for label, frame in (("ptb", ptb), ("sph", other)):
        sample = frame.iloc[:min(16, len(frame))]
        for ecg_id, row in sample.iterrows():
            signal = (read_record(input_root / PTB_RAW, row.filename_hr)
                      if label == "ptb" else sph.read_window(ecg_id))
            if signal_sha256(signal) != row.signal_sha256:
                raise ValueError("Independent canonical waveform sample disagrees")
        checks[label + "_canonical_windows"] = len(sample)
    return checks


def require_profile(input_root: Path, inputs: dict[str, str], recipe: dict[str, str]) -> dict[str, str]:
    """Require the completed bounded-memory raw profile before allowing full administrative reconstruction."""
    path = PROFILE / "receipt.json"
    receipt = json.loads((input_root / path).read_text())
    if (not receipt.get("complete") or receipt.get("scope") != "resource_profile" or
            not receipt.get("resource_profile_passed")):
        raise ValueError("A completed raw administrative resource profile is required")
    if receipt["recipe_sha256"] != recipe or receipt["input_sha256"] != inputs:
        raise ValueError("Raw role profile differs from current inputs or code")
    for name, digest in receipt["artifacts"].items():
        if sha256_file(input_root / name) != digest:
            raise ValueError("Raw role profile artifact changed")
    return {stored(path): sha256_file(input_root / path)}


def bootstrap(input_root: Path, output: Path = OUTPUT, workers: int = 4,
              profile_limit: int | None = None) -> dict:
    """
    Build fresh administrative roles and raw-quality evidence without scientific-cache prerequisites.

    Parameters
    ----------
    input_root : Path
        Checkout holding local raw public releases.
    output : Path
        New root-relative directory.
    workers : int
        One through four CPU processes with external numerical-library thread limits.
    profile_limit : int or None
        First lexical records per source for a real profile, or every required record.

    Returns
    -------
    dict
        Honest administrative/raw-quality receipt with input, protocol, source and artifact bindings.
    """
    if not 1 <= workers <= 4 or (profile_limit is not None and profile_limit < 1):
        raise ValueError("Use one to four workers and a positive profile limit")
    output = relative_path(output)
    destination = input_root / output
    if destination.exists():
        raise FileExistsError(destination)
    start = time.monotonic()
    recipe = {stored(Path(name)): sha256_file(ROOT / name) for name in (*SOURCE_FILES, str(PROTOCOL))}
    ptb, other, inputs = metadata(input_root)
    profile_binding = require_profile(input_root, inputs, recipe) if profile_limit is None else {}
    inputs.update(profile_binding)
    selected_ptb = ptb if profile_limit is None else ptb.iloc[:profile_limit]
    selected_sph = other if profile_limit is None else other.iloc[:profile_limit]
    checksums = parse_checksums((input_root / PTB_RAW / "SHA256SUMS.txt").read_text())
    jobs = [(str(input_root), int(ecg_id), row.filename_hr,
             {suffix: checksums[row.filename_hr + suffix] for suffix in (".hea", ".dat")})
            for ecg_id, row in selected_ptb.iterrows()]
    sph_jobs = [(str(input_root), str(ecg_id)) for ecg_id in selected_sph.index]
    computed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(ptb_record, jobs, chunksize=16), 1):
            computed.append(row)
            if index % 2048 == 0:
                print(f"Raw PTB administrative windows {index}/{len(jobs)}", flush=True)
        ptb_computed = pd.DataFrame(computed).set_index("ecg_id")
        computed = []
        for index, row in enumerate(pool.map(sph_record, sph_jobs, chunksize=16), 1):
            computed.append(row)
            if index % 2048 == 0:
                print(f"Raw SPH administrative windows {index}/{len(sph_jobs)}", flush=True)
    sph_computed = pd.DataFrame(computed).set_index("ecg_id")
    peak = int(max(ptb_computed.pop("worker_peak_rss_kib").max(),
                   sph_computed.pop("worker_peak_rss_kib").max()))
    ptb_all = selected_ptb.join(ptb_computed, validate="one_to_one")
    ptb_all["patient_key"] = "ptbxl:" + ptb_all.patient_id.astype(str) + ".0"
    sph_all = selected_sph.join(sph_computed, validate="one_to_one")
    sph_all["patient_key"] = "sph:" + sph_all.patient_id
    independent = representative_checks(input_root, ptb_all, sph_all)
    verify_bindings(input_root, inputs, recipe)
    destination.mkdir(parents=True)
    artifacts = {}
    if profile_limit is None:
        ptb_rows, sph_rows, sph_all = admitted_tables(ptb_all, sph_all)
        ptb_rows.to_csv(destination / "ptb_rows.csv", index=False)
        sph_rows.to_csv(destination / "sph_rows.csv")
        artifacts.update({name: sha256_file(destination / name) for name in ("ptb_rows.csv", "sph_rows.csv")})
        selected_roles = ptb_all.loc[ptb_all.target.notna() & ptb_all.flat_leads.eq(0), "role"]
        counts = {"ptb": selected_roles.value_counts().to_dict(),
                  "sph": sph_rows.role.value_counts().to_dict(), "sph_admitted": len(sph_rows),
                  "ptb_all_development": len(ptb_all), "sph_all": len(sph_all),
                  "sph_quality_fail_all": int(sph_all.quality_fail.sum()),
                  "sph_quality_fail_admitted": int(sph_rows.quality_fail.sum())}
    else:
        counts = {"ptb_profile": len(ptb_all), "sph_profile": len(sph_all)}
    ptb_all = ptb_all.drop(columns=["scp_codes", "recording_date"], errors="ignore")
    for name, frame in (("ptb_administrative.csv", ptb_all), ("sph_all.csv", sph_all)):
        frame.to_csv(destination / name)
        artifacts[name] = sha256_file(destination / name)
    verify_bindings(input_root, inputs, recipe)
    receipt = {
        "complete": True,
        "scope": "administrative_roles_raw_quality" if profile_limit is None else "resource_profile",
        "input_sha256": inputs,
        "source_sha256": {name: recipe[name] for name in SOURCE_FILES},
        "recipe_sha256": recipe, "protocol_path": stored(PROTOCOL),
        "protocol_sha256": recipe[stored(PROTOCOL)],
        "artifacts": {stored(output / name): digest for name, digest in artifacts.items()},
        "counts": counts, "independent_canonical_checks": independent,
        "resource_profile_passed": peak < 4 * 1024**2 and
                                   resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 4 * 1024**2,
        "historical_row_hashes": {name: {"actual": artifacts.get(name), "expected": expected,
                                       "equal": artifacts.get(name) == expected}
                                  for name, expected in {
            "ptb_rows.csv": "fbf7d098c4e651b86a758dabc9a1956107bb63db7f9818387f3b564f2cadd928",
            "sph_rows.csv": "810f4fdd4f24f6a86534b2eee36d507a93aaec66421175b8e77c49fe06f36cba",
        }.items()} if profile_limit is None else {},
        "seeds": {"sph_patient_roles": 20261018}, "workers": workers, "threads_per_worker": 1,
        "wall_seconds": time.monotonic() - start, "worker_peak_rss_kib": peak,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "model_scores_or_predictions_computed": False,
        "scope_note": "Fresh raw administrative roles and descriptive QC, not a historical scientific "
                      "qualification receipt; SPH descriptive QC never excludes evaluation records",
    }
    write_json_atomic(destination / "receipt.json", receipt, sort_keys=True)
    return receipt
