"""Prepare a role-gated local ECG review packet without inventing human annotations."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from PIL import Image
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.clustering.native_cache import verified_cohort, verify_hashes
from ecg_experiment.clustering.native_cache_v2 import read_waveform
from ecg_experiment.clustering02 import anchor_patient
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic
from ecg_experiment.localization.review_forms import pending_form, validate_pending_form
from ecg_experiment.paths import to_stored

REFERENCE = ROOT / "outputs/cluster03_native_reference/result.json"
REFERENCE_SHA = "64949f3472efdc309792d4ea8ca0d6bdd0c017383796bc60246b9536882e67f9"
CACHE = ROOT / "outputs/cluster03_native_cache_v2/result.json"
CACHE_SHA = "6758aee6b8d90b12e0772e28a1f877bd275e17779c062c1386cbbd0f93033fb3"
DISPLAY_SEED = 2026101907
PAGE_MM = (570, 590)
DPI = 120
SOURCES = (
    "docs/research/localization/ntt07-blinded-review/protocol.md",
    "docs/research/localization/ntt07-blinded-review/execution-layout.md",
    "ecg_experiment/localization/blinded_packet.py",
    "ecg_experiment/localization/evidence_validation.py",
    "ecg_experiment/localization/review_forms.py",
    "scripts/experiments/localization/run_ntt07.py",
    "scripts/experiments/localization/validate_evidence.py",
    "tests/localization/test_evidence_validation.py",
    "tests/localization/test_blinded_packet.py",
    "schemas/localization/pending-review-form-v1.schema.json",
    "schemas/ecg-evidence-annotation-v1.schema.json",
    "schemas/ecg-evidence-synthetic-example.json",
)
INSTRUCTIONS = """# Local ECG review packet: round one

This packet contains thirty anonymous cases and identical pending review forms. The review rubric
awaits clinician approval. No clinical annotations have been entered. Do not treat pending entries
as clinical uncertainty, absence or evidence that a review occurred.

Each PDF shows twelve unfiltered full ten-second leads in two columns. The fixed grid is 25 mm/s
and 10 mm/mV, with 1 mm fine and 5 mm major squares. Print the 570 mm by 590 mm PDF page at actual
size to preserve calibration. The PNG is a viewing copy; apparent screen scale depends on software.
The calibration pulse is 1 mV high and 200 ms wide. Signal panels span -4 to 4 mV.

Every case has the same draft prompts. Sinus rhythm is a review prompt and does not establish
clinical normality or correspond to an imported source normal label. Other finding allows later
free text. These prompts are not an exhaustive taxonomy or new clinical diagnostic criteria.

Keep uncertainty and disagreements when actual review begins. Evidence annotations use original
signal seconds or half-open sample intervals, named leads, and acceptable alternative sets; regions
inside one set are considered jointly. Unmarked samples remain unknown. A finding judged absent
is a record-level judgment and does not label every sample as negative. Do not convert pending
forms into finding annotations. Reviewer identity and review time must be recorded when review
actually occurs. The evidence contract remains a draft requiring specialist rubric review.
"""


def frozen_sources() -> dict[str, str]:
    """Bind exact committed study sources before patient selection or rendering."""
    result = {}
    for name in SOURCES:
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"Uncommitted NTT-07 source: {name}")
        result[name] = digest
    return result


def bound_receipt(path: Path, digest: str, hashes: dict) -> dict:
    """Verify predecessor source, input and artifact bytes without interpreting labels."""
    if sha256_file(path) != digest:
        raise ValueError("Frozen predecessor result differs")
    hashes[to_stored(path)] = digest
    value = json.loads(path.read_text())
    for field in ("source_sha256", "input_sha256", "artifact_sha256"):
        verify_hashes(value[field], hashes)
    if value["status"] != "completed":
        raise ValueError("Predecessor did not complete")
    return value


def reproduce_split(ids: pd.DataFrame, saved: dict, reference: dict) -> np.ndarray:
    """Exactly reproduce all nine patient splits and their bound patient hashes."""
    anchors = np.flatnonzero(ids.patient_id.map(anchor_patient).to_numpy())
    fitting = np.flatnonzero(~ids.patient_id.map(anchor_patient).to_numpy())
    samples = [fitting] + [
        np.sort(np.random.default_rng(20261019 + i).choice(fitting, int(0.8 * len(fitting)), replace=False))
        for i in range(8)
    ]
    expected = {
        "ecg_ids": ids.ecg_id.astype(int).tolist(),
        "anchors": anchors.tolist(),
        "samples": [sample.tolist() for sample in samples],
    }
    if saved != expected or (len(ids), len(anchors), len(fitting)) != (1993, 411, 1582):
        raise ValueError("Frozen CLUSTER-03 identity or split differs")
    hashes = [
        hashlib.sha256(ids.patient_id.to_numpy()[s].astype("<i8").tobytes()).hexdigest() for s in samples
    ]
    if hashes != reference["patient_sample_hashes"]:
        raise ValueError("Frozen CLUSTER-03 patient sample hashes differ")
    return anchors


def audit_coordinates(cache: dict, ids: pd.DataFrame) -> dict:
    """Reproduce exact cache identities and finite original-time coordinate invariants."""
    if not np.array_equal(cache["ecg_ids"], ids.ecg_id) or not np.array_equal(
        cache["patient_ids"], ids.patient_id
    ):
        raise ValueError("Cache identities differ from verified cohort")
    offsets, bounds = cache["region_offsets"], cache["qrs_bounds"]
    if offsets.shape != (1994,) or offsets.dtype.kind not in "iu" or offsets[0] != 0:
        raise ValueError("Invalid frozen region offsets")
    if np.any(np.diff(offsets) < 3) or offsets[-1] != len(bounds) or bounds.shape != (22156, 2):
        raise ValueError("Frozen region denominator differs")
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] < 0) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Invalid frozen region start or stop")
    if np.any(bounds[:, 1] > 10):
        raise ValueError("Frozen region coordinates exceed ten seconds")
    return {
        "records": len(ids),
        "regions": len(bounds),
        "coordinate_bounds_passed": True,
        "qrs_bounds_sha256": hashlib.sha256(bounds.astype("<f8").tobytes()).hexdigest(),
    }


def verified_inputs() -> tuple[pd.DataFrame, np.ndarray, dict, dict, dict]:
    """Gate all payloads by frozen receipts and PR67 roles before coordinate access."""
    hashes: dict[str, str] = {}
    reference = bound_receipt(REFERENCE, REFERENCE_SHA, hashes)
    saved_cache = bound_receipt(CACHE, CACHE_SHA, hashes)
    ids, _, _, cohort_hashes = verified_cohort()
    hashes.update(cohort_hashes)
    verify_hashes(saved_cache["waveform_sha256"], hashes)
    accepted = json.loads((CACHE.parent / "accepted_local.json").read_text())
    if accepted != ids.to_dict("records"):
        raise ValueError("Frozen accepted identities differ")
    saved = json.loads((REFERENCE.parent / "patient_splits_local.json").read_text())
    anchors = reproduce_split(ids, saved, reference)
    with np.load(CACHE.parent / "native_features_local.npz", allow_pickle=False) as cache:
        integrity = audit_coordinates(cache, ids)
    return ids, anchors, hashes, saved_cache, integrity


def select_cases(ids: pd.DataFrame, anchors: np.ndarray) -> list[dict]:
    """Select prospectively by patient identity only, then independently randomize display."""
    rows = ids.iloc[anchors].to_dict("records")
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"ntt07_selection:20261019:{int(row['patient_id'])}".encode()).hexdigest(),
            int(row["patient_id"]),
        ),
    )[:30]
    return [ordered[int(i)] for i in np.random.default_rng(DISPLAY_SEED).permutation(30)]


def panel_origin(index: int) -> tuple[float, float]:
    """Return a lead panel's lower-left position in physical page millimeters."""
    return 20.0 + (index // 6) * 285.0, 495.0 - (index % 6) * 92.0


def signal_to_page(index: int, seconds: float, millivolts: float) -> tuple[float, float]:
    """Map original signal coordinates to physical page millimeters."""
    left, bottom = panel_origin(index)
    return left + 25 * seconds, bottom + 40 + 10 * millivolts


def page_to_signal(index: int, x_mm: float, y_mm: float) -> tuple[float, float]:
    """Invert the physical page transform without pixel rounding."""
    left, bottom = panel_origin(index)
    return (x_mm - left) / 25, (y_mm - bottom - 40) / 10


def signal_to_png(index: int, seconds: float, millivolts: float, image_height: int) -> tuple[float, float]:
    """Map original coordinates to unrounded PNG pixels measured from the upper left."""
    x_mm, y_mm = signal_to_page(index, seconds, millivolts)
    return x_mm * DPI / 25.4, image_height - y_mm * DPI / 25.4


def png_to_signal(index: int, x_px: float, y_px: float, image_height: int) -> tuple[float, float]:
    """Invert PNG pixel coordinates before any integer cursor rounding."""
    return page_to_signal(index, x_px * 25.4 / DPI, (image_height - y_px) * 25.4 / DPI)


def render_case(signal: np.ndarray, case_ref: str, directory: Path) -> dict:
    """Render unfiltered twelve-lead signals at fixed physical scale without overlays."""
    if signal.shape != (5000, 12) or not np.isfinite(signal).all() or np.any(np.abs(signal) > 4):
        raise ValueError("Signal does not fit frozen finite ±4 mV rendering range")
    width, height = PAGE_MM
    figure = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=DPI)
    figure.text(0.035, 0.987, f"{case_ref} | 12 leads | 500 Hz | 10 seconds | 25 mm/s | 10 mm/mV", va="top")
    times = np.arange(5000) / 500
    for index, lead in enumerate(LEADS):
        left, bottom = panel_origin(index)
        axis = figure.add_axes([left / width, bottom / height, 250 / width, 80 / height])
        axis.set(xlim=(0, 10), ylim=(-4, 4))
        axis.xaxis.set_major_locator(MultipleLocator(0.2))
        axis.xaxis.set_minor_locator(MultipleLocator(0.04))
        axis.yaxis.set_major_locator(MultipleLocator(0.5))
        axis.yaxis.set_minor_locator(MultipleLocator(0.1))
        axis.grid(which="minor", color="#f8dfdf", linewidth=0.25)
        axis.grid(which="major", color="#e4aaaa", linewidth=0.5)
        axis.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.plot(times, signal[:, index], color="#171717", linewidth=0.45, clip_on=False)
        axis.text(-0.035, 0.5, lead, transform=axis.transAxes, ha="right", va="center", fontsize=12)
        for second in range(11):
            axis.text(second, -4.1, str(second), ha="center", va="top", fontsize=7)
    pulse = figure.add_axes([5 / width, 15 / height, 10 / width, 20 / height])
    pulse.plot([0, 2, 2, 7, 7, 10], [0, 0, 10, 10, 0, 0], color="black", linewidth=0.7)
    pulse.set(xlim=(0, 10), ylim=(-5, 15))
    pulse.axis("off")
    figure.text(
        0.035,
        0.025,
        "Calibration: 1 mV / 200 ms. Print actual size. Signal seconds below each lead.",
        fontsize=9,
    )
    pdf, png = directory / f"{case_ref}.pdf", directory / f"{case_ref}.png"
    figure.savefig(
        pdf,
        metadata={
            "Title": case_ref,
            "Author": "",
            "Subject": "",
            "Keywords": "",
            "Creator": "ECG review packet",
            "Producer": "ECG review packet",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(png, dpi=DPI, metadata={"Software": "ECG review packet"})
    plt.close(figure)
    with Image.open(png) as image:
        dimensions = list(image.size)
    return {
        "page_mm": list(PAGE_MM),
        "png_dimensions": dimensions,
        "png_dpi": DPI,
        "png_coordinate_convention": "continuous pixels; upper-left origin; before cursor rounding",
        "png_pixels_per_mm": DPI / 25.4,
        "png_y_from_page_mm": "image_height_pixels - page_y_mm * png_pixels_per_mm",
        "signal_panel_mm": [250, 80],
        "mV_range": [-4, 4],
        "time_scale_mm_per_second": 25,
        "amplitude_scale_mm_per_mV": 10,
        "panels": [
            {"lead": lead, "origin_mm": list(panel_origin(i)), "zero_mV_y_mm": panel_origin(i)[1] + 40}
            for i, lead in enumerate(LEADS)
        ],
    }


def audit_reviewer_case(directory: Path, case_ref: str) -> dict:
    """Inspect actual PDF text/metadata and PNG metadata against a strict template."""
    form = json.loads((directory / f"{case_ref}.json").read_text())
    validate_pending_form(form)
    pdf = directory / f"{case_ref}.pdf"
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    allowed = {
        case_ref,
        "12",
        "leads",
        "500",
        "Hz",
        "10",
        "seconds",
        "25",
        "mm/s",
        "mm/mV",
        "|",
        "Calibration:",
        "1",
        "mV",
        "/",
        "200",
        "ms.",
        "Print",
        "actual",
        "size.",
        "Signal",
        "below",
        "each",
        "lead.",
        *LEADS,
        *(str(i) for i in range(11)),
    }
    if set(text.split()) - allowed:
        raise ValueError("Unexpected reviewer-visible PDF text")
    metadata = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    fields = dict(line.split(":", 1) for line in metadata.splitlines() if ":" in line)
    fields = {key: value.strip() for key, value in fields.items()}
    if fields.get("Title") != case_ref or fields.get("Pages") != "1":
        raise ValueError("PDF title or page count differs")
    for key in ("Author", "Subject", "Keywords", "CreationDate", "ModDate"):
        if fields.get(key, ""):
            raise ValueError("Unexpected identifying PDF metadata")
    if fields.get("Creator") != "ECG review packet" or fields.get("Producer") != "ECG review packet":
        raise ValueError("Unexpected PDF generator metadata")
    page_points = np.array([float(value) for value in fields["Page size"].split()[0:3:2]])
    if not np.allclose(page_points, np.array(PAGE_MM) * 72 / 25.4, atol=0.01, rtol=0):
        raise ValueError("PDF physical page dimensions differ")
    with Image.open(directory / f"{case_ref}.png") as image:
        if set(image.info) != {"Software", "dpi"} or image.info["Software"] != "ECG review packet":
            raise ValueError("Unexpected reviewer-visible PNG metadata")
    return {"pending_form": True, "pdf_text_template_passed": True, "image_metadata_passed": True}


def budget(deadline: float) -> None:
    """Stop at the prospective 900-second stage deadline."""
    if time.perf_counter() > deadline:
        raise TimeoutError("NTT-07 packet budget exceeded")


def run_packet(output: Path) -> dict:
    """Create thirty local cases and an auditable readiness receipt, never clinical labels."""
    if output != ROOT / "outputs/ntt07_blinded_review":
        raise ValueError("Patient artifacts must remain under the fixed private NTT-07 output path")
    sources = frozen_sources()
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    started, records = time.perf_counter(), []
    write_json_atomic(output / "status.json", {"status": "running"})
    try:
        with threadpool_limits(limits=2):
            result = create_packet(output, sources, started, records)
        write_json_atomic(output / "result.json", result)
        write_json_atomic(
            output / "status.json",
            {"status": "completed", "result_sha256": sha256_file(output / "result.json")},
        )
    except Exception as error:
        write_json_atomic(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "source_sha256": sources,
                "completed_cases": len(records),
                "seconds": time.perf_counter() - started,
            },
        )
        write_json_atomic(output / "status.json", {"status": "failed"})
        raise
    return result


def create_packet(output: Path, sources: dict, started: float, records: list) -> dict:
    """Execute role-gated integrity replay, selection, rendering and privacy checks."""
    deadline = started + 900
    ids, anchors, inputs, cache, integrity = verified_inputs()
    budget(deadline)
    selected = select_cases(ids, anchors)
    reviewer, private = output / "reviewer", output / "private_key"
    reviewer.mkdir(mode=0o700)
    private.mkdir(mode=0o700)
    (reviewer / "instructions.md").write_text(INSTRUCTIONS)
    bound_waveforms = dict(cache["waveform_sha256"])
    for index, row in enumerate(selected):
        budget(deadline)
        case_ref = f"case-{index + 1:03d}"
        signal = np.asarray(read_waveform(row, bound_waveforms), dtype="<f8", order="C")
        signal_digest = hashlib.sha256(signal.tobytes(order="C")).hexdigest()
        signal_path = private / f"{case_ref}-signal.npy"
        np.save(signal_path, signal, allow_pickle=False)
        saved_signal = np.load(signal_path, allow_pickle=False)
        if hashlib.sha256(saved_signal.astype("<f8").tobytes(order="C")).hexdigest() != signal_digest:
            raise ValueError("Saved signal byte binding differs")
        rendering = render_case(signal, case_ref, reviewer)
        write_json_atomic(reviewer / f"{case_ref}.json", pending_form(case_ref))
        audit = audit_reviewer_case(reviewer, case_ref)
        records.append(
            {
                "case_ref": case_ref,
                "identity": row,
                "signal_sha256": signal_digest,
                "signal_byte_convention": "canonical samples-by-leads little-endian float64 C-order",
                "sampling_rate_hz": 500,
                "n_samples": 5000,
                "lead_order": list(LEADS),
                "rendering": rendering,
                "audit": audit,
            }
        )
        print(f"NTT-07 rendered {index + 1}/30", flush=True)
    budget(deadline)
    write_json_atomic(private / "mapping.json", {"records": records, "waveform_sha256": bound_waveforms})
    artifact_hashes = {
        to_stored(path): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "status.json"
    }
    expected_names = {"instructions.md"} | {
        f"case-{i:03d}.{ext}" for i in range(1, 31) for ext in ("pdf", "png", "json")
    }
    if {path.name for path in reviewer.iterdir()} != expected_names:
        raise ValueError("Unexpected file in reviewer directory")
    selected_patients = [int(row["patient_id"]) for row in selected]
    readiness = {
        "exactly_30_rendered_cases": len(records) == 30,
        "exactly_30_pending_forms": all(row["audit"]["pending_form"] for row in records)
        and len(records) == 30,
        "zero_fabricated_annotations": all(
            json.loads((reviewer / f"{row['case_ref']}.json").read_text())["reviewer_ref"] is None
            for row in records
        ),
        "patient_roles_verified": True,
        "support_disjoint": True,
        "unique_patients": len(set(selected_patients)) == 30,
        "identity_only_selection": True,
        "physical_coordinate_audit": True,
        "reviewer_privacy_audit": all(all(row["audit"].values()) for row in records),
        "source_and_signal_bytes_bound": True,
        "wall_budget_passed": time.perf_counter() - started <= 900,
    }
    if not all(readiness.values()):
        raise ValueError("Technical readiness gate failed")
    return {
        "study": "NTT-07",
        "status": "completed",
        "source_sha256": sources,
        "input_sha256": inputs,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in ("numpy", "matplotlib", "jsonschema", "wfdb", "Pillow")
            },
            "threads": 2,
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "predecessor_integrity": {
            **integrity,
            "fitting_patients": 1582,
            "validation_anchors": 411,
            "nine_patient_split_hashes_reproduced": True,
        },
        "selection_identity_sha256": sha256_json(selected_patients),
        "display_seed": DISPLAY_SEED,
        "rendered_cases": 30,
        "pending_forms": 30,
        "clinical_annotations": 0,
        "clinical_accuracy_evaluated": False,
        "rubric_approved": False,
        "technical_readiness_passed": True,
        "readiness_checks": readiness,
        "reviewer_directory": to_stored(reviewer),
        "private_key_directory": to_stored(private),
        "artifact_sha256": artifact_hashes,
        "seconds": time.perf_counter() - started,
        "waveform_reads": 30,
        "encoder_forwards": 0,
        "gpu_used": False,
        "training": False,
        "limitations": [
            "Human rubric approval and independent specialist review remain pending.",
            "Technical validation cannot certify clinical truth or evidence sufficiency.",
            "These are explored development anchors, not an independent clinical test.",
            "The packet was neither delivered nor uploaded.",
        ],
    }


def aggregate_receipt(result: dict, result_path: Path) -> dict:
    """Extract safe aggregate readiness evidence without identities or signal hashes."""
    keys = (
        "study",
        "status",
        "source_commit",
        "runtime",
        "rendered_cases",
        "pending_forms",
        "clinical_annotations",
        "clinical_accuracy_evaluated",
        "rubric_approved",
        "technical_readiness_passed",
        "readiness_checks",
        "seconds",
        "waveform_reads",
        "encoder_forwards",
        "gpu_used",
        "training",
        "limitations",
    )
    return {
        **{key: result[key] for key in keys},
        "local_result_sha256": sha256_file(result_path),
        "predecessor_integrity": {
            key: value for key, value in result["predecessor_integrity"].items() if "sha256" not in key
        },
        "source_sha256": result["source_sha256"],
    }
