"""Saved-only public PTB-XL examples for the standalone real-ECG explanation notebook."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from scipy.special import expit

from ecg_experiment.explanation_rule import rule_marks
from ecg_experiment.files import sha256_file
from ecg_experiment.lead_wave_maps import UnitMap, jepa_unit_map
from ecg_experiment.pvc_switch import switch_explain
from ecg_experiment.waveforms import LEADS

CASES = (
    (47, "Normal ECG annotation"),
    (91, "A second normal ECG annotation"),
    (219, "Premature ventricular beat annotation"),
    (2135, "Complete left bundle branch block annotation"),
    (1123, "Complete right bundle branch block annotation"),
    (127, "Non-diagnostic T abnormalities annotation"),
    (184, "Combined anteroseptal and inferolateral infarction-pattern annotations"),
    (1577, "Inferior infarction-pattern annotation"),
    (30, "Left ventricular hypertrophy annotation: not flagged"),
)
TEACHING = {
    47: "The dataset describes normal sinus rhythm. Compare its repeated beats with the examples below.",
    91: "A second normal ECG annotation shows that normal recordings can have different shapes.",
    219: "PVC means an early extra ventricular heartbeat. Two saved red candidates mark one unusual beat.",
    2135: "Left bundle branch block describes delayed electrical activation, which can widen QRS.",
    1123: "Right bundle branch block describes delayed activation through the right conduction branch.",
    127: "This annotation describes non-specific T-wave abnormalities, not a proven cause such as ischemia.",
    184: "Anterior and inferior infarction-pattern codes are recorded; they do not prove an active event.",
    1577: "The report says probable old inferior infarct. This is an ECG-pattern annotation.",
    30: "LVH is a voltage pattern. Enlargement is unverified; voltage-alone referral is undefined.",
}
ARTIFACTS = {
    "units": ("experiment042_lead_wave_maps_v1", "unit_scores.npz"),
    "attention": ("experiment043_ann_heads_v1/stage2", "token_maps.npz"),
    "classifier": ("experiment046_pipeline_v4_v1", "predictions.npz"),
    "switch": ("experiment048_pvc_switch_v1", "explanations.npz"),
    "findings": ("experiment049_focal_switch_v1", "explanations.npz"),
}


def checked_saved_arrays(root: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    """
    Verify and load frozen real-data caches without importing an experiment runner.

    Parameters
    ----------
    root : Path
        Repository root containing local completed experiment receipts.

    Returns
    -------
    tuple[dict, dict]
        Named saved arrays and verified repository-relative artifact hashes.
    """
    arrays, hashes = {}, {}
    for name, (folder, filename) in ARTIFACTS.items():
        directory = root / "outputs" / folder
        receipt = json.loads((directory / "result.json").read_text())
        digest = sha256_file(directory / filename)
        if digest != receipt["outputs_sha256"][filename]:
            raise ValueError(f"Frozen real-ECG artifact changed: {folder}/{filename}")
        with np.load(directory / filename) as saved:
            if name == "classifier":
                keys = ("development_record_ids", "development_v4_logit", "development_v3_binary")
            else:
                keys = saved.files
            arrays[name] = {key: saved[key] for key in keys}
        hashes[f"outputs/{folder}/{filename}"] = digest
    ids = arrays["units"]["ecg_ids"]
    for name in ("attention", "switch", "findings"):
        if not np.array_equal(ids, arrays[name]["ecg_ids"]):
            raise ValueError("Real-ECG cached row identifiers differ")
    if not np.array_equal(
        np.asarray([f"ptbxl:{int(value)}" for value in ids]),
        arrays["classifier"]["development_record_ids"],
    ):
        raise ValueError("Classifier rows do not align with cached localization rows")
    if not np.array_equal(arrays["switch"]["z_pvc"], arrays["findings"]["z_pvc"]):
        raise ValueError("Saved PVC finding scores differ")
    return arrays, hashes


def recorded_combined_rule(root: Path) -> dict[str, Any]:
    """
    Read previously executed notebook decisions and its rounded threshold bounds.

    Parameters
    ----------
    root : Path
        Repository root containing executed notebook 13.

    Returns
    -------
    dict
        Saved decisions and rounding intervals; no threshold is fitted or re-estimated.
    """
    path = root / "notebooks/13-jr-pipeline-v4-explained.ipynb"
    notebook = json.loads(path.read_text())
    text = "\n".join(
        "".join(output.get("text", []))
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output["output_type"] == "stream"
    )
    found = re.search(r"referral thresholds: ensemble score ([+-]?[\d.]+), finding score ([+-]?[\d.]+)", text)
    if found is None:
        raise ValueError("Previously executed notebook lacks recorded combined referral thresholds")
    thresholds = [float(value) for value in found.groups()]
    decisions = {}
    for identifier, decision in re.findall(r"ECG (\d+) \| (REFERRED|not referred) \|", text):
        identifier, referred = int(identifier), decision == "REFERRED"
        if identifier in decisions and decisions[identifier] != referred:
            raise ValueError("Previously executed referral decisions disagree")
        decisions[identifier] = referred
    return {
        "name": "Notebook 13 pipeline-v4 combined classifier-and-finding demonstration rule",
        "source": "notebooks/13-jr-pipeline-v4-explained.ipynb",
        "source_sha256": sha256_file(path),
        "displayed_thresholds": {"ensemble_logit": thresholds[0], "finding_z": thresholds[1]},
        "threshold_rounding_intervals": {
            "ensemble_logit": [thresholds[0] - 0.0005, thresholds[0] + 0.0005],
            "finding_z": [thresholds[1] - 0.0005, thresholds[1] + 0.0005],
        },
        "saved_executed_decisions": decisions,
        "normal_reference_records": 463,
        "referral_budget": 0.05,
        "thresholds_refitted": False,
    }


def combined_decision(identifier: int, logit: float, finding: float, rule: dict) -> tuple[bool | None, str]:
    """
    Reproduce a recorded decision or prove it invariant over recorded rounding bounds.

    Parameters
    ----------
    identifier : int
        Public development ECG identifier.
    logit, finding : float
        Saved v4 ensemble logit and maximum saved PVC/WPW finding z-score.
    rule : dict
        Recorded demonstration rule and rounded cutoff intervals.

    Returns
    -------
    tuple[bool or None, str]
        Referral proxy and exact provenance; None indicates a rounding-ambiguous example.
    """
    limits = rule["threshold_rounding_intervals"]
    binary, head = limits["ensemble_logit"], limits["finding_z"]
    bounded = True if logit > binary[1] or finding > head[1] else None
    if logit <= binary[0] and finding <= head[0]:
        bounded = False
    recorded = rule["saved_executed_decisions"].get(identifier)
    if recorded is not None:
        if bounded is not None and recorded != bounded:
            raise ValueError("Saved scores contradict a previously executed referral decision")
        return recorded, "Decision read from previously executed notebook 13 stdout"
    return bounded, "Reconstructed from saved scores; unchanged over reported threshold rounding intervals"


def read_real_signal(root: Path, filename: str) -> np.ndarray:
    """
    Read one public development waveform in native physical mV and canonical lead order.

    Parameters
    ----------
    root : Path
        Repository root containing PTB-XL 1.0.3 locally.
    filename : str
        Metadata's relative high-resolution waveform stem.

    Returns
    -------
    np.ndarray
        Finite twelve-lead by 5000-sample physical waveform.
    """
    path = Path(filename)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("PTB waveform stem escapes the public release")
    record = wfdb.rdrecord(str(root / "data/raw/ptb-xl/1.0.3" / path))
    names = [lead.upper() for lead in record.sig_name]
    canonical = [lead.upper() for lead in LEADS]
    if record.fs != 500 or record.sig_len != 5000 or len(names) != 12 or set(names) != set(canonical):
        raise ValueError("Unexpected PTB-XL sampling, duration or lead names")
    if set(record.units) != {"mV"} or not np.isfinite(record.p_signal).all():
        raise ValueError("PTB-XL physical units or samples are invalid")
    return record.p_signal[:, [names.index(lead) for lead in canonical]].T


def cached_maps(arrays: dict, row: int) -> tuple[UnitMap, UnitMap]:
    """
    Reconstruct one record's original saved normal-distance and attention candidates.

    Parameters
    ----------
    arrays : dict
        Verified frozen real-data caches.
    row : int
        Shared development row index.

    Returns
    -------
    tuple[UnitMap, UnitMap]
        Original U_B units and attention units, without running a model or fitting references.
    """
    units = arrays["units"]
    left, right = units["U_B_offsets"][row : row + 2]
    first = UnitMap(*[units[f"U_B_{field}"][left:right] for field in ("scores", "leads", "starts", "ends")])
    return first, jepa_unit_map(arrays["attention"]["attention_jepa"][row])


def candidate_marks(
    first: UnitMap, second: UnitMap, referred: bool, use_first: bool, thresholds: dict
) -> list[dict]:
    """
    Recreate the existing display rule and distinguish forced top candidates from threshold passes.

    Parameters
    ----------
    first, second : UnitMap
        Saved normal-distance and attention candidates.
    referred, use_first : bool
        Recorded referral proxy and frozen PVC-head map switch.
    thresholds : dict
        Original display thresholds from Experiment 048.

    Returns
    -------
    list[dict]
        Display marks, all explicitly algorithm candidates rather than expert pathological regions.
    """
    if not referred:
        return []
    explanation = switch_explain(first, second, use_first, thresholds["U_B"], thresholds["attention_jepa"])
    method = "U_B" if use_first else "attention_jepa"
    cutoff = thresholds[method]
    results = []
    for lead, start, end in rule_marks(explanation, cutoff):
        matches = np.flatnonzero(
            (explanation.units.leads == lead)
            & np.isclose(explanation.units.starts, start)
            & np.isclose(explanation.units.ends, end)
        )
        score = float(explanation.units.scores[matches[0]])
        above = score > cutoff
        results.append(
            {
                "label": f"{'Normal-distance' if use_first else 'Attention'} candidate for review"
                + (" (top-ranked, below display cutoff)" if not above else ""),
                "lead": int(lead),
                "lead_name": LEADS[lead],
                "start": float(start),
                "end": float(end),
                "color": ("#c0392b" if use_first else "#2471a3") if above else "#b9770e",
                "score": score,
                "display_threshold": float(cutoff),
                "above_display_threshold": above,
                "expert_annotation": False,
            }
        )
    return results


def load_real_ptb_cases(root: Path) -> list[dict[str, Any]]:
    """
    Extract nine score-independent public real ECGs with saved classifier and map provenance.

    Parameters
    ----------
    root : Path
        Repository root. Requires existing real-data caches and executed notebook 13 only.

    Returns
    -------
    list[dict[str, Any]]
        Self-contained physical signals, labels, cached model outputs, referral proxies and marks.
        No synthetic data, new inference, reference fitting, training or closed test is accessed.
    """
    arrays, hashes = checked_saved_arrays(root)
    raw = root / "data/raw/ptb-xl/1.0.3"
    metadata = pd.read_csv(raw / "ptbxl_database.csv", index_col="ecg_id").loc[[value for value, _ in CASES]]
    statements = pd.read_csv(raw / "scp_statements.csv", index_col=0)
    metadata_hashes = {
        "data/raw/ptb-xl/1.0.3/ptbxl_database.csv": sha256_file(raw / "ptbxl_database.csv"),
        "data/raw/ptb-xl/1.0.3/scp_statements.csv": sha256_file(raw / "scp_statements.csv"),
    }
    rule = recorded_combined_rule(root)
    prior = json.loads((root / "outputs/experiment048_pvc_switch_v1/result.json").read_text())
    switch = float(prior["switch_thresholds"]["0.975"])
    ids = arrays["units"]["ecg_ids"]
    results = []
    for identifier, title in CASES:
        row = int(np.flatnonzero(ids == identifier)[0])
        source = metadata.loc[identifier]
        if source.strat_fold != 9:
            raise ValueError("Selected waveform is not in the established development fold")
        codes = ast.literal_eval(source.scp_codes)
        label_rows = [
            {
                "code": code,
                "source_confidence": float(confidence),
                "description": str(statements.at[code, "description"]),
            }
            for code, confidence in codes.items()
        ]
        logit = float(arrays["classifier"]["development_v4_logit"][row])
        pvc, wpw = float(arrays["switch"]["z_pvc"][row]), float(arrays["findings"]["z_wpw"][row])
        finding = max(pvc, wpw)
        referred, provenance = combined_decision(identifier, logit, finding, rule)
        legacy = bool(arrays["switch"]["E_q0.975_referred"][row])
        rule_name = rule["name"]
        if referred is None:
            referred, provenance = (
                legacy,
                "Rounding-ambiguous combined rule: explicitly use saved048 E-only proxy",
            )
            rule_name = "Experiment 048 saved ensemble-only referral proxy"
        signal = read_real_signal(root, source.filename_hr)
        first, second = cached_maps(arrays, row)
        use_first = pvc > switch
        notes = [
            "Real public development ECG; this example is not independent performance evidence.",
            "All SCP codes are shown. Source confidence values are not disease probabilities.",
            "Sinus rhythm is a rhythm annotation and does not establish a clinically healthy heart.",
            "Colored areas are model candidates for review, without expert pathological-region labels.",
            "U_B P/QRS/ST/T are fixed offset windows, not verified physiological wave boundaries.",
            "The classifier score is not a calibrated clinical diagnosis probability.",
        ]
        if identifier == 184:
            notes.append(
                "ASMI and ILMI both have source confidence100; this is not an isolated anterior infarct."
            )
        if identifier in (127, 30):
            notes.append(
                "The saved screening rule does not refer this annotated ECG; this is not a clinical verdict."
            )
        if identifier in (47, 91):
            notes.append("NORM100/SR means a normal ECG annotation, not proof of general clinical health.")
        results.append(
            {
                "id": identifier,
                "kind": "ptb",
                "title": title,
                "plain_language": TEACHING[identifier],
                "source": "Public PTB-XL1.0.3 development ECG",
                "source_url": "https://physionet.org/content/ptb-xl/1.0.3/",
                "source_path": f"data/raw/ptb-xl/1.0.3/{source.filename_hr}",
                "source_hashes": {
                    **metadata_hashes,
                    **{
                        f"data/raw/ptb-xl/1.0.3/{source.filename_hr}.{extension}": sha256_file(
                            raw / f"{source.filename_hr}.{extension}"
                        )
                        for extension in ("hea", "dat")
                    },
                },
                "fs": 500,
                "units": "mV",
                "leads": list(LEADS),
                "signal": signal.tolist(),
                "time": (np.arange(5000) / 500).tolist(),
                "marks": candidate_marks(first, second, referred, use_first, prior["map_thresholds"]),
                "annotation_notes": notes,
                "annotations": label_rows,
                "codes": codes,
                "source_report": str(source.report),
                "patient_id": f"ptbxl:{source.patient_id}",
                "classifier": {
                    "v4_logit": logit,
                    "v4_sigmoid_model_score": float(expit(logit)),
                    "v3_saved_model_score": float(arrays["classifier"]["development_v3_binary"][row]),
                    "pvc_z": pvc,
                    "wpw_z": wpw,
                    "finding_z": finding,
                    "referred": referred,
                    "decision_provenance": provenance,
                    "048_E_only_referred": legacy,
                    "rule_name": rule_name,
                    "threshold_rounding_intervals": rule["threshold_rounding_intervals"],
                    "ranking_only_not_calibrated_clinical_probability": True,
                },
                "map_method": "Saved normal-distance U_B" if use_first else "Saved JEPA attention",
                "map_switch_threshold": switch,
                "map_display_thresholds": prior["map_thresholds"],
                "saved_artifact_hashes": hashes,
                "combined_rule_source_sha256": rule["source_sha256"],
                "selection": (
                    "Fixed label-based examples; no score-based selection; explicit existing screening misses"
                ),
            }
        )
    return results
