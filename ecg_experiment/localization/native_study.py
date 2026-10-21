"""Patient-separated native ECG reference matching and contextual controls for NTT-06."""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.eda.avance2_features import AFTER_S, BEFORE_S, qrs_peaks
from ecg_experiment.files import sha256_file
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.native_reference_ntt05 import (
    NativeTokens,
    ReferenceBank,
    ReferenceInstance,
    SignalRegion,
    candidate_regions,
    match_region,
    pool_region,
    reference_bank,
    soft_decay,
    unit_vector,
)
from ecg_experiment.paths import to_stored
from ecg_experiment.same_wave_ntt04 import (
    BASELINES,
    FINDINGS,
    METHODS,
    SEEDS,
    balanced,
    classify,
    summarize,
)

NTT04_SHA = "79cfa5130ccc8d275e398ddb15b056a1affe1beb812fa32c52e12a8c2211e789"
NTT05_SHA = "e554a50a695ec6bc96c6b066abee28658d79396a3f90e48899154a032e8d9fed"
NATIVE_METHODS = (
    "native_qrs",
    "native_late",
    "native_wrong_reference",
    "native_global_pair",
    "native_global_eight",
    "native_shuffled_labels",
)
PAIR = ("V1", "V6")


def predecessor_inputs() -> tuple[dict, dict, list[dict], dict, dict]:
    """Verify frozen receipts and patient roles before consuming saved feature payloads.

    Returns
    -------
    tuple
        Selection, integer-indexed features, predictions, predecessor result and bound hashes.
    """
    base = ROOT / "outputs/ntt04_same_wave"
    previous = ROOT / "outputs/ntt05_native_signal_adapter/result.json"
    result_path = base / "result.json"
    if sha256_file(previous) != NTT05_SHA or sha256_file(result_path) != NTT04_SHA:
        raise ValueError("Frozen predecessor result differs")
    result = json.loads(result_path.read_text())
    hashes = {to_stored(previous): NTT05_SHA, to_stored(result_path): NTT04_SHA}
    for name in (
        "reservation_manifest_local.json",
        "selection_local.json",
        "features_local.json",
        "predictions_local.json",
    ):
        path = base / name
        expected = result["artifact_sha256"][to_stored(path)]
        if sha256_file(path) != expected:
            raise ValueError(f"Predecessor artifact differs: {name}")
        hashes[to_stored(path)] = expected
    manifest = json.loads((base / "reservation_manifest_local.json").read_text())
    permitted = {int(r["ecg_id"]): int(r["patient_id"]) for r in manifest["records"] if r["role"] == "train"}
    selection = json.loads((base / "selection_local.json").read_text())
    rows = selection["targets"] + [r for g in selection["support_draws"] for r in g["supports"]]
    if any(permitted.get(r["ecg_id"]) != r["patient_id"] for r in rows):
        raise ValueError("Selected patient outside the frozen training role")
    targets = {r["patient_id"] for r in selection["targets"]}
    supports = {r["patient_id"] for g in selection["support_draws"] for r in g["supports"]}
    if len(targets) != 60 or len(supports) != 27 or targets & supports:
        raise ValueError("Frozen patient separation or denominator differs")
    raw = json.loads((base / "features_local.json").read_text())
    features = {
        int(k): {
            n: (
                np.asarray(v, dtype=np.float32 if n in ("dino", "late") else np.float64)
                if n in ("raw", "raw_full_context", "scalars", "dino", "late")
                else v
            )
            for n, v in row.items()
        }
        for k, row in raw.items()
    }
    predictions = json.loads((base / "predictions_local.json").read_text())
    return selection, features, predictions, result, hashes


def reproduce_predecessor(selection: dict, features: dict, predictions: list[dict], result: dict) -> dict:
    """Reconstruct all nine predecessor methods and its exact patient-bootstrap summary."""
    replayed = classify(features, selection["targets"], selection["support_draws"])
    maximum = 0.0
    if len(replayed) != len(predictions) or len(replayed) != 180:
        raise ValueError("Predecessor prediction count differs")
    for actual, expected in zip(replayed, predictions, strict=True):
        for key in ("patient_id", "ecg_id", "finding", "truth", "seed", "chosen", "correct"):
            if actual[key] != expected[key]:
                raise ValueError(f"Predecessor replay differs: {key}")
        for method in METHODS:
            maximum = max(
                maximum,
                float(np.max(np.abs(np.array(actual["scores"][method]) - expected["scores"][method]))),
            )
    if maximum > 1e-12 or summarize(replayed) != result["summary"]:
        raise ValueError("Predecessor scores or summary differ")
    correct = sum(r["correct"]["raw_full_context"] for r in replayed)
    if correct != 167:
        raise ValueError("NTT-05 integrity count differs")
    return {
        "all_method_choices_exact": True,
        "patient_draw_predictions": 180,
        "methods": len(METHODS),
        "physical_correct": int(correct),
        "maximum_score_error": maximum,
        "summary_exact": True,
    }


def automatic_regions(signal: np.ndarray) -> tuple[list[SignalRegion], list[SignalRegion]]:
    """Detect complete automatic beat anchors without diagnostic labels.

    Parameters
    ----------
    signal : np.ndarray
        Canonical samples-by-twelve-leads physical-mV signal at 500 Hz.

    Returns
    -------
    tuple
        QRS and equally numerous, equal-duration shifted regions in original seconds.
    """
    if signal.shape != (5000, 12) or not np.isfinite(signal).all():
        raise ValueError("Expected finite canonical ten-second signal")
    peaks = qrs_peaks(signal, 500)
    peaks = peaks[(peaks >= int(BEFORE_S * 500)) & (peaks + int(AFTER_S * 500) <= 5000)]
    if len(peaks) < 3:
        raise ValueError("Fewer than three complete automatic beats")
    qrs = [SignalRegion(float(p / 500 - 0.15), float(p / 500 + 0.15), PAIR) for p in peaks]
    late = [SignalRegion(float(p / 500 + 0.25), float(p / 500 + 0.55), PAIR) for p in peaks]
    return qrs, late


def descriptors(
    tokens: NativeTokens, qrs: list[SignalRegion], late: list[SignalRegion]
) -> dict[str, np.ndarray]:
    """Average normalized beat vectors equally within patient and retain global controls."""
    if not qrs or len(qrs) != len(late):
        raise ValueError("QRS and shifted controls require equal nonzero counts")
    return {
        "native_qrs": unit_vector(np.mean([pool_region(tokens, r) for r in qrs], axis=0)),
        "native_late": unit_vector(np.mean([pool_region(tokens, r) for r in late], axis=0)),
        "native_global_pair": pool_region(tokens, SignalRegion(0, 10, PAIR)),
        "native_global_eight": pool_region(
            tokens, SignalRegion(0, 10, ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6"))
        ),
    }


def banks_for_draw(features: dict, groups: list[dict], seed: int, space: str) -> dict[str, ReferenceBank]:
    """Build equal-patient prototypes in one frozen embedding space."""
    banks = {}
    for method in ("native_qrs", "native_late", "native_global_pair", "native_global_eight"):
        instances = [
            ReferenceInstance(
                str(r["patient_id"]),
                g["finding"],
                features[r["ecg_id"]][method],
                "source record annotation; automatic regions are weak references",
                space,
            )
            for g in groups
            if g["seed"] == seed
            for r in g["supports"]
        ]
        banks[method] = reference_bank(instances, FINDINGS)
    return banks


def native_predictions(
    features: dict, selection: dict, previous: list[dict], space: str
) -> tuple[list[dict], dict]:
    """Compare native prototypes and fixed controls while retaining historical physical choices."""
    old = {(r["seed"], r["ecg_id"]): r for r in previous}
    predictions, all_banks = [], {}
    for seed in SEEDS:
        banks = banks_for_draw(features, selection["support_draws"], seed, space)
        all_banks[seed] = banks["native_qrs"]
        for target in selection["targets"]:
            if any(str(target["patient_id"]) in bank.patients for bank in banks.values()):
                raise ValueError("Target patient leaked into reference bank")
            feature = features[target["ecg_id"]]
            scores = {m: np.clip(b.prototypes @ feature[m], -1, 1) for m, b in banks.items()}
            scores["native_wrong_reference"] = np.clip(
                banks["native_late"].prototypes @ feature["native_qrs"], -1, 1
            )
            scores["native_shuffled_labels"] = np.roll(scores["native_qrs"], 1)
            saved = old[(seed, target["ecg_id"])]
            scores.update({m: np.asarray(v) for m, v in saved["scores"].items()})
            chosen = {m: int(np.argmax(v)) for m, v in scores.items()}
            truth = FINDINGS.index(target["finding"])
            predictions.append(
                {
                    "patient_id": target["patient_id"],
                    "ecg_id": target["ecg_id"],
                    "finding": target["finding"],
                    "seed": seed,
                    "truth": truth,
                    "scores": {m: v.tolist() for m, v in scores.items()},
                    "chosen": chosen,
                    "correct": {m: int(c == truth) for m, c in chosen.items()},
                }
            )
    return predictions, all_banks


def native_summary(predictions: list[dict], draws: int = 2000) -> dict:
    """Apply the frozen primary gate and patient-paired regional-control comparisons."""
    patients = sorted({r["patient_id"] for r in predictions})
    methods = (*NATIVE_METHODS, *METHODS)
    truth = np.array([next(r["truth"] for r in predictions if r["patient_id"] == p) for p in patients])
    correct = {
        m: np.array(
            [np.mean([r["correct"][m] for r in predictions if r["patient_id"] == p]) for p in patients]
        )
        for m in methods
    }
    accuracy = {m: balanced(v, truth) for m, v in correct.items()}
    best = max(BASELINES, key=accuracy.get)
    controls = ("native_late", "native_global_pair", "native_global_eight")
    differences = {m: [] for m in ("primary", *controls)}
    rng, groups, skipped = np.random.default_rng(20261018), patient_groups(np.array(patients)), 0
    for _ in range(draws):
        rows = patient_resample(groups, rng)
        if len(np.unique(truth[rows])) != 3:
            skipped += 1
            continue
        first = balanced(correct["native_qrs"][rows], truth[rows])
        differences["primary"].append(first - max(balanced(correct[m][rows], truth[rows]) for m in BASELINES))
        for m in controls:
            differences[m].append(first - balanced(correct[m][rows], truth[rows]))
    if not differences["primary"]:
        raise ValueError("No bootstrap retained three classes")
    exact = (
        sum(
            (
                Fraction(
                    sum(
                        r["correct"]["native_qrs"] - r["correct"][best]
                        for r in predictions
                        if r["truth"] == i
                    ),
                    sum(r["truth"] == i for r in predictions),
                )
                for i in range(3)
            ),
            Fraction(),
        )
        / 3
    )
    intervals = {
        m: {
            "difference": float(exact) if m == "primary" else accuracy["native_qrs"] - accuracy[m],
            "ci_low": float(np.percentile(v, 2.5)),
            "ci_high": float(np.percentile(v, 97.5)),
            "draws": draws,
            "skipped_draws": skipped,
            "seed": 20261018,
        }
        for m, v in differences.items()
    }
    intervals["primary"]["baseline_maximum_recomputed_each_draw"] = True
    return {
        "balanced_accuracy": accuracy,
        "best_physical_baseline": best,
        "primary": intervals.pop("primary"),
        "regional_control_differences": intervals,
        "advance_gate_passed": bool(
            exact >= Fraction(1, 20) and np.percentile(differences["primary"], 2.5) > 0
        ),
        "by_draw": {
            str(s): {
                m: float(np.mean([r["correct"][m] for r in predictions if r["seed"] == s])) for m in methods
            }
            for s in SEEDS
        },
        "by_finding": {
            f: {
                m: float(np.mean([r["correct"][m] for r in predictions if r["finding"] == f]))
                for m in methods
            }
            for f in FINDINGS
        },
        "mean_score_margin": {
            m: float(
                np.mean([np.sort(r["scores"][m])[-1] - np.sort(r["scores"][m])[-2] for r in predictions])
            )
            for m in NATIVE_METHODS
        },
    }


def region_matches(
    tokens: NativeTokens, qrs: list[SignalRegion], bank: ReferenceBank, patient: str
) -> list[dict]:
    """Return native automatic-region similarities and separate redundancy-adjusted scores."""
    grid = candidate_regions(10, (PAIR,), (0.4, 0.8, 1.6), 0.4)
    regions = qrs + [r for r in grid if r not in qrs]
    matches = [match_region(tokens, r, bank, patient, -1, 0) for r in regions]
    adjusted = {id(m): float(s) for m, s in soft_decay(matches)}
    return [
        {
            "start_seconds": m.region.start,
            "stop_seconds": m.region.stop,
            "leads": list(m.region.leads),
            "proposal": "automatic_qrs" if i < len(qrs) else "fixed_grid",
            "finding": m.finding,
            "cosine_scores": m.scores.tolist(),
            "raw_similarity": m.similarity,
            "margin": m.margin,
            "adjusted_similarity": adjusted[id(m)],
            "context_scope": m.context_scope,
            "embedding_space": m.embedding_space,
            "patient_id_local": patient,
            "record_frame_local": m.record_frame,
        }
        for i, m in enumerate(matches)
    ]


def regions_from_saved(rows: list[dict]) -> list[SignalRegion]:
    """Restore typed half-open regions without changing their coordinate frame."""
    return [SignalRegion(r["start"], r["stop"], tuple(r["leads"])) for r in rows]


def validate_profile(profile: dict, sources: dict, runtime: dict, checkpoint_sha: str) -> None:
    """Require the actual passing profile from the same sources, checkpoint and runtime."""
    if (
        profile.get("stage") != "synthetic_pretrained_cpu_profile"
        or profile.get("source_sha256") != sources
        or profile.get("runtime") != runtime
        or profile.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("CPU profile identity, source, checkpoint or runtime differs")
    required = {
        "finite_dense_shape",
        "official_mean_equivalence",
        "deterministic_repeat",
        "forward_budget",
        "rss_budget",
        "profile_budget",
        "all_parameters_frozen",
    }
    if (
        profile.get("profile_gate_passed") is not True
        or set(profile.get("gates", {})) != required
        or not all(value is True for value in profile["gates"].values())
    ):
        raise ValueError("CPU profile gate did not pass completely")
    if (
        profile["dense_shape"] != [400, 768]
        or profile["parameters"] != 85374720
        or profile["mean_absolute_max_error"] > 1e-5
        or profile["repeat_absolute_max_error"] > 1e-6
        or max(*profile["forward_seconds"], profile["official_pooled_seconds"]) > 20
        or profile["seconds"] > 300
        or profile["peak_rss_bytes"] > 6 * 1024**3
    ):
        raise ValueError("CPU profile measurements violate the prospective admission limits")
