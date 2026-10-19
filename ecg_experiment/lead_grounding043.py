"""Fixed finding-conditioned benchmark lead grounding using cached ECG measurements."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment import cpu_screening041 as cached
from ecg_experiment.files import sha256_file, write_json_atomic, write_npz_atomic
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.paths import to_stored

LEADS = cached.LEADS
MEASUREMENTS = ("std", "ptp", "kurt", "r_amp", "s_depth", "q_depth", "qrs_area", "st", "t_amp")
SHARED_FEATURES = ("context_code", "lead_code", *(f"offered_{lead}" for lead in LEADS))
WAVE_FEATURES = (
    *SHARED_FEATURES,
    *(f"raw__{name}" for name in cached.RAW_FEATURES),
    *(f"candidate__{name}" for name in MEASUREMENTS),
)
CONFIG = dict(cached.HGB_CONFIG)
SEED = 20261018
DRAWS = 2000
SOURCE = ROOT / "data/raw/ecg-reasoning-benchmark/90a53f1/ptbxl.jsonl"
METADATA = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
SOURCE_SHA = "353ddc413b68c751a0e2c5c44a11a182de4519e278c5ae49fe0bf089c674b39b"
BLOB_SHA = "da62062c76828d53297e76f00ff02153e5713246"
METADATA_SHA = "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"
SOURCE_BYTES = 13659066
EXPECTED_ROLES = {"fit": (1117, 823, 776), "unused": (154, 110, 99), "development": (137, 100, 97)}
MODELS = ("context_hgb", "wave_hgb", "context_lookup")
Query = dict[str, Any]


def parse_grounding(item: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate literal offered options and exact indexed answers, then canonicalize leads."""
    options, answer, indices = item["options"], item["answer"], item["answer_idx"]
    if not isinstance(options, list) or not options or not all(isinstance(x, str) for x in options):
        raise ValueError("grounding options must be a nonempty string list")
    allowed = {f"Lead {lead}" for lead in LEADS}
    if not set(options) <= allowed or len(set(options)) != len(options):
        raise ValueError("unsupported literal lead options or repeated option")
    if not isinstance(answer, list) or not isinstance(indices, list):
        raise ValueError("grounding answers and indices must both be lists")
    if any(type(index) is not int or not 0 <= index < len(options) for index in indices):
        raise ValueError("unsupported answer index")
    if len(set(indices)) != len(indices) or [options[index] for index in indices] != answer:
        raise ValueError("answer text and indexed options disagree")
    offered = tuple(lead for lead in LEADS if f"Lead {lead}" in options)
    correct = tuple(lead for lead in LEADS if f"Lead {lead}" in answer)
    if not correct or len(correct) == len(offered):
        raise ValueError("empty or all-positive grounding target")
    return offered, correct


def validate_metadata(metadata: pd.DataFrame) -> None:
    """Require unique ECG joins and a single official fold per patient."""
    if metadata.index.has_duplicates or metadata.index.isna().any():
        raise ValueError("metadata ECG IDs must be unique and present")
    if metadata[["patient_id", "strat_fold"]].isna().any().any():
        raise ValueError("missing official patient or fold")
    if not metadata.strat_fold.isin(range(1, 11)).all():
        raise ValueError("unsupported official fold")
    if metadata.groupby("patient_id").strat_fold.nunique().max() != 1:
        raise ValueError("official patient crosses folds")


def load_queries(path: Path, metadata: pd.DataFrame) -> tuple[list[Query], dict[str, int]]:
    """Parse only non-test grounding steps, retaining distinct questions and offered sets."""
    validate_metadata(metadata)
    unique: dict[tuple[Any, ...], Query] = {}
    audit = {
        "source_rows": 0,
        "fold10_rows_skipped_before_steps": 0,
        "duplicate_queries": 0,
        "conflicting_queries": 0,
        "missing_joins": 0,
        "empty_targets": 0,
        "all_positive_targets": 0,
    }
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            audit["source_rows"] += 1
            ecg_id = int(row["metadata"]["ecg_id"])
            if ecg_id not in metadata.index:
                raise ValueError("benchmark ECG is absent from official metadata")
            record = metadata.loc[ecg_id]
            fold = int(record.strat_fold)
            if fold == 10:
                audit["fold10_rows_skipped_before_steps"] += 1
                continue
            _append_queries(row, ecg_id, int(record.patient_id), fold, unique, audit)
    return list(unique.values()), audit


def _append_queries(
    row: dict[str, Any],
    ecg_id: int,
    patient_id: int,
    fold: int,
    unique: dict[tuple[Any, ...], Query],
    audit: dict[str, int],
) -> None:
    """Append validated lead questions from one already role-filtered payload row."""
    diagnosis = row["metadata"]["target_dx"]
    if not isinstance(diagnosis, str):
        raise ValueError("target diagnosis context must be text")
    for step in row["data"]["reasoning"]:
        for item in step["ecg_grounding"]:
            if item["question_type"] != "lead_grounding":
                continue
            question = item["question"]
            if not isinstance(question, str):
                raise ValueError("question context must be text")
            offered, correct = parse_grounding(item)
            context = (diagnosis, question)
            key = (ecg_id, context, offered)
            query = {
                "ecg_id": ecg_id,
                "patient_id": patient_id,
                "fold": fold,
                "context": context,
                "offered": offered,
                "correct": correct,
            }
            if key in unique:
                if unique[key]["correct"] != correct:
                    raise ValueError("conflicting answers for a grounding case key")
                audit["duplicate_queries"] += 1
            else:
                unique[key] = query


def informative_groups(queries: list[Query]) -> set[tuple[Any, ...]]:
    """Find exact context/offered-set answer variation using fitting queries only."""
    answers: dict[tuple[Any, ...], set[tuple[str, ...]]] = defaultdict(set)
    for query in queries:
        if query["fold"] <= 7:
            answers[(query["context"], query["offered"])].add(query["correct"])
    return {key for key, values in answers.items() if len(values) >= 2}


def qualify(queries: list[Query], audit: dict[str, int], require_expected: bool = True) -> dict[str, Any]:
    """Freeze prefit role accounting and train-only informative groups, with no scores."""
    roles = {
        "fit": [q for q in queries if q["fold"] <= 7],
        "unused": [q for q in queries if q["fold"] == 8],
        "development": [q for q in queries if q["fold"] == 9],
    }
    patients = [{q["patient_id"] for q in role} for role in roles.values()]
    if any(patients[a] & patients[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("grounding patients cross roles")
    groups = informative_groups(queries)
    informative = [q for q in roles["development"] if (q["context"], q["offered"]) in groups]
    contexts: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
    for query in roles["fit"]:
        contexts[query["context"]].add(query["correct"])
    counts = {
        name: {
            "queries": len(role),
            "ecgs": len({q["ecg_id"] for q in role}),
            "patients": len({q["patient_id"] for q in role}),
            "answer_cardinality": dict(sorted(Counter(len(q["correct"]) for q in role).items())),
            "offered_cardinality": dict(sorted(Counter(len(q["offered"]) for q in role).items())),
        }
        for name, role in roles.items()
    }
    result = {
        "roles": counts,
        "parser": audit,
        "training_contexts": len(contexts),
        "constant_training_contexts": sum(len(values) == 1 for values in contexts.values()),
        "informative_training_groups": len(groups),
        "informative_development_queries": len(informative),
        "informative_development_patients": len({q["patient_id"] for q in informative}),
    }
    if require_expected:
        _expected_qualification(result)
    return result


def _expected_qualification(result: dict[str, Any]) -> None:
    """Reject any departure from the frozen read-only setup accounting."""
    for name, expected in EXPECTED_ROLES.items():
        observed = result["roles"][name]
        if tuple(observed[key] for key in ("queries", "ecgs", "patients")) != expected:
            raise ValueError(f"{name} grounding role fails frozen qualification counts")
    audit = result["parser"]
    if any(
        audit[name] != 0
        for name in (
            "duplicate_queries",
            "conflicting_queries",
            "missing_joins",
            "empty_targets",
            "all_positive_targets",
        )
    ):
        raise ValueError("grounding parser audit fails frozen zero counts")
    expected = (23, 0, 69, 56)
    keys = (
        "training_contexts",
        "constant_training_contexts",
        "informative_development_queries",
        "informative_development_patients",
    )
    if tuple(result[key] for key in keys) != expected:
        raise ValueError("context variation or informative-subset qualification changed")
    if result["roles"]["development"]["patients"] < 50 or result["informative_development_patients"] < 30:
        raise ValueError("insufficient development or informative patients")


def context_codes(queries: list[Query]) -> dict[tuple[str, str], int]:
    """Assign stable context categories from fitting queries alone."""
    return {
        context: index
        for index, context in enumerate(sorted({q["context"] for q in queries if q["fold"] <= 7}))
    }


def expand_queries(
    queries: list[Query], contexts: dict[tuple[str, str], int], raw: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Expand offered leads with equal total query weight and optional cached measurements."""
    rows, labels, weights = [], [], []
    for query in queries:
        offered = query["offered"]
        mask = [float(lead in offered) for lead in LEADS]
        global_values = (
            raw.loc[query["ecg_id"], list(cached.RAW_FEATURES)].to_list() if raw is not None else []
        )
        for lead in offered:
            row = [contexts.get(query["context"], np.nan), LEADS.index(lead), *mask, *global_values]
            if raw is not None:
                row.extend(float(raw.at[query["ecg_id"], f"{name}_{lead}"]) for name in MEASUREMENTS)
            rows.append(row)
            labels.append(int(lead in query["correct"]))
            weights.append(1 / len(offered))
    columns = WAVE_FEATURES if raw is not None else SHARED_FEATURES
    values = pd.DataFrame(rows, columns=list(columns), dtype=float).replace([np.inf, -np.inf], np.nan)
    return values, np.asarray(labels, dtype=int), np.asarray(weights, dtype=float)


def feature_missingness(queries: list[Query], values: pd.DataFrame, raw: pd.DataFrame) -> dict[str, Any]:
    """Count raw missing values per unique ECG and candidate values per expanded offered row."""
    raw_rows = raw.loc[sorted({q["ecg_id"] for q in queries}), list(cached.RAW_FEATURES)]
    candidates = values[[f"candidate__{name}" for name in MEASUREMENTS]]
    return {
        "raw_unique_ecg_count": len(raw_rows),
        "raw_ecgs_with_any_missing": int(raw_rows.isna().any(axis=1).sum()),
        "raw_missing_counts_per_unique_ecg": {
            name: int(count) for name, count in raw_rows.isna().sum().items()
        },
        "expanded_offered_row_count": len(values),
        "candidate_missing_counts_per_expanded_row": {
            name: int(count) for name, count in candidates.isna().sum().items()
        },
        "unknown_context_expanded_rows": int(values.context_code.isna().sum()),
    }


def fit_models(
    values: pd.DataFrame, y: np.ndarray, weights: np.ndarray
) -> tuple[dict[str, Any], dict[str, float]]:
    """Fit exactly the two fixed categorical HGB arms on identical candidate rows and weights."""
    numeric = values.drop(columns=["context_code", "lead_code"])
    if numeric.isna().all().any():
        raise ValueError("entirely missing numeric training column is unsupported by installed HGB")
    if values.context_code.isna().any():
        raise ValueError("fitting contexts must have training categories")
    if not 0 < values.context_code.nunique() <= 255:
        raise ValueError("context categories exceed native histogram capacity")
    models, timings = {}, {}
    for name, columns in (("context_hgb", SHARED_FEATURES), ("wave_hgb", WAVE_FEATURES)):
        model = HistGradientBoostingClassifier(categorical_features=["context_code", "lead_code"], **CONFIG)
        start = time.monotonic()
        model.fit(values[list(columns)], y, sample_weight=weights)
        models[name], timings[name] = model, time.monotonic() - start
    return models, timings


def fit_lookup(queries: list[Query]) -> dict[str, Any]:
    """Count weighted candidates for the fixed Beta/global/context/exact-set hierarchy."""
    global_counts: dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(2))
    context_counts: dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(2))
    exact_counts: dict[Any, np.ndarray] = defaultdict(lambda: np.zeros(2))
    for query in queries:
        if query["fold"] > 7:
            raise ValueError("lookup fit accepts only official fitting folds")
        for lead in query["offered"]:
            contribution = np.array([float(lead in query["correct"]), 1.0]) / len(query["offered"])
            global_counts[lead] += contribution
            context_counts[(query["context"], lead)] += contribution
            exact_counts[(query["context"], query["offered"], lead)] += contribution
    return {
        "global_counts": dict(global_counts),
        "context_counts": dict(context_counts),
        "exact_counts": dict(exact_counts),
        "beta_prior": [1, 1],
        "shrinkage_strength": 10,
    }


def lookup_probability(state: dict[str, Any], query: Query, lead: str) -> float:
    """Apply unseen lead, context and offered-set fallbacks without fitted-data updates."""
    if lead not in state["global_counts"]:
        return 0.5
    positive, total = state["global_counts"][lead]
    probability = (positive + 1) / (total + 2)
    context = (query["context"], lead)
    if context not in state["context_counts"]:
        return float(probability)
    positive, total = state["context_counts"][context]
    probability = (positive + 10 * probability) / (total + 10)
    exact = (query["context"], query["offered"], lead)
    if exact in state["exact_counts"]:
        positive, total = state["exact_counts"][exact]
        probability = (positive + 10 * probability) / (total + 10)
    return float(probability)


def query_metrics(queries: list[Query], scores: np.ndarray) -> dict[str, np.ndarray]:
    """Compute tie-aware AP, Brier and strict-threshold exact sets on offered leads only."""
    if scores.ndim != 1 or len(scores) != sum(len(q["offered"]) for q in queries):
        raise ValueError("candidate scores must align exactly to the offered-query expansion")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("candidate probabilities must be finite and within [0,1]")
    metrics: dict[str, list[float]] = {"average_precision": [], "brier": [], "exact_set_accuracy": []}
    offset = 0
    for query in queries:
        y = np.array([lead in query["correct"] for lead in query["offered"]], dtype=int)
        prob = scores[offset : offset + len(y)]
        metrics["average_precision"].append(float(average_precision_score(y, prob)))
        metrics["brier"].append(float(np.mean((prob - y) ** 2)))
        metrics["exact_set_accuracy"].append(float(np.array_equal(prob > 0.5, y.astype(bool))))
        offset += len(y)
    return {name: np.asarray(values) for name, values in metrics.items()}


def patient_means(patients: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average queries within each sorted patient, preserving equal weight between patients."""
    if len(patients) != len(values) or values.ndim != 1 or not len(patients):
        raise ValueError("patient statistics require aligned nonempty query values")
    return np.unique(patients), np.array([np.mean(values[group]) for group in patient_groups(patients)])


def paired_patient_difference(
    patients: np.ndarray, first: np.ndarray, second: np.ndarray, draws: int = DRAWS, seed: int = SEED
) -> dict[str, float | int]:
    """Bootstrap paired differences of patient-level query means with the shared draw convention."""
    ids, first_means = patient_means(patients, first)
    _, second_means = patient_means(patients, second)
    differences = first_means - second_means
    groups = patient_groups(ids)
    rng = np.random.default_rng(seed)
    estimates = [float(np.mean(differences[patient_resample(groups, rng)])) for _ in range(draws)]
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "difference": float(np.mean(differences)),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": draws,
        "skipped_draws": 0,
        "seed": seed,
        "patients": len(ids),
    }


def decision(primary: dict[str, Any], informative_gain: float, lookup_gain: float) -> dict[str, Any]:
    """Apply all four frozen grounding gates without rounding or control selection."""
    gates = {
        "primary_gain_at_least_0.05": primary["difference"] >= 0.05,
        "primary_ci_low_positive": primary["ci_low"] > 0,
        "informative_primary_gain_positive": informative_gain > 0,
        "wave_lookup_gain_nonnegative": lookup_gain >= 0,
    }
    return {
        "outcome": "advance_benchmark_grounding" if all(gates.values()) else "do_not_advance_grounding",
        "gates": gates,
    }


def summarize(
    queries: list[Query], probabilities: dict[str, np.ndarray], informative: np.ndarray, draws: int = DRAWS
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    """Calculate patient-macro endpoints and paired intervals from aligned query probabilities."""
    patients = np.array([q["patient_id"] for q in queries])
    metrics = {name: query_metrics(queries, probabilities[name]) for name in MODELS}
    summaries, saved = {}, {"patient_id": np.unique(patients)}
    for name, arm in metrics.items():
        summaries[name] = {}
        for metric, values in arm.items():
            _, means = patient_means(patients, values)
            summaries[name][metric] = float(np.mean(means))
            saved[f"{name}__{metric}"] = means
        _, means = patient_means(patients[informative], arm["average_precision"][informative])
        summaries[name]["informative_average_precision"] = float(np.mean(means))
    ap = {name: values["average_precision"] for name, values in metrics.items()}
    primary = paired_patient_difference(patients, ap["wave_hgb"], ap["context_hgb"], draws=draws)
    guard = paired_patient_difference(
        patients[informative], ap["wave_hgb"][informative], ap["context_hgb"][informative], draws=draws
    )
    lookup = {
        name: paired_patient_difference(patients, ap[name], ap["context_lookup"], draws=draws)
        for name in ("context_hgb", "wave_hgb")
    }
    contexts = _context_metrics(queries, metrics)
    result = {
        "models": summaries,
        "primary_wave_minus_context": primary,
        "informative_wave_minus_context": guard,
        "contrasts_to_lookup": lookup,
        "contexts": contexts,
        "query_count": len(queries),
        "patient_count": len(np.unique(patients)),
        "informative_query_count": int(informative.sum()),
        "informative_patient_count": len(np.unique(patients[informative])),
        "decision": decision(primary, guard["difference"], lookup["wave_hgb"]["difference"]),
    }
    return result, metrics, saved


def _context_metrics(queries: list[Query], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Report descriptive patient-macro AP within each exact condition/question context."""
    contexts = sorted({q["context"] for q in queries})
    patients = np.array([q["patient_id"] for q in queries])
    result = []
    for context in contexts:
        mask = np.array([q["context"] == context for q in queries])
        result.append(
            {
                "target_dx": context[0],
                "question": context[1],
                "queries": int(mask.sum()),
                "patients": len(np.unique(patients[mask])),
                "aggregation": "patient_macro_within_context",
                "average_precision": {
                    name: float(np.mean(patient_means(patients[mask], values["average_precision"][mask])[1]))
                    for name, values in metrics.items()
                },
            }
        )
    return result


def provenance(protocol: Path) -> dict[str, Any]:
    """Validate public payload identity and bind local inputs, sources and numerical versions."""
    if SOURCE.stat().st_size != SOURCE_BYTES or sha256_file(SOURCE) != SOURCE_SHA:
        raise ValueError("benchmark payload does not match the frozen source")
    digest = hashlib.sha1(f"blob {SOURCE_BYTES}\0".encode())
    with SOURCE.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != BLOB_SHA or sha256_file(METADATA) != METADATA_SHA:
        raise ValueError("source Git blob or official metadata hash changed")
    files = [
        SOURCE,
        METADATA,
        cached.RAW_PATH,
        protocol,
        Path(__file__),
        ROOT / "scripts/experiments/run_lead_grounding043.py",
        ROOT / "ecg_experiment/cpu_screening041.py",
        ROOT / "ecg_experiment/intervals.py",
        ROOT / "ecg_experiment/files.py",
        ROOT / "ecg_experiment/paths.py",
        ROOT / "ecg_experiment/__init__.py",
        ROOT / "ecg_experiment/eda/ptbxl.py",
        ROOT / "ecg_experiment/eda/signals.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    return {
        "files_sha256": {to_stored(path): sha256_file(path) for path in files},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol": to_stored(protocol),
        "annotation_bytes": SOURCE_BYTES,
        "annotation_git_blob": BLOB_SHA,
        "authored_tree": "90a53f107023c7c196cbf08de183b871b2a5798c",
        "corrected_release": "0.0.2",
        "release_tree": "7c84653c8606557657e76b8357e0958163802f26",
        "release_update_commit": "04686428ae78f9733a202ff4c72cc31d0e38af79",
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")
            },
        },
    }


def save_predictions(
    output: Path,
    queries: list[Query],
    probabilities: dict[str, np.ndarray],
    metrics: dict[str, Any],
    informative: np.ndarray,
) -> None:
    """Save aligned query identities/masks and offered-only predictions for independent recomputation."""
    offered = np.array([[lead in q["offered"] for lead in LEADS] for q in queries])
    arrays = {
        "ecg_id": np.array([q["ecg_id"] for q in queries]),
        "patient_id": np.array([q["patient_id"] for q in queries]),
        "strat_fold": np.array([q["fold"] for q in queries]),
        "target_dx": np.array([q["context"][0] for q in queries]),
        "question": np.array([q["context"][1] for q in queries]),
        "offered": offered,
        "target": np.array(
            [[int(lead in q["correct"]) if lead in q["offered"] else -1 for lead in LEADS] for q in queries],
            dtype=np.int8,
        ),
        "informative": informative,
        "lead_order": np.array(LEADS),
    }
    for name, scores in probabilities.items():
        aligned = np.full(offered.shape, np.nan)
        aligned[offered] = scores
        arrays[f"{name}__probability"] = aligned
        arrays.update({f"{name}__{metric}": values for metric, values in metrics[name].items()})
    write_npz_atomic(output / "predictions.npz", **arrays)


def run_experiment(output: Path, protocol: Path, threads: int = 4) -> dict[str, Any]:
    """Execute the two-fit experiment only after frozen qualification and write exclusive receipts."""
    if not 1 <= threads <= 4:
        raise ValueError("threads must be between 1 and 4")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        with threadpool_limits(limits=threads):
            result = _run(output, protocol, threads)
        result["total_seconds"] = time.monotonic() - start
        write_json_atomic(output / "result.json", result)
        if sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > 256 * 1024 * 1024:
            raise ValueError("output exceeds the frozen 256 MiB artifact budget")
        return result
    except Exception as error:
        write_json_atomic(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error).replace(str(ROOT), "."),
                "elapsed_seconds": time.monotonic() - start,
                "test_scoring": False,
            },
        )
        raise


def _run(output: Path, protocol: Path, threads: int) -> dict[str, Any]:
    """Keep the qualification gate ahead of raw-feature loading and both fixed fits."""
    sources = provenance(protocol)
    metadata = pd.read_csv(METADATA, usecols=["ecg_id", "patient_id", "strat_fold"], index_col="ecg_id")
    queries, audit = load_queries(SOURCE, metadata)
    qualification = qualify(queries, audit)
    write_json_atomic(output / "qualification.json", qualification)
    fit = [q for q in queries if q["fold"] <= 7]
    development = [q for q in queries if q["fold"] == 9]
    ids = pd.DataFrame({"ecg_id": sorted({q["ecg_id"] for q in fit + development})})
    raw = cached.raw_measurements(ids)
    raw.index = ids.ecg_id.to_numpy()
    contexts = context_codes(fit)
    train_values, y, weights = expand_queries(fit, contexts, raw)
    dev_values, _, _ = expand_queries(development, contexts, raw)
    missingness = {
        "fit": feature_missingness(fit, train_values, raw),
        "development": feature_missingness(development, dev_values, raw),
        "unused_fold8": "raw features not read or modeled",
    }
    write_json_atomic(output / "missingness.json", missingness)
    models, fit_seconds = fit_models(train_values, y, weights)
    lookup = fit_lookup(fit)
    probabilities, inference_seconds = {}, {}
    for name, columns in (("context_hgb", SHARED_FEATURES), ("wave_hgb", WAVE_FEATURES)):
        start = time.monotonic()
        probabilities[name] = models[name].predict_proba(dev_values[list(columns)])[:, 1]
        inference_seconds[name] = time.monotonic() - start
    start = time.monotonic()
    probabilities["context_lookup"] = np.array(
        [lookup_probability(lookup, q, lead) for q in development for lead in q["offered"]]
    )
    inference_seconds["context_lookup"] = time.monotonic() - start
    training_informative = informative_groups(fit)
    informative = np.array([(q["context"], q["offered"]) in training_informative for q in development])
    summary, metrics, means = summarize(development, probabilities, informative)
    joblib.dump(
        {
            "models": models,
            "contexts": contexts,
            "lookup": lookup,
            "informative_groups": training_informative,
            "lead_order": LEADS,
            "feature_columns": {"context_hgb": SHARED_FEATURES, "wave_hgb": WAVE_FEATURES},
        },
        output / "fitted_models.joblib",
    )
    save_predictions(output, development, probabilities, metrics, informative)
    write_npz_atomic(output / "patient_means.npz", **means)
    artifacts = (
        "qualification.json",
        "missingness.json",
        "fitted_models.joblib",
        "predictions.npz",
        "patient_means.npz",
    )
    return {
        "status": "completed",
        "experiment": "043",
        "provenance": sources,
        "qualification": qualification,
        "missingness": missingness,
        **summary,
        "config": {
            "threads": threads,
            "model": CONFIG,
            "categorical_features": ["context_code", "lead_code"],
            "feature_columns": {"context_hgb": SHARED_FEATURES, "wave_hgb": WAVE_FEATURES},
            "raw_feature_columns": cached.RAW_FEATURES,
            "candidate_measurements": MEASUREMENTS,
            "query_weight": "1/number_of_offered_leads",
            "lookup_beta_prior": [1, 1],
            "lookup_shrinkage": 10,
            "bootstrap_draws": DRAWS,
            "seed": SEED,
            "exact_set_comparator": ">",
            "exact_set_threshold": 0.5,
            "fit_folds": list(range(1, 8)),
            "unused_fold": 8,
            "development_fold": 9,
            "test_scoring": False,
            "nonfinite_policy": "NaN; no imputation or clipping",
        },
        "timing": {"fit_seconds": fit_seconds, "inference_seconds": inference_seconds},
        "model_fit_status": {
            name: {"iterations": int(model.n_iter_), "expected_iterations": 200}
            for name, model in models.items()
        },
        "artifact_sha256": {to_stored(output / name): sha256_file(output / name) for name in artifacts},
        "artifacts": {name: to_stored(output / name) for name in (*artifacts, "result.json")},
    }
