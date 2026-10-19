"""Synthetic scientific checks for conditional grounding and the two frozen CPU paths."""

import json
import time

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from ecg_experiment import lead_grounding043 as grounding


def query(ecg=1, patient=1, fold=1, context=("condition", "question"), offered=("I", "II"), correct=("I",)):
    return {
        "ecg_id": ecg,
        "patient_id": patient,
        "fold": fold,
        "context": context,
        "offered": offered,
        "correct": correct,
    }


def item(options=None, answer=None, indices=None, question="question"):
    return {
        "question_type": "lead_grounding",
        "question": question,
        "options": options if options is not None else ["Lead II", "Lead I", "Lead V1"],
        "answer": answer if answer is not None else ["Lead I"],
        "answer_idx": indices if indices is not None else [1],
    }


def payload(ecg, items):
    return {
        "metadata": {"ecg_id": str(ecg), "target_dx": "condition", "dx_label": False, "path_idx": 99},
        "data": {"reasoning": [{"ecg_grounding": items, "answer": "forbidden prior answer"}]},
    }


def test_answer_index_semantics_option_invariance_and_multiselect():
    first = grounding.parse_grounding(item())
    permuted = grounding.parse_grounding(item(["Lead V1", "Lead I", "Lead II"], ["Lead I"], [1]))
    assert first == permuted == (("I", "II", "V1"), ("I",))
    multi = grounding.parse_grounding(item(answer=["Lead V1", "Lead I"], indices=[2, 1]))
    assert multi == (("I", "II", "V1"), ("I", "V1"))


@pytest.mark.parametrize(
    ("options", "answer", "indices", "match"),
    [
        (["I", "Lead II"], ["I"], [0], "literal"),
        (["Lead I", "Lead I"], ["Lead I"], [0], "literal"),
        (["Lead I", "Lead II"], ["Lead II"], [0], "disagree"),
        (["Lead I", "Lead II"], ["Lead I"], [2], "index"),
        (["Lead I", "Lead II"], ["Lead I"], [True], "index"),
        (["Lead I", "Lead II"], [], [], "empty or all-positive"),
        (["Lead I", "Lead II"], ["Lead I", "Lead II"], [0, 1], "empty or all-positive"),
    ],
)
def test_unsupported_answers_stop(options, answer, indices, match):
    with pytest.raises(ValueError, match=match):
        grounding.parse_grounding(item(options, answer, indices))


def test_parser_filters_test_before_steps_preserves_questions_and_deduplicates(tmp_path):
    metadata = pd.DataFrame({"patient_id": [101, 102], "strat_fold": [1, 10]}, index=[1, 2])
    test = {"metadata": {"ecg_id": "2"}, "data": "malformed test steps must never be read"}
    rows = [test, payload(1, [item(), item(question="different exact question")]), payload(1, [item()])]
    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    queries, audit = grounding.load_queries(path, metadata)
    assert len(queries) == 2
    assert audit["duplicate_queries"] == 1
    assert audit["fold10_rows_skipped_before_steps"] == 1
    assert {q["context"][1] for q in queries} == {"question", "different exact question"}
    path.write_text(json.dumps(payload(1, [item(), item(answer=["Lead II"], indices=[0])])))
    with pytest.raises(ValueError, match="conflicting"):
        grounding.load_queries(path, metadata)


def test_missing_join_and_patient_fold_overlap_stop(tmp_path):
    metadata = pd.DataFrame({"patient_id": [101, 101], "strat_fold": [1, 9]}, index=[1, 2])
    with pytest.raises(ValueError, match="crosses folds"):
        grounding.validate_metadata(metadata)
    metadata.loc[2, "patient_id"] = 102
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(payload(99, [item()])))
    with pytest.raises(ValueError, match="absent"):
        grounding.load_queries(path, metadata)


def test_context_codes_and_informative_groups_are_fit_only():
    fit = [query(), query(2, 2, correct=("II",))]
    unused = query(3, 3, 8, context=("unused", "question"))
    dev = query(4, 4, 9, context=("unseen", "question"))
    contexts = grounding.context_codes(fit + [unused, dev])
    assert contexts == {("condition", "question"): 0}
    values, _, _ = grounding.expand_queries([dev], contexts)
    assert values.context_code.isna().all()
    groups = grounding.informative_groups(fit + [unused, dev])
    assert groups == {(("condition", "question"), ("I", "II"))}
    constant_fit = [query(), query(2, 2)]
    changed_dev = query(3, 3, 9, correct=("II",))
    assert grounding.informative_groups(constant_fit + [changed_dev]) == set()
    different_offer = query(4, 4, offered=("I", "II", "III"), correct=("II",))
    assert grounding.informative_groups([query(), different_offer]) == set()


def synthetic_raw(ids):
    return pd.DataFrame(
        np.arange(len(ids) * 130).reshape(len(ids), 130).astype(float),
        index=ids,
        columns=grounding.cached.RAW_FEATURES,
    )


def test_offered_only_expansion_exact_candidate_features_and_equal_query_weights():
    queries = [query(), query(2, 2, offered=("I", "II", "III"), correct=("III",))]
    raw = synthetic_raw([1, 2])
    values, y, weights = grounding.expand_queries(queries, grounding.context_codes(queries), raw)
    assert values.columns.to_list() == list(grounding.WAVE_FEATURES)
    assert len(values) == 5
    assert weights[:2].sum() == pytest.approx(1)
    assert weights[2:].sum() == pytest.approx(1)
    assert y.tolist() == [1, 0, 0, 0, 1]
    assert values.lead_code.tolist() == [0, 1, 0, 1, 2]
    assert values.loc[1, "candidate__st"] == raw.at[1, "st_II"]
    assert values.loc[4, "candidate__q_depth"] == raw.at[2, "q_depth_III"]
    assert set(values.columns).isdisjoint(
        {"ecg_id", "patient_id", "dx_label", "path_idx", "answer", "age", "sex"}
    )
    raw.at[1, "st_II"] = np.inf
    nonfinite, _, _ = grounding.expand_queries(queries, grounding.context_codes(queries), raw)
    assert np.isnan(nonfinite.loc[1, "candidate__st"])


def test_lookup_weighted_counts_and_exact_hierarchy_fallbacks():
    queries = [query(), query(2, 2, offered=("I", "II", "III"), correct=("II",))]
    state = grounding.fit_lookup(queries)
    np.testing.assert_allclose(state["global_counts"]["I"], [0.5, 0.5 + 1 / 3])
    global_i = (0.5 + 1) / (0.5 + 1 / 3 + 2)
    context_i = (0.5 + 10 * global_i) / (0.5 + 1 / 3 + 10)
    exact_i = (0.5 + 10 * context_i) / (0.5 + 10)
    assert grounding.lookup_probability(state, queries[0], "I") == pytest.approx(exact_i)
    new_offer = query(offered=("I", "V1"))
    assert grounding.lookup_probability(state, new_offer, "I") == pytest.approx(context_i)
    new_context = query(context=("new", "question"))
    assert grounding.lookup_probability(state, new_context, "I") == pytest.approx(global_i)
    assert grounding.lookup_probability(state, queries[0], "V6") == 0.5
    with pytest.raises(ValueError, match="only official fitting folds"):
        grounding.fit_lookup([query(fold=8)])


def test_ap_ties_exact_strict_threshold_and_patient_averaging():
    queries = [query(), query(2, 1, offered=("I", "II", "III"), correct=("I",)), query(3, 2)]
    scores = np.array([0.5, 0.5, 0.9, 0.1, 0.1, 0.1, 0.9])
    metrics = grounding.query_metrics(queries, scores)
    np.testing.assert_allclose(metrics["average_precision"], [0.5, 1, 0.5])
    assert metrics["exact_set_accuracy"].tolist() == [0, 1, 0]
    ids, means = grounding.patient_means(np.array([1, 1, 2]), metrics["average_precision"])
    assert ids.tolist() == [1, 2]
    np.testing.assert_allclose(means, [0.75, 0.5])
    assert means.mean() == 0.625
    assert metrics["average_precision"].mean() != means.mean()
    with pytest.raises(ValueError, match="align exactly"):
        grounding.query_metrics(queries, np.r_[scores, 0.2])


def test_bootstrap_uses_patient_level_means_and_shared_draws():
    patients = np.array([1, 1, 2])
    first, second = np.array([0.5, 1, 0.5]), np.array([0.25, 0.25, 0.25])
    result = grounding.paired_patient_difference(patients, first, second, draws=100)
    assert result["difference"] == 0.375
    assert result["patients"] == 2
    rng = np.random.default_rng(grounding.SEED)
    observed = [np.mean(np.array([0.5, 0.25])[rng.integers(0, 2, 2)]) for _ in range(100)]
    low, high = np.percentile(observed, [2.5, 97.5])
    assert result["ci_low"] == low
    assert result["ci_high"] == high
    assert result["skipped_draws"] == 0


@pytest.mark.parametrize(
    ("gain", "low", "informative", "lookup", "advance"),
    [
        (0.05, 1e-12, 1e-12, 0, True),
        (0.049999, 0.01, 0.01, 0.01, False),
        (0.05, 0, 0.01, 0.01, False),
        (0.05, 0.01, 0, 0.01, False),
        (0.05, 0.01, 0.01, -1e-12, False),
    ],
)
def test_each_decision_gate(gain, low, informative, lookup, advance):
    result = grounding.decision({"difference": gain, "ci_low": low}, informative, lookup)
    assert (result["outcome"] == "advance_benchmark_grounding") == advance


def test_prefit_frozen_qualification_counts_and_role_overlap():
    queries = [query(), query(2, 2, 9)]
    audit = dict.fromkeys(
        (
            "duplicate_queries",
            "conflicting_queries",
            "missing_joins",
            "empty_targets",
            "all_positive_targets",
        ),
        0,
    )
    qualification = grounding.qualify(queries, audit, require_expected=False)
    assert qualification["roles"]["fit"]["queries"] == 1
    with pytest.raises(ValueError, match="frozen qualification"):
        grounding.qualify(queries, audit)
    with pytest.raises(ValueError, match="cross roles"):
        grounding.qualify([query(), query(2, 1, 9)], audit, require_expected=False)


def test_actual_fixed_artificial_fit_paths_and_serialization(tmp_path, record_property):
    rng = np.random.default_rng(73)
    queries = [
        query(
            i,
            i,
            context=("condition", f"question{i % 2}"),
            offered=grounding.LEADS,
            correct=(grounding.LEADS[i % 12],),
        )
        for i in range(1, 85)
    ]
    raw = pd.DataFrame(rng.normal(size=(84, 130)), index=range(1, 85), columns=grounding.cached.RAW_FEATURES)
    raw.iloc[0, 0] = np.nan
    contexts = grounding.context_codes(queries)
    values, y, weights = grounding.expand_queries(queries, contexts, raw)
    with threadpool_limits(limits=2):
        models, timings = grounding.fit_models(values, y, weights)
        record_property("artificial_fit_seconds", json.dumps(timings))
        path = tmp_path / "models.joblib"
        joblib.dump(models, path)
        reloaded = joblib.load(path)
        for name, columns in (
            ("context_hgb", grounding.SHARED_FEATURES),
            ("wave_hgb", grounding.WAVE_FEATURES),
        ):
            start = time.monotonic()
            expected = models[name].predict_proba(values[list(columns)])[:, 1]
            record_property(f"{name}_artificial_inference_seconds", time.monotonic() - start)
            assert models[name].n_iter_ == 200
            assert models[name].categorical_features == ["context_code", "lead_code"]
            np.testing.assert_array_equal(expected, reloaded[name].predict_proba(values[list(columns)])[:, 1])
            assert np.isfinite(expected).all()
        values["raw__std_I"] = np.nan
        with pytest.raises(ValueError, match="entirely missing"):
            grounding.fit_models(values, y, weights)


def test_saved_predictions_recompute_all_stats_without_new_inference(tmp_path):
    queries = [query(fold=9), query(2, 1, 9, correct=("II",)), query(3, 2, 9)]
    probabilities = {
        "context_hgb": np.array([0.5] * 6),
        "wave_hgb": np.array([0.9, 0.1, 0.1, 0.9, 0.8, 0.2]),
        "context_lookup": np.array([0.6, 0.4] * 3),
    }
    informative = np.array([True, True, False])
    result, metrics, patients = grounding.summarize(queries, probabilities, informative, draws=50)
    grounding.save_predictions(tmp_path, queries, probabilities, metrics, informative)
    saved = np.load(tmp_path / "predictions.npz")
    assert np.isnan(saved["wave_hgb__probability"][~saved["offered"]]).all()
    assert np.all(saved["target"][~saved["offered"]] == -1)
    assert saved["target"].dtype == np.int8
    assert set(saved["target"][saved["offered"]]) == {0, 1}
    assert result["models"]["wave_hgb"]["average_precision"] == 1
    assert result["contexts"][0]["aggregation"] == "patient_macro_within_context"
    assert patients["patient_id"].tolist() == [1, 2]
    recomputed = grounding.query_metrics(queries, saved["wave_hgb__probability"][saved["offered"]])
    np.testing.assert_array_equal(recomputed["average_precision"], saved["wave_hgb__average_precision"])


def test_qualification_failure_prevents_fits_and_keeps_exclusive_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(grounding, "provenance", lambda _: {})
    monkeypatch.setattr(grounding.pd, "read_csv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grounding, "load_queries", lambda *_: ([query()], {}))

    def fail_qualification(*_):
        raise ValueError("artificial qualification failure")

    def forbidden_fit(*_):
        pytest.fail("qualification failure reached model fitting")

    monkeypatch.setattr(grounding, "qualify", fail_qualification)
    monkeypatch.setattr(grounding, "fit_models", forbidden_fit)
    output = tmp_path / "exclusive"
    with pytest.raises(ValueError, match="artificial qualification"):
        grounding.run_experiment(output, tmp_path / "protocol", threads=1)
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        grounding.run_experiment(output, tmp_path / "protocol", threads=1)


def test_missingness_uses_unique_ecgs_and_expanded_candidate_denominators():
    queries = [query(), query(context=("condition", "second question"))]
    raw = synthetic_raw([1])
    raw.at[1, "st_I"] = np.nan
    values, _, _ = grounding.expand_queries(queries, grounding.context_codes(queries), raw)
    counts = grounding.feature_missingness(queries, values, raw)
    assert counts["raw_unique_ecg_count"] == 1
    assert counts["raw_missing_counts_per_unique_ecg"]["st_I"] == 1
    assert counts["expanded_offered_row_count"] == 4
    assert counts["candidate_missing_counts_per_expanded_row"]["candidate__st"] == 2
