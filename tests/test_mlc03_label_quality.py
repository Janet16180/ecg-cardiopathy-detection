"""Tests of the MLC-03 label-quality code: measurements, source comparison, confident learning, role guard."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import mlc03_label_quality as lq
from ecg_experiment import mlc03_measurements as mm
from ecg_experiment.mlc02_data import CONDITIONS, MANIFEST


def measured(**settings: float) -> tuple[dict, dict]:
    signal, truth = mm.synthetic_record(seed=3, **settings)
    return mm.measure_record(signal), truth


def test_heart_rate_qrs_and_pr_on_a_clean_synthetic_record():
    result, truth = measured(rate_bpm=72, pr_ms=180, qrs_ms=96)
    assert result["status"] == "ok"
    assert abs(result["hr_bpm"] - 72) < 1
    assert abs(result["qrs_ms"] - 96) <= 20
    assert abs(result["pr_ms"] - 180) <= 25


def test_qrs_differences_are_preserved():
    narrow, _ = measured(qrs_ms=90)
    wide, _ = measured(qrs_ms=140)
    assert abs((wide["qrs_ms"] - narrow["qrs_ms"]) - 50) <= 10


def test_long_pr_is_measured_long():
    result, _ = measured(pr_ms=280, rate_bpm=60)
    assert result["pr_ms"] > 240


def test_frontal_axis_follows_the_heart_vector():
    for axis in (-60.0, 0.0, 60.0, 110.0):
        result, truth = measured(axis_deg=axis)
        assert abs(lq.angle_difference(result["axis_deg"], truth["axis_net_deg"])) < 10


def test_terminal_right_anterior_delay_gives_rsr_prime_in_v1():
    result, _ = measured(qrs_ms=140, rbbb=1)
    plain, _ = measured(qrs_ms=90)
    assert result["v1_positive_peaks"] >= 2
    assert result["v1_terminal_mv"] > 0
    assert result["term_I"] < 0
    assert plain["v1_positive_peaks"] <= 1
    assert plain["v1_terminal_mv"] < 0
    assert result["qrs_ms"] >= 130


def test_atrial_fibrillation_has_no_p_wave_and_irregular_rr():
    af, _ = measured(af=1, rate_bpm=90)
    sinus, _ = measured(rate_bpm=90)
    assert af["p_snr"] < lq.P_SNR_MIN <= sinus["p_snr"]
    assert af["rr_nrmssd"] >= lq.IRREGULAR_NRMSSD > sinus["rr_nrmssd"]


def test_premature_wide_beats_are_counted():
    ectopy, truth = measured(pvc_every=4)
    plain, _ = measured()
    assert ectopy["n_premature_aberrant"] >= 1
    assert plain["n_premature_aberrant"] == 0
    assert truth["ectopic"].sum() >= 2


def test_st_depression_and_t_inversion_levels():
    depressed, truth = measured(st_mv=-0.2)
    levels = mm.synthetic_levels(truth)
    assert abs(depressed["st60_II"] - levels["st60_II"]) < 0.05
    inverted, _ = measured(t_mv=-0.3)
    assert inverted["t_II"] < -0.15
    assert inverted["t_V5"] < -0.1


def test_q_wave_of_a_negative_qrs_spans_the_complex():
    result, _ = measured(axis_deg=110)
    assert result["qdur_I"] >= 60
    assert result["r_I"] < 0.2


def test_flat_record_is_reported_not_measured():
    result = mm.measure_record(np.zeros((12, 5000)))
    assert result["status"] == "flat"


def test_synthetic_limb_leads_obey_einthoven():
    signal, _ = mm.synthetic_record(seed=1, axis_deg=30)
    np.testing.assert_allclose(signal[2], signal[1] - signal[0], atol=1e-9)
    np.testing.assert_allclose(signal[5], signal[1] - signal[0] / 2, atol=1e-9)


def test_q_wave_rules():
    assert mm.q_wave(np.array([0.0, 0.1, 0.5, 0.2, -0.1])) == (0.0, 0.0)
    duration, depth = mm.q_wave(np.array([0.0, -0.05, -0.2, -0.1, 0.3, 0.5] + [0.4] * 10))
    assert duration == 8.0
    assert depth == pytest.approx(0.2)


def test_consensus_keeps_method_b_where_both_agree():
    table = pd.DataFrame(
        {
            "hr_a": [60.0, 60.0, np.nan],
            "hr_b": [62.0, 80.0, 70.0],
            "qrs_a": [100.0, 100.0, 100.0],
            "qrs_b": [110.0, 130.0, np.nan],
            "pr_a": [150.0, np.nan, 150.0],
            "pr_b": [160.0, 150.0, 200.0],
            "axis_a": [175.0, 0.0, 0.0],
            "axis_b": [-175.0, 30.0, 10.0],
        }
    )
    out = lq.consensus(table)
    assert out["hr"].tolist()[0] == 62.0
    assert np.isnan(out["hr"].iloc[1])
    assert np.isnan(out["hr"].iloc[2])
    assert out["axis_agree"].tolist() == [True, False, True]
    assert out["qrs_both"].tolist() == [True, True, False]


def test_region_masks_with_tolerance():
    value = pd.Series([95.0, 103.0, 120.0, np.nan])
    masks = lq.region_masks(value, 100.0, np.inf, 5.0)
    assert masks["outside"].tolist() == [True, False, False, False]
    assert masks["outside_tol"].tolist() == [False, False, False, False]
    assert masks["inside_tol"].tolist() == [False, False, True, False]


def prevalence_table(rates: dict[str, float], n: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for source, rate in rates.items():
        labels = np.zeros(n)
        labels[: int(round(rate * n))] = 1
        for label in rng.permutation(labels):
            rows.append({"source": source, "label": label, "value": 50.0})
    return pd.DataFrame(rows)


def test_matched_prevalence_flags_a_two_fold_difference_with_disjoint_intervals(monkeypatch):
    table = prevalence_table({"ptbxl": 0.6, "ningbo": 0.2, "georgia": 0.55}, 400)
    table["status_b"] = "ok"
    monkeypatch.setitem(
        lq.MATCHED, "toy", {"label": "label", "measure": "value", "population": "measured", "bins": (40, 60)}
    )
    prevalence = lq.matched_prevalence(table, "toy")
    pairs = lq.inconsistent_bins(prevalence).set_index(["higher", "lower"])
    assert pairs.loc[("ptbxl", "ningbo"), "inconsistent"]
    assert pairs.loc[("georgia", "ningbo"), "inconsistent"]
    assert not pairs.loc[("ptbxl", "georgia"), "inconsistent"]


def test_matched_prevalence_needs_disjoint_intervals_and_enough_records(monkeypatch):
    table = prevalence_table({"ptbxl": 0.2, "ningbo": 0.08}, 40)
    table["status_b"] = "ok"
    monkeypatch.setitem(
        lq.MATCHED, "toy", {"label": "label", "measure": "value", "population": "measured", "bins": (40, 60)}
    )
    pairs = lq.inconsistent_bins(lq.matched_prevalence(table, "toy"))
    assert pairs["ratio"].iloc[0] >= 2
    assert not pairs["inconsistent"].iloc[0]
    assert lq.inconsistent_bins(lq.matched_prevalence(table, "toy"), minimum=41).empty


def test_confident_learning_recovers_planted_label_noise():
    rng = np.random.default_rng(5)
    truth = (rng.random(20000) < 0.2).astype(int)
    score = np.where(truth == 1, rng.normal(3.0, 1.0, truth.size), rng.normal(-3.0, 1.0, truth.size))
    probability = 1 / (1 + np.exp(-score))
    given = truth.copy()
    flip_positive = rng.choice(np.flatnonzero(truth == 0), 400, replace=False)
    flip_negative = rng.choice(np.flatnonzero(truth == 1), 200, replace=False)
    given[flip_positive], given[flip_negative] = 1, 0
    result = lq.confident_learning(given, probability)
    total = result["estimated_false_positives"] + result["estimated_false_negatives"]
    assert total == pytest.approx(600, rel=0.15)
    assert result["estimated_false_positives"] == pytest.approx(400, rel=0.3)
    assert result["estimated_false_negatives"] == pytest.approx(200, rel=0.3)
    assert np.isin(np.flatnonzero(result["flag_false_positive"]), flip_positive).mean() > 0.95
    assert np.isin(np.flatnonzero(result["flag_false_negative"]), flip_negative).mean() > 0.95


def test_confident_learning_finds_little_without_noise():
    rng = np.random.default_rng(6)
    truth = (rng.random(5000) < 0.3).astype(int)
    probability = 1 / (
        1 + np.exp(-np.where(truth == 1, rng.normal(3, 1, truth.size), rng.normal(-3, 1, truth.size)))
    )
    result = lq.confident_learning(truth, probability)
    assert result["estimated_false_positives"] + result["estimated_false_negatives"] < 25


def test_suspects_rank_by_self_confidence():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.5, 0.2, 0.8, 0.05])
    ranked = lq.suspects(y, p, count=2)
    assert ranked["false_positive"].tolist() == [1, 2]
    assert ranked["false_negative"].tolist() == [4, 3]


def test_reading_list_takes_sources_in_turn_by_rank():
    rows = [
        (kind, source, rank)
        for kind in ("false_positive", "false_negative")
        for source in ("georgia", "ptbxl", "ningbo")
        for rank in range(1, 6)
    ]
    listed = pd.DataFrame(rows, columns=["kind", "source", "rank"]).assign(condition="lvh")
    listed["record_key"] = listed["source"] + ":" + listed["rank"].astype(str)
    reading = lq.reading_list(listed, per_kind=4)
    first = reading[reading["kind"] == "false_positive"]
    assert first["record_key"].tolist() == ["ptbxl:1", "ningbo:1", "georgia:1", "ptbxl:2"]
    assert reading.groupby("kind").size().tolist() == [4, 4]


def test_rhythm_statement_reads_both_vocabularies():
    table = pd.DataFrame(
        {
            "source": ["ptbxl", "ptbxl", "georgia", "georgia", "cpsc_2018"],
            "raw_codes": [
                "{'NORM': 100.0, 'SR': 0.0}",
                "{'IMI': 50.0}",
                "59118001,426783006",
                "59118001",
                "164889003",
            ],
        }
    )
    assert lq.rhythm_stated(table).tolist() == [True, False, True, False, True]


def test_group_folds_keep_groups_together_and_balance_sources():
    groups = np.array([f"p{i // 3}" for i in range(300)] + [f"r{i}" for i in range(200)])
    sources = np.array(["ptbxl"] * 300 + ["georgia"] * 200)
    folds = lq.group_folds(groups, sources)
    assert pd.Series(folds).groupby(groups).nunique().max() == 1
    assert np.bincount(folds[sources == "georgia"]).tolist() == [40] * 5


@pytest.mark.parametrize("role", ["test_same_source", "test_external", "sph_calibration"])
def test_open_roles_refuse_test_roles(role):
    with pytest.raises(PermissionError):
        lq.check_open_roles(("train", role))
    with pytest.raises(PermissionError):
        lq.load_open_records(roles=("validation", role))


def test_ptbxl_plus_loader_drops_fold_ten(tmp_path: Path):
    pd.DataFrame({"ecg_id": [1, 2, 3], "strat_fold": [1, 9, 10]}).to_csv(
        tmp_path / "database.csv", index=False
    )
    for program, columns in lq.PLUS_COLUMNS.items():
        table = pd.DataFrame({"ecg_id": [1, 2, 3], **{column: [1.0, 2.0, 3.0] for column in columns}})
        table.to_csv(tmp_path / f"{program}_features.csv", index=False)
    keys = np.array(["ptbxl:1", "ptbxl:2", "ptbxl:3", "georgia:E00001"])
    loaded = lq.load_ptbxl_plus(keys, directory=tmp_path, database=tmp_path / "database.csv")
    assert sorted(loaded["record_key"]) == ["ptbxl:1", "ptbxl:2"]
    assert 3.0 not in loaded.drop(columns="record_key").to_numpy()


def test_loader_never_parses_closed_rows(tmp_path: Path):
    columns = [
        *lq.STRUCTURE,
        *[f"b_{c}" for c in CONDITIONS],
        *[f"c_{c}" for c in lq.EXTRA_LABELS],
        *[f"flat_{lead}" for lead in mm.LEADS],
    ]
    rows = []
    for key, role, value in (
        ("a", "train", "1"),
        ("b", "test_external", "not-a-number"),
        ("c", "validation", ""),
        ("d", "test_same_source", "oops"),
    ):
        row = dict.fromkeys(columns, "")
        row |= {
            "record_key": key,
            "role": role,
            "included": "True",
            "source": "ptbxl",
            "raw_codes": "{}",
            "b_lvh": value,
            "age": "50",
            "duration_s": "10",
            "n_flat_leads": "0",
            "n_missing_leads": "0",
        }
        row |= {f"flat_{lead}": "False" for lead in mm.LEADS} | {"flat_V1": "True" if key == "a" else "False"}
        rows.append(row)
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    frame = lq.load_open_records(path)
    assert frame["record_key"].tolist() == ["a", "c"]
    assert frame["flat_V1"].tolist() == [True, False]
    assert frame["b_lvh"].iloc[0] == 1.0
    assert np.isnan(frame["b_lvh"].iloc[1])


def test_undefined_reasons_follow_the_label_rules():
    none = frozenset()
    assert lq.undefined_reason("sinus_tachycardia", "cpsc_2018", frozenset({"59118001"}), none).startswith(
        "not annotated"
    )
    ptb = lq.record_codes("ptbxl", "{'SR': 0.0}", "LAD")
    reason = lq.undefined_reason("left_axis_deviation", "ptbxl", ptb, none)
    assert reason == "ambiguous: PTB-XL heart_axis LAD"
    both = frozenset({"713427006", "713426002"})
    assert (
        lq.undefined_reason("complete_rbbb", "georgia", both, none) == "contradictory pair: incomplete_rbbb"
    )
    flagged = frozenset({("georgia", "b_mi_any")})
    assert lq.undefined_reason("mi_any", "georgia", frozenset({"426783006"}), flagged) == "minimum positives"


@pytest.mark.skipif(not MANIFEST.exists(), reason="the MLC-01 v4 manifest is local only")
def test_real_manifest_loads_train_and_validation_only():
    frame = lq.load_open_records()
    assert set(frame["role"]) == {"train", "validation"}
    assert len(frame) == 71388


def test_premature_wide_beat_at_the_record_edge_is_counted():
    signal, truth = mm.synthetic_record(seed=3, pvc_every=4)
    ectopic = truth["onsets"][truth["ectopic"]]
    onset = int(ectopic[ectopic > 3000][0])
    result = mm.measure_record(signal[:, : onset + mm.samples(350)])
    assert result["n_premature_aberrant_edge"] >= 1
