import numpy as np
import pandas as pd
import pytest

from ecg_experiment import normative_map062 as nm
from ecg_experiment import replacement070 as rp
from ecg_experiment.localization import nbr01_analysis as na
from ecg_experiment.localization import nbr01_counterpart as nc
from ecg_experiment.localization import nbr01_selection as ns


def synthetic_knots(beats: int = 6) -> np.ndarray:
    starts = 300.0 + 800.0 * np.arange(beats)
    return np.stack([starts + offset for offset in (0.0, 150.0, 200.0, 240.0, 330.0, 380.0)], axis=1)


def test_beat_timeline_matches_070_with_one_template():
    rng = np.random.default_rng(0)
    template = rng.normal(size=(12, nm.BINS))
    knots = synthetic_knots()
    expected = rp.template_timeline(template, knots)
    found, weight = nc.beat_timeline(np.repeat(template[None], len(knots), axis=0), knots)
    assert np.array_equal(found, expected)
    assert weight.max() == pytest.approx(1.0)


def test_variants_are_fixed():
    items = nc.variants()
    assert len(items) == 105
    assert len({item.name for item in items}) == 105
    assert items[0].source == "none"
    assert sum(item.source == "swap" for item in items) == 52


def test_variant_signals_keep_original_and_bound_change():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(12, 5000))
    components = np.zeros((3, 5000), bool)
    components[0, 1000:1050] = components[1, 1100:1150] = components[2, 1200:1400] = True
    plan = {"difference": rng.normal(size=(12, 5000)).astype(np.float32),
            "swap": rng.normal(size=(12, 5000)).astype(np.float32), "components": components}
    items = nc.variants()
    signals, energy = nc.variant_signals(raw, plan, items)
    assert np.array_equal(signals[0], raw)
    assert energy[0] == 0.0
    change = signals[1:] - raw
    lead_6_qrs = [item.name for item in items].index("grid_6_1")
    touched = np.flatnonzero(np.abs(change[lead_6_qrs - 1]).sum(axis=1))
    assert touched.tolist() == [6]
    full = [item.name for item in items].index("full")
    bound = np.abs(plan["difference"].astype(float)) + 1e-12
    assert np.all(np.abs(signals[full] - raw) <= bound)
    assert energy[full] >= energy[lead_6_qrs]


def test_shares_and_territory_contrast():
    values = np.array([[0.0, -1.0], [2.0, 2.0]])
    assert np.allclose(na.shares(values), [[0.5, 0.5], [0.5, 0.5]])
    groups = np.array(["inferior", "inferior", "anterior", "anterior", "stt"])
    lead_shares = np.zeros((5, 12))
    lead_shares[:2, [1, 2, 5]] = 1 / 3
    lead_shares[2:4, [6, 7, 8, 9]] = 1 / 4
    lead_shares[4] = 1 / 12
    assert na.territory_contrast(lead_shares, groups) == pytest.approx(1.0)
    hits = na.top_hits(lead_shares, groups, 1)
    assert hits[:4].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert np.isnan(hits[4])
    assert np.isnan(na.territory_contrast(lead_shares[:2], groups[:2]))


def test_wave_contrast_sign():
    groups = np.array(["rbbb", "lbbb", "stt", "stt", "inferior"])
    component = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]])
    result = na.wave_contrasts(component, groups)
    assert result["qrs_bbb"] == pytest.approx(1 - (1 / 3) / 3)
    assert result["stt_stt"] == pytest.approx(1 - (1 / 3) / 3)


def test_group_mapping():
    assert ns.group_of({"161"}, 1) == "inferior"
    assert ns.group_of({"165", "146"}, 1) == "anterior"
    assert ns.group_of({"161", "165"}, 1) is None
    assert ns.group_of({"166"}, 1) is None
    assert ns.group_of({"104", "146"}, 1) == "lbbb"
    assert ns.group_of({"106", "145"}, 1) == "rbbb"
    assert ns.group_of({"145", "22"}, 1) == "stt"
    assert ns.group_of({"145", "105"}, 1) is None
    assert ns.group_of({"1"}, 0) == "normal"
    assert ns.statement_codes("22;145+362") == {"22", "145"}


def test_draw_sample_one_per_patient():
    rows = pd.DataFrame({"ecg_id": [f"A{i:03d}" for i in range(12)],
                         "patient_key": ["p1", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p1",
                                         "p10"],
                         "group": ["rbbb"] * 6 + ["normal"] * 6})
    first = ns.draw_sample(rows, 3, 2)
    assert first.equals(ns.draw_sample(rows, 3, 2))
    assert first["patient_key"].is_unique
    assert first["group"].value_counts().to_dict() == {"rbbb": 3, "normal": 2}


def test_bootstrap_runs_on_synthetic_scores():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.array(["inferior", "anterior", "lbbb", "rbbb", "stt", "normal"]), 6)
    scores = rng.uniform(size=(len(groups), 105, 2, 3))
    energy = rng.uniform(size=(len(groups), 105))
    thresholds = {f"{model}_{kind}": 0.5 for model in ("all_age", "ptb_baseline")
                  for kind in ("hgb", "xecg", "blend")}
    table = na.record_table(scores, energy, thresholds)
    result = na.bootstrap(table, groups, np.arange(len(groups)).astype(str), draws=20, seed=0)
    primary = result["all_age_blend/territory"]
    assert primary["low"] <= primary["point"] <= primary["high"] or primary["valid_draws"] < 20
    paired = result["all_age_blend/territory_minus_energy"]["point"]
    assert paired == pytest.approx(primary["point"] - result["energy/territory"]["point"])
    assert 0.0 <= result["all_age_blend/abnormal/grid_flip"]["point"] <= 1.0
