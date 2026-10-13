import numpy as np
import pandas as pd
import pytest

from ecg_experiment import criteria069 as cr
from ecg_experiment.waveforms import LEADS


def test_qrs_measures_q_wave_and_r_s():
    q = np.concatenate([-np.linspace(0, 0.3, 10), np.linspace(-0.3, 1.0, 10), np.linspace(1.0, -0.4, 15)])
    m = cr.qrs_measures(q)
    first_positive = int(np.flatnonzero(q >= 0)[1])
    assert m["Q_DUR"] == pytest.approx(cr.MS * first_positive)
    assert m["Q_DEPTH"] == pytest.approx(0.3)
    assert m["R"] == pytest.approx(1.0)
    assert m["S"] == pytest.approx(0.4)
    assert m["PP"] == pytest.approx(1.4)


def test_qrs_measures_qs_complex_and_no_q():
    qs = -np.sin(np.linspace(0, np.pi, 40))
    m = cr.qrs_measures(qs)
    assert m["Q_DUR"] == pytest.approx(cr.MS * 40)
    assert m["R"] == pytest.approx(0.0)
    assert m["S"] == 0.0
    upright = np.sin(np.linspace(0, np.pi, 40))
    assert cr.qrs_measures(upright)["Q_DUR"] == 0.0


def test_signed_extreme_and_terminal_force():
    assert cr.signed_extreme(np.array([0.1, -0.3, 0.2])) == pytest.approx(-0.3)
    assert np.isnan(cr.signed_extreme(np.array([])))
    p = np.array([0.0, 0.05, 0.1, 0.02, -0.05, -0.1, -0.05])
    assert cr.terminal_force(p) == pytest.approx(0.1 * cr.MS * 3)
    assert cr.terminal_force(np.array([0.0, 0.1, 0.05])) == 0.0


def test_thresholds_depend_on_age_and_sex():
    age = np.array([30.0, 50.0, 30.0, np.nan])
    sex = np.array([0, 0, 1, 0])
    assert cr.threshold_values("ste_v2v3", age, sex).tolist() == [0.25, 0.2, 0.15, 0.2]
    assert cr.threshold_values("cornell", age, sex).tolist() == [2.8, 2.8, 2.0, 2.8]
    assert cr.threshold_values("qtc_long", age, sex).tolist() == [450.0, 450.0, 460.0, 450.0]
    assert cr.threshold_values(0.1, age, sex).tolist() == [0.1] * 4


def _table(n: int = 2) -> pd.DataFrame:
    columns = {name: np.zeros(n) for name in cr.used_columns()}
    for lead in LEADS:
        columns[f"R_{lead}"] = np.full(n, 0.5)
        columns[f"S_{lead}"] = np.full(n, 0.2)
        columns[f"T_{lead}"] = np.full(n, 0.3)
    columns.update({"QRSD": np.full(n, 90.0), "PR": np.full(n, 160.0), "QTC": np.full(n, 420.0)})
    columns.update({"AXIS": np.full(n, 30.0), "PDUR": np.full(n, 100.0), "LIMB_PP": np.full(n, 1.0)})
    columns.update({"PREC_PP": np.full(n, 2.0), "SOKOLOW": np.full(n, 2.0), "CORNELL": np.full(n, 1.0)})
    table = pd.DataFrame(columns)
    for lead in LEADS:
        table[f"RS_{lead}"] = table[f"R_{lead}"] - table[f"S_{lead}"]
        table[f"SR_{lead}"] = -table[f"RS_{lead}"]
    return table


def test_exceedances_hinge_and_composites():
    table = _table()
    table.loc[1, "ST_J_II"] = 0.2
    table.loc[1, "QRSD"] = 140.0
    table.loc[1, "TERM_V1"] = 0.1
    scales = pd.Series(1.0, index=table.columns)
    scales["ST_J_II"] = 0.05
    z = cr.exceedances(table, scales, np.array([50.0, 50.0]), np.array([0, 0]))
    names = [c.name for c in cr.CRITERIA]
    ste = names.index("ste")
    assert z[1, ste, LEADS.index("II")] == pytest.approx(2.0)
    assert np.isnan(z[0, ste, LEADS.index("V2")])
    rbbb = names.index("rbbb_v1")
    assert z[1, rbbb, LEADS.index("V1")] == pytest.approx(0.1)
    e = cr.evidence(z)
    assert e[0, names.index("rvh_v1"), LEADS.index("V1")] == pytest.approx(0.3)
    assert e[0, names.index("ste")].max() == 0.0
    assert e.min() == 0.0
    assert cr.met_any_lead(z)[1, ste]


def test_lead_wave_takes_maximum_per_component():
    scores = np.zeros((1, len(cr.CRITERIA), 12))
    names = [c.name for c in cr.CRITERIA]
    scores[0, names.index("ste"), LEADS.index("II")] = 1.5
    scores[0, names.index("std"), LEADS.index("II")] = 0.5
    scores[0, names.index("pr_long"), :] = 0.7
    out = cr.lead_wave(scores)
    assert out.shape == (1, 12, 5)
    assert out[0, LEADS.index("II"), cr.COMPONENTS.index("ST")] == 1.5
    assert out[0, 0, cr.COMPONENTS.index("P")] == 0.7
    assert out[0, 0, cr.COMPONENTS.index("PQ")] == 0.7
    assert out[0, 0, cr.COMPONENTS.index("T")] == 0.0
    assert cr.evidence_leads(out)[0].all()


def test_support_heads_and_mil_weights():
    names = [c.name for c in cr.CRITERIA]
    q_path = cr.CRITERIA[names.index("q_path")]
    assert cr.support_heads(q_path, "II") == ("IMI", "ILMI", "QWAVE")
    heads = ["NORM", "IMI", "ILMI", "QWAVE", "LNGQT"]
    extra = sorted(
        {h for c in cr.CRITERIA for lead in c.leads for h in cr.support_heads(c, lead)} - set(heads)
    )
    heads += extra
    prob = np.zeros((1, len(heads)))
    prob[0, heads.index("IMI")] = 0.8
    prob[0, heads.index("NORM")] = 0.3
    w = cr.mil_weights(prob, heads)
    assert w[0, names.index("q_path"), LEADS.index("II")] == pytest.approx(0.8)
    assert w[0, names.index("qtc_short"), 0] == pytest.approx(0.7)
    assert w[0, names.index("q_path"), LEADS.index("III")] == 0.0


def test_consensus_reliability_and_wrap():
    a = pd.DataFrame({"x": [1.0, 1.0], "y": [np.nan, 2.0]})
    b = pd.DataFrame({"x": [2.0, 3.0], "y": [4.0, np.nan]})
    c = pd.DataFrame({"x": [10.0, np.nan]})
    out = cr.consensus([a, b, c])
    assert out["x"].tolist() == [2.0, 2.0]
    assert out["y"].tolist() == [4.0, 2.0]
    r = cr.reliability(np.array([[0.0, 1.0, 4.0, np.nan]]), np.array([2.0, 2.0, 2.0, 2.0]))
    assert r.tolist() == [[1.0, 1.0, 0.5, 1.0]]
    assert cr.wrap_degrees(pd.Series([190.0, -180.0, 45.0])).tolist() == [-170.0, 180.0, 45.0]


def test_paint_and_lexicographic():
    intervals = np.array([[[10, 20], [20, 30], [30, 40], [40, 50], [50, 70]]])
    scores = np.zeros((12, 5))
    scores[3, cr.COMPONENTS.index("ST")] = 2.0
    field = cr.paint(scores, intervals, samples=100)
    assert field[3, 40:50].tolist() == [2.0] * 10
    assert field[3, :40].max() == 0.0
    lex = cr.lexicographic(field, intervals, np.zeros((12, 100)))
    assert lex[0, 30:40].min() == pytest.approx(2.0)
    assert lex[3, 45] == pytest.approx(2.0 / 3.0)


def test_design_appends_two_standardized_columns():
    units = np.array([[0, k % 12, k % 5, 0, 1] for k in range(20)])
    raw = np.random.default_rng(0).random((20, 9))
    extra = np.column_stack([np.arange(20.0), np.zeros(20)])
    x = cr.design(raw, np.zeros(12), units, extra)
    assert x.shape == (20, len(cr.FEATURES))
    assert x[:, -2].mean() == pytest.approx(0.0)
    assert np.all(x[:, -1] == 0.0)
    assert cr.criteria_features(0.25)[-2:] == cr.EXTRA_MAPS


def test_normal_reference_applies_floors():
    normals = pd.DataFrame({"ST_J_II": np.zeros(50), "QRSD": np.linspace(80, 120, 50)})
    ref = cr.normal_reference(normals, ["ST_J_II", "QRSD"])
    assert ref.loc["ST_J_II", "scale"] == pytest.approx(0.025)
    assert ref.loc["QRSD", "scale"] == pytest.approx(20.0 / 1.349, rel=1e-3)
    assert cr.measure_of("ST_J_aVL") == "ST_J"
    assert cr.measure_of("SOKOLOW") == "SOKOLOW"
