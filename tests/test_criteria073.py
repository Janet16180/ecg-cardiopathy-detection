"""Tests for the Experiment 073 per-lead and beat-level criteria."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecg_experiment import criteria069 as cr
from ecg_experiment import criteria073 as pl
from ecg_experiment.waveforms import LEADS


def test_table_replaces_every_global_interval() -> None:
    measures = {m for c in pl.CRITERIA if c.name not in pl.RECORD_LEVEL for m, _, _ in c.conditions}
    assert not measures & {"QRSD", "PR", "QTC", "PDUR"}
    short = next(c for c in pl.CRITERIA if c.name == "qtc_short")
    assert short.conditions == (("QTC", -1, 390.0),)
    assert {"QRSDL", "PRL", "QTCL", "PDURL"} <= measures
    names = [c.name for c in pl.CRITERIA]
    assert "lafb" not in names
    assert "lpfb" not in names
    assert len(names) == len(cr.CRITERIA) + 2


def test_fascicular_patterns() -> None:
    by_name = {c.name: c for c in pl.CRITERIA}
    assert by_name["lafb_rs"].leads == ("II", "III", "aVF")
    assert by_name["lafb_qr"].leads == ("aVL",)
    assert by_name["lpfb_rs"].leads == ("I", "aVL")
    assert by_name["lpfb_qr"].leads == ("III", "aVF")
    assert ("SR", 1, 0.0) in by_name["lafb_rs"].conditions
    assert ("AXIS", -1, -45.0) in by_name["lafb_qr"].conditions


def _table(qrs: dict[str, float]) -> tuple[pd.DataFrame, pd.Series]:
    columns = pl.used_columns()
    table = pd.DataFrame([dict.fromkeys(columns, np.nan)])
    for lead, value in qrs.items():
        table[f"QRSDL_{lead}"] = value
    scales = pd.Series(1.0, index=columns)
    return table, scales


def test_ivcd_paints_only_leads_over_threshold() -> None:
    table, scales = _table(dict.fromkeys(LEADS, 100.0) | {"V1": 130.0, "V2": 112.0})
    z = pl.exceedances(table, scales, np.array([50.0]), np.array([0.0]))
    e = pl.lead_wave(cr.evidence(z))
    qrs = e[0, :, pl.COMPONENTS.index("QRS")]
    assert qrs[LEADS.index("V1")] == 20.0
    assert qrs[LEADS.index("V2")] == 2.0
    assert (qrs[[k for k, lead in enumerate(LEADS) if lead not in ("V1", "V2")]] == 0).all()


def test_shift_moves_threshold() -> None:
    table, scales = _table(dict.fromkeys(LEADS, 100.0))
    offsets = {f"QRSDL_{lead}": -15.0 for lead in LEADS}
    z = pl.exceedances(table, scales, np.array([50.0]), np.array([0.0]), offsets=offsets)
    ivcd = [c.name for c in pl.CRITERIA].index("ivcd")
    assert np.allclose(z[0, ivcd], 5.0)


def test_unig_intervals_need_p() -> None:
    raw = pd.DataFrame({name: [1.0, 1.0] for name in pl.unig_columns()[1:]}, index=[1, 2])
    raw["P_Found_Global"] = [1, 0]
    raw["QRS_On_V1"] = [400.0, 400.0]
    raw["P_On_V1"] = [240.0, 240.0]
    raw["QT_Int_V1"] = [400.0, 400.0]
    out = pl.unig_intervals(raw, pd.Series([1000.0, 500.0], index=[1, 2]))
    assert out.loc[1, "PRL_V1"] == 160.0
    assert np.isnan(out.loc[2, "PRL_V1"])
    assert out.loc[1, "QTCL_V1"] == 400.0
    assert out.loc[2, "QTCL_V1"] == 477.0


def test_sl12_zero_duration_missing() -> None:
    raw = pd.DataFrame({name: [0.0, 110.0] for name in pl.sl12_columns()[1:]}, index=[1, 2])
    out = pl.sl12_intervals(raw)
    assert np.isnan(out.loc[1, "PDURL_II"])
    assert out.loc[2, "PDURL_II"] == 110.0


def test_own_intervals_medians() -> None:
    bounds = np.full((3, 12, 3, 2), -1, np.int64)
    bounds[:, :, 1] = [100, 145]
    bounds[:, :, 0] = [20, 70]
    bounds[:, :, 2] = [200, 300]
    bounds[2, 0, 0] = -1
    bounds[1:, 1, 0] = -1
    out = pl.own_intervals(bounds, 1000.0)
    assert out["QRSDL_I"] == 90.0
    assert out["PRL_I"] == 160.0
    assert out["PDURL_I"] == 100.0
    assert np.isnan(out["PRL_II"])
    assert out["QTCL_I"] == 400.0


def test_beat_features_flag_premature_deviant_beat() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 0.005, (12, 5000))
    starts = [300, 1100, 1900, 2700, 3200, 4000]
    for s in starts:
        signal[:, s : s + 40] += np.hanning(40)
    signal[:, 3200:3240] -= 2 * np.hanning(40)
    intervals = np.zeros((len(starts), 5, 2), np.int64)
    for b, s in enumerate(starts):
        intervals[b] = [[s - 80, s - 40], [s - 40, s], [s, s + 40], [s + 40, s + 100], [s + 100, s + 200]]
    bounds = np.zeros((len(starts), 12, 3, 2), np.int64)
    bounds[:, :, 1, 0] = np.array(starts)[:, None]
    bounds[:, :, 1, 1] = np.array(starts)[:, None] + 40
    bounds[4, :, 1, 1] = 3200 + 70
    f = pl.beat_features(signal, intervals, bounds)
    assert np.isnan(f["pi"][0])
    assert f["pi"][4] < 0.7
    assert (f["delta"][4] > 1.0).all()
    assert (np.delete(f["delta"], 4, axis=0) < 0.6).all()
    assert f["duration"][4, 0] == 140.0
    assert f["duration"][0, 0] == 80.0
    ref = {
        "pi01": 0.8,
        "pi_scale": 0.05,
        "delta99": np.full(12, 0.6),
        "delta_scale": np.full(12, 0.1),
        "duration_scale": np.full(12, 10.0),
    }
    z = pl.beat_exceedances(f, ref)
    b = pl.beat_wave(z)
    assert (b[4, :, pl.COMPONENTS.index("QRS")] > 0).all()
    assert (b[4, :, pl.COMPONENTS.index("T")] > 0).all()
    assert (np.delete(b, 4, axis=0) == 0).all()
    assert np.allclose(z[4, 1], 2.0)


def test_beat_reference_quantiles() -> None:
    features = [
        {
            "pi": np.r_[np.nan, np.linspace(0.8, 1.2, 99)],
            "delta": np.ones((100, 12)),
            "duration": np.ones((100, 12)),
        }
    ]
    ref = pl.beat_reference(features)
    assert np.isclose(ref["pi01"], np.percentile(np.linspace(0.8, 1.2, 99), 1))
    assert np.allclose(ref["delta99"], 1.0)
    assert np.allclose(ref["delta_scale"], pl.BEAT_FLOORS["delta"])


def test_paint_beats_matches_069_without_beats() -> None:
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 2, (12, 5))
    intervals = (
        np.array([[[10, 20], [20, 30], [30, 50], [50, 70], [70, 100]]] * 2)
        + np.array([0, 300])[:, None, None]
    )
    assert np.array_equal(pl.paint_beats(scores, None, intervals), cr.paint(scores, intervals))
    zeros = np.zeros((2, 12, 5))
    assert np.array_equal(pl.paint_beats(scores, zeros, intervals), cr.paint(scores, intervals))
    beats = zeros.copy()
    beats[1, 3, 2] = 9.0
    field = pl.paint_beats(scores, beats, intervals)
    assert field[3, 335] == 9.0
    assert field[3, 35] == scores[3, 2]


def test_shift_offsets() -> None:
    normals = pd.DataFrame({f"{m}_{lead}": [80.0, 90.0] for m in pl.INTERVALS for lead in LEADS})
    for name in pl.GLOBAL_OF.values():
        normals[name] = [95.0, 95.0]
    offsets = pl.shift_offsets(normals)
    assert offsets["QRSDL_V1"] == -10.0
    assert len(offsets) == 48


def test_normal_reference_floors() -> None:
    normals = pd.DataFrame({"QRSDL_V1": [80.0, 81.0, 82.0], "R_V1": [0.5, 0.6, 0.7]})
    ref = pl.normal_reference(normals, ["QRSDL_V1", "R_V1"])
    assert ref.loc["QRSDL_V1", "scale"] == 10.0
    assert ref.loc["R_V1", "floor"] == cr.FLOORS["R"]


def test_same_components() -> None:
    a = np.zeros((0, 5, 2), np.int64)
    assert pl.same_components(a, None)
    assert not pl.same_components(np.ones((1, 5, 2), np.int64), None)
