"""Hand-written cases for every MLC-01 label rule."""

import math

import pandas as pd
import pytest

from ecg_experiment import ROOT, challenge_labels, mlc_labels
from ecg_experiment.mlc_labels import (
    CONDITIONS,
    SPACE_B,
    SPACE_C,
    condition_states,
    parse_aha,
    parse_snomed,
    ptbxl_codes,
    record_labels,
    sensitivity_state,
    space_a_states,
)

PTBXL_DIR = ROOT / "data/raw/ptb-xl/1.0.3"
SPH_CODES = ROOT / "data/raw/sph/sph/17912507/code.csv"
HAVE_TABLES = (PTBXL_DIR / "scp_statements.csv").exists() and SPH_CODES.exists() and all(
    (challenge_labels.MAPPING_DIR / name).exists() for name in challenge_labels.MAPPING_SHA256)
CLASSES = {"NORM": "NORM", "IMI": "MI", "ASMI": "MI", "INJAS": "MI", "LVH": "HYP", "NDT": "STTC",
           "ISC_": "STTC", "IRBBB": "CD", "CRBBB": "CD", "1AVB": "CD"}
DIAGNOSTIC = frozenset(CLASSES)
UNDEFINED = float("nan")


def nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


def snomed(source: str, *codes: str) -> dict[str, float]:
    return condition_states(source, frozenset(codes))


def ptbxl(codes: dict[str, float], axis: str = "") -> dict[str, float]:
    return condition_states("ptbxl", ptbxl_codes(codes, axis))


def aha(text: str) -> dict[str, float]:
    return condition_states("sph", parse_aha(text))


def test_parsers_keep_codes_modifiers_and_axis():
    assert parse_snomed("164889003, 59118001,") == {"164889003", "59118001"}
    assert parse_aha("22;145+362;60+310+340") == (("22", frozenset()), ("145", frozenset({"362"})),
                                                   ("60", frozenset({"310", "340"})))
    assert ptbxl_codes({"SR": 0.0}, "ALAD") == {"SR": 0.0, "axis:ALAD": 0.0}
    assert ptbxl_codes({"SR": 0.0}, "") == {"SR": 0.0}


def test_spaces_have_the_protocol_sizes_and_share_definitions():
    assert len(SPACE_B) == 14
    assert len(SPACE_C) == 30
    assert len(set(SPACE_C) - set(SPACE_B)) == 18
    assert set(SPACE_B) | set(SPACE_C) <= set(CONDITIONS)
    labels = record_labels("ningbo", frozenset({"426177001", "164934002"}), CLASSES, DIAGNOSTIC)
    for key in set(SPACE_B) & set(SPACE_C):
        assert labels[f"b_{key}"] == labels[f"c_{key}"]


def test_positive_negative_and_ambiguous_states():
    states = snomed("ningbo", "426177001")
    assert states["sinus_bradycardia"] == 1
    assert states["sinus_tachycardia"] == 0
    assert nan(snomed("ningbo", "426627000")["sinus_bradycardia"])
    assert snomed("ningbo", "426627000", "426177001")["sinus_bradycardia"] == 1
    assert nan(snomed("georgia", "6374002")["complete_rbbb"])
    assert nan(snomed("georgia", "428750005")["t_wave_abnormality"])
    assert snomed("georgia", "428750005", "164934002")["t_wave_abnormality"] == 1


def test_cpsc_2018_is_defined_only_for_conditions_among_its_nine_classes():
    states = snomed("cpsc_2018", "426783006")
    defined = {"af_or_flutter", "premature_ventricular_complex", "first_degree_av_block", "complete_rbbb",
               "st_depression"}
    for key in SPACE_B:
        assert (states[key] == 0) if key in defined else nan(states[key]), key
    assert nan(snomed("cpsc_2018", "427084000")["sinus_tachycardia"])
    assert snomed("cpsc_2018_extra", "427084000")["sinus_tachycardia"] == 1
    assert snomed("cpsc_2018", "164884008")["premature_ventricular_complex"] == 1
    c_states = snomed("cpsc_2018", "164909002", "164931005")
    assert c_states["complete_lbbb"] == 1
    assert c_states["st_elevation"] == 1
    assert nan(c_states["atrial_flutter"])


def test_cpsc_2018_partial_annotation_keeps_composite_labels_defined():
    assert snomed("cpsc_2018", "164889003")["af_or_flutter"] == 1
    assert snomed("cpsc_2018", "426783006")["af_or_flutter"] == 0
    groups = space_a_states("cpsc_2018", frozenset({"429622005"}), CLASSES)
    assert groups["STTC"] == 1
    assert groups["CD"] == 0
    assert nan(groups["MI"])
    assert nan(groups["HYP"])
    assert ("cpsc_2018", "af_or_flutter") in mlc_labels.PARTIAL
    status = {(row["label"], row["source"]): row["status"] for row in mlc_labels.status_rows()}
    assert status[("b_af_or_flutter", "cpsc_2018")] == "partial"
    assert status[("b_lvh", "cpsc_2018")] == "not annotated"
    assert status[("a_MI", "cpsc_2018")] == "not annotated"
    assert status[("a_STTC", "cpsc_2018")] == "partial"


def test_sources_that_do_not_annotate_a_condition_are_undefined():
    assert nan(snomed("chapman_shaoxing", "426783006")["sinus_arrhythmia"])
    assert snomed("ningbo", "426783006")["sinus_arrhythmia"] == 0
    assert nan(snomed("chapman_shaoxing", "426783006")["incomplete_rbbb"])
    assert nan(snomed("chapman_shaoxing", "426783006")["lafb"])
    assert nan(snomed("chapman_shaoxing", "426783006")["lpfb"])
    assert nan(snomed("ningbo", "164889003")["atrial_fibrillation"])
    assert snomed("ningbo", "164889003")["af_or_flutter"] == 1


def test_georgia_st_depression_is_undefined():
    assert nan(snomed("georgia", "429622005")["st_depression"])
    assert nan(snomed("georgia", "426783006")["st_depression"])
    assert snomed("ningbo", "429622005")["st_depression"] == 1
    assert snomed("georgia", "164931005")["st_elevation"] == 1


def test_sph_st_deviation_needs_a_direction_modifier():
    assert aha("145+362")["st_depression"] == 1
    assert aha("145+363")["st_depression"] == 0
    assert aha("145+363")["st_elevation"] == 1
    assert nan(aha("145")["st_depression"])
    assert nan(aha("146")["st_depression"])
    assert aha("145;145+362")["st_depression"] == 1
    assert aha("1")["st_depression"] == 0
    assert aha("146")["t_wave_abnormality"] == 1


def test_lv_high_voltage_alone_leaves_hypertrophy_undefined():
    assert nan(snomed("ningbo", "55827005")["lvh"])
    assert snomed("ningbo", "55827005", "164873001")["lvh"] == 1
    assert nan(ptbxl({"VCLVH": 0.0, "SR": 0.0})["lvh"])
    assert ptbxl({"VCLVH": 0.0, "LVH": 50.0})["lvh"] == 1
    assert nan(space_a_states("ningbo", frozenset({"55827005"}), CLASSES)["HYP"])
    assert space_a_states("ningbo", frozenset({"55827005", "164873001"}), CLASSES)["HYP"] == 1
    assert nan(space_a_states("ptbxl", {"VCLVH": 0.0}, CLASSES)["HYP"])
    assert space_a_states("ptbxl", {"NDT": 100.0}, CLASSES)["HYP"] == 0


def test_ptbxl_left_axis_deviation_uses_only_alad():
    assert ptbxl({"SR": 0.0}, "ALAD")["left_axis_deviation"] == 1
    assert nan(ptbxl({"SR": 0.0}, "LAD")["left_axis_deviation"])
    assert ptbxl({"SR": 0.0}, "MID")["left_axis_deviation"] == 0
    assert ptbxl({"SR": 0.0}, "")["left_axis_deviation"] == 0


def test_ptbxl_injury_alone_leaves_infarction_undefined():
    assert nan(ptbxl({"INJAS": 100.0})["mi_any"])
    assert ptbxl({"INJAS": 100.0, "IMI": 15.0})["mi_any"] == 1
    assert ptbxl({"NORM": 100.0})["mi_any"] == 0
    assert space_a_states("ptbxl", {"INJAS": 100.0}, CLASSES)["MI"] == 1


def test_contradictory_pairs_make_both_undefined():
    states = snomed("georgia", "59118001", "713426002")
    assert nan(states["complete_rbbb"])
    assert nan(states["incomplete_rbbb"])
    states = snomed("chapman_shaoxing", "164909002", "59118001")
    assert nan(states["complete_rbbb"])
    assert nan(states["complete_lbbb"])
    assert snomed("georgia", "713426002")["incomplete_rbbb"] == 1
    assert snomed("georgia", "164909002")["complete_lbbb"] == 1
    states = ptbxl({"CRBBB": 100.0, "IRBBB": 100.0})
    assert nan(states["complete_rbbb"])
    assert nan(states["incomplete_rbbb"])
    assert nan(aha("104;106")["complete_rbbb"])
    assert aha("104;105")["incomplete_rbbb"] == 1


def test_unstated_completeness_follows_the_official_equivalences():
    assert snomed("chapman_shaoxing", "59118001")["complete_rbbb"] == 1
    assert snomed("georgia", "164909002")["complete_lbbb"] == 1
    pairs = (("733534002", "164909002"), ("713427006", "59118001"), ("284470004", "63593006"),
             ("427172004", "17338001"))
    for first, second in pairs:
        for definition in CONDITIONS.values():
            positive = definition["positive"]["snomed"]
            assert (first in positive) == (second in positive), (first, second)
        assert snomed("ningbo", first) == snomed("ningbo", second)


def test_sph_code_153_sets_no_condition_and_counts_for_hyp_in_space_a():
    states = aha("153")
    for key in SPACE_B:
        assert states[key] == 0, key
    assert aha("142;153")["lvh"] == 1
    assert aha("142;153")["t_wave_abnormality"] == 0
    groups = space_a_states("sph", parse_aha("153"), CLASSES)
    assert groups["HYP"] == 1
    assert groups["STTC"] == 0
    groups = space_a_states("sph", parse_aha("147;153"), CLASSES)
    assert groups["STTC"] == 1
    assert groups["HYP"] == 1


def test_codes_a_vocabulary_cannot_express_are_undefined():
    assert nan(aha("1")["paced_rhythm"])
    assert nan(snomed("georgia", "426783006")["inferior_mi"])
    assert nan(snomed("georgia", "164865005")["anterior_mi"])
    assert snomed("georgia", "54329005")["anterior_mi"] == 1


def test_likelihood_flag_marks_positives_resting_on_low_likelihood_diagnostic_statements():
    def flags(codes: dict[str, float]) -> dict[str, float | bool]:
        return record_labels("ptbxl", ptbxl_codes(codes, ""), CLASSES, DIAGNOSTIC)

    low = flags({"LVH": 35.0})
    assert low["b_lvh"] == 1
    assert low["lik50_lvh"] is True
    assert flags({"LVH": 50.0})["lik50_lvh"] is False
    assert flags({"IMI": 15.0, "ASMI": 100.0})["lik50_mi_any"] is False
    assert flags({"IMI": 15.0})["lik50_mi_any"] is True
    assert flags({"1AVB": 15.0, "LPR": 0.0})["lik50_first_degree_av_block"] is False
    assert flags({"STD_": 0.0})["lik50_st_depression"] is False
    assert flags({"NORM": 100.0})["lik50_lvh"] is False
    assert record_labels("sph", parse_aha("142"), CLASSES, DIAGNOSTIC)["lik50_lvh"] is False
    assert nan(sensitivity_state(1.0, True))
    assert sensitivity_state(1.0, False) == 1
    assert sensitivity_state(0.0, False) == 0


def test_normal_column_follows_each_source_definition():
    def normal(source: str, record: object) -> float:
        return record_labels(source, record, CLASSES, DIAGNOSTIC)["normal"]

    assert normal("ptbxl", ptbxl_codes({"NORM": 100.0, "SR": 0.0}, "MID")) == 1
    assert normal("ptbxl", ptbxl_codes({"NORM": 100.0, "SR": 0.0, "STACH": 0.0}, "")) == 0
    assert normal("georgia", frozenset({"426783006"})) == 1
    assert normal("georgia", frozenset({"426783006", "426177001"})) == 0
    assert nan(normal("cpsc_2018_extra", frozenset({"426783006"})))
    assert normal("sph", parse_aha("1")) == 1
    assert normal("sph", parse_aha("1;22")) == 0


def test_state_names():
    assert mlc_labels.state_name(1.0) == "positive"
    assert mlc_labels.state_name(0.0) == "negative"
    assert mlc_labels.state_name(UNDEFINED) == "undefined"


def test_mapping_rows_cover_every_condition_and_vocabulary():
    rows = pd.DataFrame(mlc_labels.mapping_rows())
    assert set(rows["condition"]) == set(CONDITIONS)
    assert (rows.groupby("condition")["vocabulary"].nunique() == 3).all()
    assert set(rows.loc[rows["code"] == "", "role"]) == {"not codable"}


@pytest.mark.skipif(not HAVE_TABLES, reason="official code tables are not available")
def test_every_code_exists_in_its_official_table():
    rows = pd.DataFrame(mlc_labels.mapping_rows())
    rows = rows[rows["code"] != ""]
    scp = set(pd.read_csv(PTBXL_DIR / "scp_statements.csv", index_col=0).index)
    axis = {"axis:ALAD", "axis:LAD", "axis:ARAD", "axis:RAD"}
    official = challenge_labels.load_official()
    aha_codes = set(pd.read_csv(SPH_CODES, dtype=str)["Code"])
    for _, row in rows.iterrows():
        code = row["code"]
        if row["vocabulary"] == "scp":
            assert code in scp | axis, code
        elif row["vocabulary"] == "snomed":
            assert code in official.index, code
        else:
            parts = code.rstrip("=").split("+")
            assert set(parts) <= aha_codes, code


@pytest.mark.skipif(not HAVE_TABLES, reason="official code tables are not available")
def test_official_equivalence_notes_match_the_hand_written_pairs():
    canonical = challenge_labels.equivalences(challenge_labels.load_official())
    pairs = {frozenset(code for code, target in canonical.items() if target == head)
             for head in set(canonical.values())}
    assert pairs == {frozenset(pair) for pair in (("733534002", "164909002"), ("713427006", "59118001"),
                                                  ("284470004", "63593006"), ("427172004", "17338001"))}
