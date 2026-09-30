"""Checks of the Experiment 032 finding groups and per-set statistics."""

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import challenge_labels
from ecg_experiment.rhythm_findings import (
    GROUPS,
    PARTIAL_ANNOTATION,
    SCHEMES,
    budget_sensitivity,
    clear_normal,
    finding_label,
    frequent_pvc,
    label_table,
    set_statistics,
    usability,
)

SCP_CODES = {"PVC", "BIGU", "TRIGU", "PRC(S)", "WPW", "AFIB", "AFLT", "SVTAC", "PSVT", "SVARR", "2AVB",
             "3AVB", "LNGQT"}
SPH_CODES = {"60", "108", "80", "50", "51", "54", "84", "85", "86", "87", "88", "148"}


def test_every_group_has_positive_and_ambiguous_lists_for_every_scheme_without_overlap():
    for definition in GROUPS.values():
        for scheme in SCHEMES:
            assert definition["positive"][scheme]
            assert not definition["positive"][scheme] & definition["ambiguous"][scheme]


def test_group_codes_are_known_codes_of_their_scheme():
    for definition in GROUPS.values():
        for kind in ("positive", "ambiguous"):
            assert definition[kind]["ptbxl"] <= SCP_CODES
            assert definition[kind]["sph"] <= SPH_CODES


def test_snomed_codes_are_in_the_official_tables_when_they_are_available():
    try:
        official = challenge_labels.load_official()
    except FileNotFoundError:
        pytest.skip("official Challenge tables are not available")
    for definition in GROUPS.values():
        for kind in ("positive", "ambiguous"):
            assert definition[kind]["snomed"] <= set(official.index)


def test_finding_label_is_positive_ambiguous_or_negative():
    assert finding_label({"426783006", "164889003"}, "af_flutter", "snomed") == 1.0
    assert np.isnan(finding_label({"426783006", "195042002"}, "high_grade_av_block", "snomed"))
    assert finding_label({"426783006", "195042002", "27885002"}, "high_grade_av_block", "snomed") == 1.0
    assert finding_label({"426783006"}, "preexcitation", "snomed") == 0.0
    assert finding_label({"NORM", "SR"}, "ventricular_ectopy", "ptbxl") == 0.0
    assert np.isnan(finding_label({"BIGU"}, "ventricular_ectopy", "ptbxl"))


def test_label_table_leaves_unannotated_groups_undefined_for_cpsc_2018():
    labels = label_table([{"164884008"}, {"164884008"}], "snomed", ["cpsc_2018", "georgia"])
    assert labels.loc[0, "ventricular_ectopy"] == 1.0
    assert labels.loc[1, "ventricular_ectopy"] == 1.0
    for group in GROUPS:
        if group not in PARTIAL_ANNOTATION["cpsc_2018"]:
            assert np.isnan(labels.loc[0, group])
            assert labels.loc[1, group] == 0.0


def test_frequent_pvc_reads_the_sph_modifiers():
    assert frequent_pvc("22;60+310")
    assert frequent_pvc("60+341")
    assert not frequent_pvc("60+308")
    assert not frequent_pvc("60")
    assert not frequent_pvc("30+310")


def test_clear_normal_requires_a_standard_negative_without_a_positive_finding():
    labels = pd.DataFrame({"af_flutter": [0.0, 1.0, 0.0, np.nan], "svt": [0.0, 0.0, 0.0, 0.0]})
    mask = clear_normal(np.array([0.0, 0.0, 1.0, 0.0]), labels)
    assert mask.tolist() == [True, False, False, True]


def test_budget_sensitivity_counts_positives_strictly_above_the_normal_quantile():
    scores = np.concatenate([np.arange(100.0), [95.5, 98.5, 50.0]])
    y = np.array([0] * 100 + [1, 1, 1])
    normal = y == 0
    assert budget_sensitivity(scores, y, normal, 50) == pytest.approx(2 / 3)
    assert budget_sensitivity(scores, y, normal, 10) == pytest.approx(1 / 3)


def test_set_statistics_gives_paired_contrasts_and_reproducible_intervals():
    rng = np.random.default_rng(3)
    y = (rng.random(400) < 0.2).astype(np.int64)
    good, weak = y + rng.normal(0, 0.5, 400), y + rng.normal(0, 2.0, 400)
    units = np.arange(400).astype(str)
    normal = y == 0
    found = set_statistics(y, units, normal, {"good": good, "weak": weak}, [("good", "weak")], (50,), 50, 7)
    again = set_statistics(y, units, normal, {"good": good, "weak": weak}, [("good", "weak")], (50,), 50, 7)
    assert found == again
    good_auroc = found["scores"]["good"]["auroc"]
    assert good_auroc["ci_low"] <= good_auroc["value"] <= good_auroc["ci_high"]
    difference = found["contrasts"]["good_minus_weak"]["auroc"]
    weak_auroc = found["scores"]["weak"]["auroc"]["value"]
    assert difference["difference"] == pytest.approx(good_auroc["value"] - weak_auroc)
    assert difference["ci_low"] > 0
    assert "sensitivity_at_50" in found["scores"]["good"]
    assert found["positives"] == int(y.sum())


def test_usability_applies_the_pre_registered_rule():
    assert usability({"value": 0.93, "ci_low": 0.86, "ci_high": 0.97}) == "usable"
    assert usability({"value": 0.93, "ci_low": 0.80, "ci_high": 0.97}) == "undetermined"
    assert usability({"value": 0.80, "ci_low": 0.75, "ci_high": 0.89}) == "not_usable"
    assert usability({"value": 0.88, "ci_low": 0.84, "ci_high": 0.91}) == "undetermined"
