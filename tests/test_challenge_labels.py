"""Checks of the Challenge SNOMED-to-endpoint mapping."""

import math

import pandas as pd
import pytest

from ecg_experiment import challenge_labels
from ecg_experiment.challenge_labels import (
    MAPPING_DIR,
    SUPERCLASSES,
    code_group,
    equivalences,
    load_official,
    mapping_table,
    record_labels,
)

HAVE_OFFICIAL = all((MAPPING_DIR / name).exists() for name in challenge_labels.MAPPING_SHA256)


def small_official() -> pd.DataFrame:
    rows = {
        "426783006": ("sinus rhythm", True, ""),
        "426177001": ("sinus bradycardia", True, ""),
        "164889003": ("atrial fibrillation", True, ""),
        "164934002": ("t wave abnormal", True, ""),
        "733534002": ("complete left bundle branch block", True,
                      "We score 733534002 and 164909002 as the same diagnosis"),
        "164909002": ("left bundle branch block", True,
                      "We score 733534002 and 164909002 as the same diagnosis"),
        "164865005": ("myocardial infarction", False, ""),
        "55827005": ("left ventricular high voltage", False, ""),
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["name", "scored", "notes"]).assign(
        abbreviation="")


def test_sinus_rhythm_alone_is_the_only_primary_negative():
    official = small_official()
    assert record_labels(["426783006"], official)["primary"] == 0
    bradycardia = record_labels(["426783006", "426177001"], official)
    assert math.isnan(bradycardia["primary"])
    assert bradycardia["secondary"] == 0


def test_any_superclass_code_makes_a_record_positive():
    official = small_official()
    result = record_labels(["426783006", "164889003", "164934002"], official)
    assert result["primary"] == 1
    assert result["secondary"] == 1
    assert result["STTC"] == 1
    assert result["MI"] == 0


def test_other_and_unknown_codes_stay_undefined_never_negative():
    official = small_official()
    for codes in (["426783006", "164889003"], ["426783006", "55827005"], ["426783006", "999999"], []):
        result = record_labels(codes, official)
        assert math.isnan(result["primary"])
        assert math.isnan(result["secondary"])
    assert code_group("999999", official) == "unknown"
    assert code_group("164889003", official) == "other"


def test_equivalences_come_from_the_official_notes():
    official = small_official()
    assert equivalences(official) == {"733534002": "733534002", "164909002": "733534002"}
    table = mapping_table(official)
    assert table.loc["164909002", "canonical"] == "733534002"
    assert table.loc["164909002", "group"] == table.loc["733534002", "group"] == "CD"


def test_superclasses_do_not_overlap():
    members = [code for codes in SUPERCLASSES.values() for code in codes]
    assert len(members) == len(set(members))


@pytest.mark.skipif(not HAVE_OFFICIAL, reason="official Challenge mapping not downloaded")
def test_official_tables_define_every_mapped_code_and_the_four_pairs():
    official = load_official()
    assert set().union(*SUPERCLASSES.values()) <= set(official.index)
    pairs = equivalences(official)
    assert {code: pairs[code] for code in ("59118001", "164909002", "63593006", "17338001")} == {
        "59118001": "713427006", "164909002": "733534002", "63593006": "284470004", "17338001": "427172004"}
    table = mapping_table(official)
    assert table.groupby("canonical")["group"].nunique().max() == 1
