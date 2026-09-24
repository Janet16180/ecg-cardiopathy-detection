"""Selection summaries retain zero-accepted groups and distinct exclusion reasons."""

from scripts.audit_ecg_selection_bias import aggregate, age_band, sex_value


def test_demographic_categories_at_boundaries() -> None:
    assert [age_band(value) for value in
            (None, "17", "18", "30", "31", "50", "51", "70", "71", "120", "121", "bad")] == [
                "unknown", "under_18", "18_to_30", "18_to_30", "31_to_50",
                "31_to_50", "51_to_70", "51_to_70", "over_70", "over_70",
                "unknown", "unknown"]
    assert [sex_value(value) for value in (" Male ", "True", "female", "false", "X", None)] == [
        "male", "male", "female", "female", "unknown", "unknown"]


def test_retention_summary_keeps_zero_accepted_and_sparse_outcomes() -> None:
    items = [
        ("beta", "18_to_30", "female", "duration_contract"),
        ("alpha", "unknown", "unknown", "accepted"),
        ("beta", "18_to_30", "female", "constant_lead"),
        ("beta", "18_to_30", "female", "accepted"),
        ("beta", "over_70", "male", "duration_contract"),
    ]
    result = aggregate(items)

    assert list(result) == ["alpha", "beta"]
    assert result["alpha"]["records"] == result["alpha"]["accepted"] == 1
    assert result["alpha"]["sex"]["unknown"] == {
        "total": 1, "accepted": 1, "retained_percent": 100.0, "other_outcomes": {}}
    assert result["beta"]["records"] == 4
    assert result["beta"]["accepted"] == 1
    assert result["beta"]["age_bands"]["18_to_30"] == {
        "total": 3, "accepted": 1, "retained_percent": 33.33,
        "other_outcomes": {"constant_lead": 1, "duration_contract": 1}}
    assert result["beta"]["age_bands"]["over_70"] == {
        "total": 1, "accepted": 0, "retained_percent": 0.0,
        "other_outcomes": {"duration_contract": 1}}
    assert result["beta"]["sex"]["male"] == result["beta"]["age_bands"]["over_70"]
    assert aggregate([]) == {}
