"""Check quarantine, coordinate geometry and reference-patient separation for NTT-02."""

import json

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.reference_transfer_ntt02 import (
    class_groups,
    digest_integer,
    eligibility_table,
    inner_fold,
    interval_features,
    overlap,
    paired_interval,
    patient_macro,
    prototype,
    read_questions,
    support_patients,
    timestamp_options,
)
from scripts.reports.reference_transfer_ntt02 import verify_partition


def question(options=("[1.00s - 1.20s]", "[2.00s - 2.20s]"), answer=0):
    return {"options": list(options), "answer_idx": [answer], "answer": [options[answer]]}


def test_patient_partition_matches_pinned_digest_rule():
    assert digest_integer("ssl_synthetic_partition_v1:20261018:123") == int(
        "fd9b4916ce2a070e65928d82c61c2526e59b4b8d23c823a8ef175ecd538651e2", 16
    )


def test_mixed_fold_patient_and_reserved_roles_are_quarantined_before_content(tmp_path):
    train = next(
        p for p in range(1, 100) if digest_integer(f"ssl_synthetic_partition_v1:20261018:{p}") % 10000 < 8000
    )
    reserved = next(
        p for p in range(1, 100) if digest_integer(f"ssl_synthetic_partition_v1:20261018:{p}") % 10000 >= 8000
    )
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "patient_id": [train, train, reserved],
            "strat_fold": [1, 10, 1],
            "filename_lr": ["unused"] * 3,
        }
    ).to_csv(metadata, index=False)
    table = eligibility_table(metadata)
    assert not table.eligible.any()
    source = tmp_path / "data.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"metadata": {"data_source": "ptbxl", "ecg_id": str(i)}, "data": None})
            for i in (1, 2, 3)
        )
    )
    questions, counts = read_questions(source, table)
    assert questions == []
    assert counts["ineligible_cases_identity_only"] == 3


def test_timestamp_options_preserve_answer_coordinates():
    np.testing.assert_array_equal(timestamp_options(question(answer=1)), [[1, 1.2], [2, 2.2]])
    assert timestamp_options(question(("Between QRS 1 and 2", "Between QRS 2 and 3"))) is None


@pytest.mark.parametrize("options", [("[1s - 0s]", "[2s - 3s]"), ("[0s - 11s]", "[2s - 3s]")])
def test_invalid_coordinates_raise(options):
    with pytest.raises(ValueError, match="timestamp"):
        timestamp_options(question(options))


def test_inconsistent_released_answer_raises():
    item = question()
    item["answer"] = ["unselected"]
    with pytest.raises(ValueError, match="disagree"):
        timestamp_options(item)


def test_feature_windows_use_exact_native_samples_and_preserve_polarity():
    signal = np.zeros((1000, 12))
    signal[100:120] = np.arange(20)[:, None] * np.arange(1, 13)[None, :]
    signal[200:220] = -signal[100:120]
    features, scalars = interval_features(signal, np.array([[1, 1.2], [2, 2.2]]), np.array([1.1, 2.1]))
    np.testing.assert_allclose(features[0], -features[1])
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1)
    assert features.shape == (2, 384)
    np.testing.assert_allclose(scalars[:, 0], np.log(0.2))
    np.testing.assert_allclose(scalars[:, 2], 0, atol=1e-12)


def test_support_draws_are_nested_and_patient_disjoint():
    rows = [{"patient_id": p} for p in range(1, 50)]
    for fold in range(3):
        first = support_patients(rows, "criterion", fold, 20261018, 1)
        third = support_patients(rows, "criterion", fold, 20261018, 3)
        assert first == third[:1]
        assert len(set(third)) == 3
        assert all(inner_fold(p) != fold for p in third)


def test_prototypes_weight_patients_equally_and_swap_only_reference_windows():
    rows = [
        {
            "patient_id": p,
            "answer": 0,
            "intervals": [[0, 1], [1, 2]],
            "features": np.array([first, [-1, -1]]),
            "scalars": np.ones((2, 5)),
        }
        for p, first in ((1, [1, 0]), (1, [1, 0]), (2, [0, 1]))
    ]
    vector, _ = prototype(rows, [1, 2])
    np.testing.assert_allclose(vector, [1 / np.sqrt(2), 1 / np.sqrt(2)])
    wrong, _ = prototype(rows, [1, 2], wrong=True)
    np.testing.assert_allclose(wrong, [-1 / np.sqrt(2), -1 / np.sqrt(2)])


def test_group_eligibility_reports_every_failure_without_scores():
    rows = [
        {"class_key": key, "patient_id": p}
        for key, count in (("a", 20), ("b", 20), ("c", 4))
        for p in range(1, count + 1)
    ]
    groups, inventory = class_groups(rows)
    assert list(groups) == ["a", "b"]
    assert [row["eligible"] for row in inventory] == [True, True, False]


def test_patient_macro_does_not_overweight_repeat_questions():
    rows = [{"patient_id": 1, "accuracy": {"m": 1}}] * 10 + [{"patient_id": 2, "accuracy": {"m": 0}}]
    patients, means = patient_macro(rows, "accuracy", "m")
    assert means.mean() == 0.5
    result = paired_interval(patients, means)
    assert result["difference"] == 0.5
    assert result["draws"] == 2000


def test_interval_iou_has_known_geometry():
    assert overlap([1, 2], [1, 2]) == 1
    assert overlap([1, 2], [2, 3]) == 0
    assert overlap([1, 3], [2, 4]) == pytest.approx(1 / 3)


def test_shared_manifest_mismatch_stops_before_scoring(tmp_path):
    from ecg_experiment.files import sha256_file

    metadata = tmp_path / "metadata.csv"
    metadata.write_text("identity metadata")
    partition = tmp_path / "partition.json"
    partition.write_text(
        json.dumps(
            {"metadata_sha256": sha256_file(metadata), "records": [{"ecg_id": "1", "role": "validation"}]}
        )
    )
    table = pd.DataFrame({"eligible": [True], "patient_id": [1]}, index=[1])
    with pytest.raises(ValueError, match="disagrees"):
        verify_partition(table, metadata, partition, tmp_path)
