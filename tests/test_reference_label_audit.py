"""Scientific quarantine and answer-schema checks for NTT-01."""

import json

import pandas as pd
import pytest

from ecg_experiment.reference_label_audit import audit_local_labels, audit_reasoning, grounding_summary
from scripts.reports import audit_reference_labels as runner


def grounding(indices=(0,), options=("[1.00s - 1.20s]", "[2.00s - 2.20s]")):
    return {
        "question_type": "wave_grounding",
        "options": list(options),
        "answer_idx": list(indices),
        "answer": [options[i] for i in indices],
    }


def test_quarantine_skips_poisoned_clinical_payload_before_access(tmp_path):
    rows = [
        {"metadata": {"id": i, "data_source": "ptbxl", "ecg_id": str(i)}, "data": None} for i in (9, 10, 999)
    ]
    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    ids = pd.DataFrame({"patient_id": [91, 101], "strat_fold": [9, 10]}, index=[9, 10])
    result = audit_reasoning(path, ids)
    assert result["quarantined_cases"] == 2
    assert result["unknown_ecg_cases"] == 1
    assert result["safe_cases"] == 0
    assert result["safe_diagnosis_cases"] == {}


@pytest.mark.parametrize("value", [None, 1.9, True, "1.9", "", -1])
def test_invalid_ecg_ids_are_counted_before_payload_access(tmp_path, value):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"metadata": {"id": 0, "data_source": "ptbxl", "ecg_id": value}}))
    ids = pd.DataFrame({"patient_id": [11], "strat_fold": [1]}, index=[1])
    result = audit_reasoning(path, ids)
    assert result["invalid_ecg_identifiers"] == 1
    assert result["safe_cases"] == 0


def test_missing_case_id_is_counted_before_payload_access(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({"metadata": {"data_source": "ptbxl", "ecg_id": "1"}}))
    ids = pd.DataFrame({"patient_id": [11], "strat_fold": [1]}, index=[1])
    result = audit_reasoning(path, ids)
    assert result["invalid_case_identifiers"] == 1
    assert result["safe_cases"] == 0


def test_repeated_cases_are_counted_without_becoming_independent_ecgs(tmp_path):
    question = grounding()
    rows = [
        {
            "metadata": {
                "id": i,
                "data_source": "ptbxl",
                "ecg_id": "1",
                "target_dx": "example",
                "dx_label": True,
            },
            "data": {
                "initial_diagnostic_question": {"question_type": "initial_diagnostic_question"},
                "reasoning": [{"ecg_grounding": [question]}],
            },
        }
        for i in (1, 2)
    ]
    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    ids = pd.DataFrame({"patient_id": [11], "strat_fold": [1]}, index=[1])
    result = audit_reasoning(path, ids)
    assert result["safe_cases"] == 2
    assert result["safe_unique_ecgs"] == result["safe_unique_patients"] == 1
    assert result["safe_repeat_cases_beyond_unique_ecg"] == 1


def test_multiple_answers_are_excluded_from_position_baseline():
    result = grounding_summary([grounding(), grounding((1,)), grounding((0, 1)), grounding(())])
    assert result["single_answers"] == 2
    assert result["multiple_answers"] == result["empty_answers"] == 1
    assert result["option_zero_exact_accuracy"] == result["uniform_position_expectation"] == 0.5
    assert result["shortcut_flag"] is False


@pytest.mark.parametrize("indices", [[2], [-1], [True], [0, 0]])
def test_invalid_answer_indices_do_not_enter_baselines(indices):
    question = grounding()
    question["answer_idx"] = indices
    result = grounding_summary([question])
    assert result["invalid_answer_schema"] == 1
    assert result["single_answers"] == 0


@pytest.mark.parametrize("key", ["options", "answer", "answer_idx"])
def test_missing_answer_fields_are_counted(key):
    question = grounding()
    question.pop(key)
    assert grounding_summary([question])["invalid_answer_schema"] == 1


def test_non_string_options_are_counted_without_range_parsing():
    question = grounding()
    question["options"][0] = question["answer"][0] = 1
    assert grounding_summary([question])["invalid_answer_schema"] == 1


@pytest.mark.parametrize("value", ["[1..0s - 2s]", "[2s - 1s]", "[9s - 11s]", "[1ms - 2ms]"])
def test_malformed_or_out_of_record_ranges_are_counted(value):
    result = grounding_summary([grounding(options=(value,))])
    assert result["invalid_selected_time_ranges"] == 1
    assert result["selected_time_ranges"] == 0


def test_existing_labels_and_independent_logic_agree_without_opening_closed_codes(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {"patient_id": 1, "strat_fold": 1, "scp_codes": "{'SR': 0, 'MI': 100}"},
            {"patient_id": 2, "strat_fold": 1, "scp_codes": "{'NORM': 100, 'SR': 0}"},
            {"patient_id": 3, "strat_fold": 1, "scp_codes": "{'NORM': 100, 'MI': 100}"},
            {"patient_id": 4, "strat_fold": 9, "scp_codes": "MUST NOT BE PARSED"},
        ]
    ).to_csv(metadata, index=False)
    statements = tmp_path / "statements.csv"
    pd.DataFrame({"diagnostic": [1, 1], "diagnostic_class": ["NORM", "MI"]}, index=["NORM", "MI"]).to_csv(
        statements
    )
    result = audit_local_labels(metadata, statements)
    assert result["label_integrity_exact"] is True
    assert result["standard_labels"] == {"positive": 2, "negative": 1, "undefined": 0}
    assert result["strict_project_labels"] == {"positive": 1, "negative": 1, "undefined": 1}
    assert result["sr_with_diagnostic_abnormal"] == result["norm_with_diagnostic_abnormal"] == 1


def test_runner_preserves_failure_and_refuses_reusing_run_directory(tmp_path, monkeypatch):
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(output)])
    monkeypatch.setattr(runner, "to_stored", lambda path: path.name)

    def fail(_):
        raise ValueError("untrusted source")

    monkeypatch.setattr(runner, "run_audit", fail)
    with pytest.raises(ValueError, match="untrusted source"):
        runner.main()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "result.json").exists()
    with pytest.raises(FileExistsError):
        runner.main()
