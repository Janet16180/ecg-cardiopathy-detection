import pytest
from scripts.validation.audit_public_pool_overlap import classify_overlap


def test_same_identity_under_different_record_id_cannot_append():
    rows=[{'ecg_id':'new:a','source':'new','signal_sha256':'x'},
          {'ecg_id':'same:id','source':'new','signal_sha256':'changed'},
          {'ecg_id':'new:b','source':'new','signal_sha256':'y'}]
    novel,overlap=classify_overlap(rows,{'ptbxl_all_folds':{'x'},'mimic':set()},{'same:id'})
    assert [r['ecg_id'] for r in novel]==['new:b']
    assert overlap[0]['exact_signal_reference_pools']=='ptbxl_all_folds'
    assert overlap[1]['existing_record_id']=='true'


def test_duplicates_in_candidate_fail_gate():
    with pytest.raises(ValueError,match='duplicate identity'):
        classify_overlap([{'ecg_id':'a','signal_sha256':'x'}, {'ecg_id':'b','signal_sha256':'x'}],{},set())
