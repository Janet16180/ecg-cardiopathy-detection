"""Scientific coordinate, reference and target checks for Experiment 057."""

import numpy as np
import pytest

from ecg_experiment.st_episode057 import (
    FS,
    change_windows,
    episode_targets,
    location_summary,
    mean_interval,
    parse_episodes,
    subject_id,
)


def _waveform():
    signal = np.zeros((FS * 120, 2))
    anchors = np.arange(1, 119) * FS
    for anchor in anchors:
        signal[anchor - 5 : anchor + 6, 0] = np.linspace(0, 1, 11)
        signal[anchor - 5 : anchor + 6, 1] = np.linspace(0, -0.5, 11)
    return signal, anchors


def test_first_thirty_seconds_reference_preserves_later_st_changes():
    signal, anchors = _waveform()
    for anchor in anchors[anchors >= FS * 60]:
        signal[anchor + 30 : anchor + 40, 1] += 0.15
    features = change_windows(signal, anchors)
    np.testing.assert_allclose(features["st"][:, 1], [0, 0.15, 0.15], atol=1e-12)
    np.testing.assert_allclose(features["st"][:, 0], 0, atol=1e-12)
    np.testing.assert_array_equal(features["centers"], [45, 75, 105])
    assert features["whole"][-1, 1] < features["st"][-1, 1]


def test_constant_channel_offsets_do_not_change_scores():
    signal, anchors = _waveform()
    original = change_windows(signal, anchors)
    shifted = change_windows(signal + np.array([2.0, -3.0]), anchors)
    for name in ("st", "whole"):
        np.testing.assert_allclose(original[name], shifted[name], atol=1e-12)


def test_channels_permute_with_waveform_scores():
    signal, anchors = _waveform()
    signal[FS * 70 : FS * 71, 0] += 0.2
    original = change_windows(signal, anchors)
    permuted = change_windows(signal[:, ::-1], anchors)
    for name in ("st", "whole"):
        np.testing.assert_array_equal(original[name][:, ::-1], permuted[name])


def test_episode_case_and_channel_are_preserved():
    samples = np.array([40, 50, 60, 70, 80, 90]) * FS
    episodes = parse_episodes(samples, ["(ST0-", "AST0-200", "ST0-)", "(st1+", "ast1+120", "st1+)"])
    centers = np.array([45, 75])
    np.testing.assert_array_equal(episode_targets(centers, episodes, "ST"), [[True, False], [False, False]])
    np.testing.assert_array_equal(episode_targets(centers, episodes, "st"), [[False, False], [False, True]])
    assert episodes[0]["peaks"][0]["microvolts"] == 200


@pytest.mark.parametrize("notes", [["ST0-)"], ["(ST0-"], ["(ST0-", "AST0-garbage"]])
def test_malformed_or_unpaired_episodes_raise(notes):
    with pytest.raises(ValueError, match="episode"):
        parse_episodes(np.arange(len(notes)) * FS, notes)


def test_ties_and_deterministic_marks_are_reported_separately():
    found = location_summary(np.array([[2.0, 2.0], [0.0, 1.0]]), np.array([[False, True], [False, False]]))
    assert found == {"hit": 0.5, "displayed_hit": 0.0, "chance": 0.25, "ties": 2}


def test_patient_macro_does_not_overweight_repeated_records():
    values = np.array([1.0, 1.0, 1.0, 0.0])
    found = mean_interval(values, np.array(["one", "one", "one", "two"]))
    assert found["value"] == 0.5
    assert found["patients"] == 2
    assert found == mean_interval(values, np.array(["one", "one", "one", "two"]))
    assert subject_id("e0122") == "e0118"
    assert subject_id("e0121") == "e0118"
    assert subject_id("e0103") == "e0103"


def test_reference_failure_is_explicit():
    signal, _ = _waveform()
    with pytest.raises(ValueError, match="Fewer than three"):
        change_windows(signal, np.array([40, 50, 60]) * FS)


def test_nul_padding_and_unknown_terminal_end_are_explicit():
    notes = ["(ST0-\x00unrelated-padding", "AST0-200", "ST0-)", "(T1+", "AT1+300"]
    episodes = parse_episodes(np.arange(1, 6) * FS, notes, allow_censored=True)
    assert episodes[0]["end"] == 3
    assert episodes[1]["end"] is None
    assert episodes[1]["censored"] is True
    np.testing.assert_array_equal(
        episode_targets(np.array([2, 5]), episodes, "T"), [[False, False], [False, False]]
    )
