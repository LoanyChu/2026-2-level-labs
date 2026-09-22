"""
Checks the first lab language detection function
"""

# pylint: disable=duplicate-code,assignment-from-no-return

import pytest
from pytest import MonkeyPatch

from admin_utils.constants import FLOAT_TOLERANCE
from lab_1_classify_profile.main import detect_language_advanced

known_profile = [
    (
        "de",
        {
            "tee": 0.0833,
            "happy": 0.1666,
            "nein": 0.0833,
            "wein": 0.0833,
            "klug": 0.0833,
            "morgen": 0.0833,
            "correct": 0.1666,
            "brot": 0.25,
        },
        8,
    ),
    (
        "en",
        {
            "popular": 0.2,
            "yes": 0.1,
            "morgen": 0.1,
            "happy": 0.2,
            "allein": 0.2,
            "made": 0.1,
            "nein": 0.1,
        },
        7,
    ),
]


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_detect_language_advanced_ideal() -> None:
    """
    Ideal scenario
    """

    unknown_profile = (
        "unk",
        {
            "made": 0.0909,
            "morgen": 0.0909,
            "happy": 0.1818,
            "popular": 0.1818,
            "yes": 0.0909,
            "sonn": 0.0909,
            "nein": 0.0909,
            "allein": 0.1818,
        },
        8,
    )

    expected = [("en", {"MSE": 0.0012, "Top-N": 1.0}), ("de", {"MSE": 0.0156, "Top-N": 0.3333})]
    actual = detect_language_advanced(unknown_profile, known_profile, 3)

    for expected_tuple_with_distance, actual_tuple_with_distance in zip(expected, actual):
        assert expected_tuple_with_distance[0] == actual_tuple_with_distance[0]
        assert actual_tuple_with_distance[1]["MSE"] == pytest.approx(
            expected_tuple_with_distance[1]["MSE"], abs=FLOAT_TOLERANCE
        )
        assert actual_tuple_with_distance[1]["Top-N"] == pytest.approx(
            expected_tuple_with_distance[1]["Top-N"], abs=FLOAT_TOLERANCE
        )


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_detect_language_advanced_bad_input() -> None:
    """
    Bad input scenario
    """
    unknown_profile = (
        "unk",
        {
            "made": 0.0909,
            "morgen": 0.0909,
            "happy": 0.1818,
            "popular": 0.1818,
            "yes": 0.0909,
            "sonn": 0.0909,
            "nein": 0.0909,
            "allein": 0.1818,
        },
        8,
    )

    bad_unknown_profiles = ["", {}, (), [], 3.14, -9, None]
    bad_known_profiles = [{}, 3.14, -9, None, True]

    for bad_unknown in bad_unknown_profiles:
        assert detect_language_advanced(bad_unknown, known_profile, 5) is None
    for bad_known in bad_known_profiles:
        assert detect_language_advanced(unknown_profile, bad_known, 5) is None


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_detect_language_advanced_invalid_known_profile_element() -> None:
    """
    Bad input scenario, invalid known profile element
    """
    unknown_profile = ("unk", {"made": 0.0909}, 8)
    bad_known_profiles = ["this_is_an_invalid_profile_string"]

    actual = detect_language_advanced(unknown_profile, bad_known_profiles, 3)
    assert actual is None


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_detect_language_advanced_invalid_check_profile(monkeypatch: MonkeyPatch) -> None:
    """
    Bad function output scenario, invalid check_profile

    Args:
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock check_profile
    """
    monkeypatch.setattr("lab_1_classify_profile.main.check_profile", lambda *args, **kwargs: None)
    unknown_profile = (
        "unk",
        {
            "made": 0.0909,
            "morgen": 0.0909,
            "happy": 0.1818,
            "popular": 0.1818,
            "yes": 0.0909,
            "sonn": 0.0909,
            "nein": 0.0909,
            "allein": 0.1818,
        },
        8,
    )
    expected = None
    actual = detect_language_advanced(unknown_profile, known_profile, 3)
    assert actual is expected
