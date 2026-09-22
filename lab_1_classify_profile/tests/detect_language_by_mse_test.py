"""
Checks the first lab language detection function
"""

# pylint: disable=duplicate-code,disable=assignment-from-no-return

import pytest

from lab_1_classify_profile.main import detect_language_by_mse


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark8
@pytest.mark.mark10
def test_detect_language_by_mse_ideal() -> None:
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

    en_profile = (
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
    )

    de_profile = (
        "de",
        {
            "nein": 0.0666,
            "sonn": 0.0333,
            "allein": 0.0666,
            "made": 0.0666,
            "tee": 0.0666,
            "brot": 0.1333,
            "wein": 0.0666,
            "morgen": 0.1,
            "happy": 0.1666,
            "correct": 0.1666,
        },
        10,
    )

    actual = detect_language_by_mse(unknown_profile, en_profile, de_profile)
    assert "en" == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark8
@pytest.mark.mark10
def test_detect_language_by_mse_german_ideal() -> None:
    """
    Ideal scenario with german profile
    """

    unknown_profile = (
        "unk",
        {
            "morgen": 0.2222,
            "happy": 0.0555,
            "correct": 0.0555,
            "tee": 0.0555,
            "bingo": 0.0555,
            "wein": 0.0555,
            "lay": 0.0555,
            "klug": 0.0555,
            "brot": 0.1111,
            "word": 0.0555,
            "nein": 0.1111,
        },
        10,
    )

    en_profile = (
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
    )

    de_profile = (
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
    )

    actual = detect_language_by_mse(unknown_profile, en_profile, de_profile)
    assert "de" == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark8
@pytest.mark.mark10
def test_detect_language_by_mse_bad_input() -> None:
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

    en_profile = (
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
    )

    de_profile = (
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
    )

    bad_profiles = ["goodbye", (), None, 9, 9.34, True, [None], [], {"name": 1}]

    for bad_profile in bad_profiles:
        assert detect_language_by_mse(bad_profile, en_profile, de_profile) is None
        assert detect_language_by_mse(unknown_profile, bad_profile, de_profile) is None
        assert detect_language_by_mse(unknown_profile, en_profile, bad_profile) is None


def test_detect_language_by_mse_same_mse() -> None:
    """
    Languages have the same MSE value scenario.
    """
    unknown_profile = ("unknown", {"hola": 0.5, "hi": 0.5}, 2)

    en_profile = ("en", {"hi": 1.0}, 1)

    es_profile = ("es", {"hola": 1.0}, 1)

    assert detect_language_by_mse(unknown_profile, es_profile, en_profile) == "en"
