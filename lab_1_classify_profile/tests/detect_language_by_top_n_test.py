"""
Checks the first lab language detection function
"""

# pylint: disable=duplicate-code,disable=assignment-from-no-return

import pytest
from pytest import MonkeyPatch

from lab_1_classify_profile.main import detect_language_by_top_n


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_detect_language_by_top_n_ideal() -> None:
    """
    Ideal scenario
    """

    unknown_profile = (
        "unk",
        {
            "happy": 0.5,
            "she": 0.2,
            "man": 0.1,
        },
        3,
    )

    de_profile = (
        "de",
        {
            "ich": 0.3333,
            "weiß": 0.1111,
            "nicht": 0.1111,
            "machen": 0.1111,
            "möchte": 0.1111,
            "vielleicht": 0.1111,
            "überlegen": 0.1111,
            "man": 0.1111,
        },
        8,
    )

    en_profile = (
        "en",
        {
            "happy": 0.3333,
            "he": 0.1667,
            "man": 0.1667,
        },
        3,
    )

    actual = detect_language_by_top_n(unknown_profile, en_profile, de_profile, 2)
    assert "en" == actual

    actual = detect_language_by_top_n(unknown_profile, de_profile, en_profile, 2)
    assert "en" == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_detect_language_by_top_n_german_ideal() -> None:
    """
    Ideal scenario with german profile
    """

    unknown_profile = (
        "unk",
        {
            "weiß": 0.5,
            "überlegen": 0.2,
            "nicht": 0.1,
        },
        3,
    )

    de_profile = (
        "de",
        {
            "ich": 0.3333,
            "weiß": 0.1111,
            "nicht": 0.1111,
            "machen": 0.1111,
            "möchte": 0.1111,
            "vielleicht": 0.1111,
            "überlegen": 0.1111,
            "man": 0.1111,
        },
        8,
    )

    en_profile = (
        "en",
        {"happy": 0.2, "he": 0.1, "man": 0.1},
        3,
    )

    actual = detect_language_by_top_n(unknown_profile, en_profile, de_profile, 2)
    assert de_profile[0] == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_detect_language_by_top_n_alphabetical() -> None:
    """
    Detect language when scores are the same
    """

    unknown_profile = (
        "unk",
        {
            "computer": 0.5,
            "the": 0.2,
            "world": 0.1,
        },
        3,
    )

    en_profile = (
        "en",
        {
            "computer": 0.2,
            "she": 0.1,
            "woman": 0.1,
        },
        3,
    )

    de_profile = (
        "de",
        {
            "sie": 0.3,
            "haben": 0.1,
            "viel": 0.1,
            "computer": 0.2,
        },
        4,
    )

    actual = detect_language_by_top_n(unknown_profile, en_profile, de_profile, 2)
    assert de_profile[0] == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_detect_language_by_top_n_bad_input() -> None:
    """
    Bad input scenario
    """

    profile = (
        "en",
        {
            "computer": 0.2,
            "she": 0.1,
            "woman": 0.1,
        },
        3,
    )

    de_profile = (
        "de",
        {
            "ich": 0.3333,
            "weiß": 0.1111,
            "nicht": 0.1111,
            "machen": 0.1111,
            "möchte": 0.1111,
            "vielleicht": 0.1111,
            "überlegen": 0.1111,
            "man": 0.1111,
        },
        8,
    )

    bad_ns = [{}, set(), (), [], 3.14, "string", None]
    for bad_n in bad_ns:
        assert detect_language_by_top_n(profile, profile, de_profile, bad_n) is None


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_detect_language_by_top_n_bad_input_profile() -> None:
    """
    Bad input scenario
    """

    unknown_profile = (
        "de",
        {
            "ich": 0.3,
            "weiß": 0.1,
            "nicht": 0.1,
            "machen": 0.1,
            "möchte": 0.1,
            "vielleicht": 0.1,
            "überlegen": 0.1,
            "man": 0.1,
        },
        8,
    )

    de_profile = (
        "de",
        {
            "ich": 0.3333,
            "weiß": 0.1111,
            "nicht": 0.1111,
            "machen": 0.1111,
            "möchte": 0.1111,
            "vielleicht": 0.1111,
            "überlegen": 0.1111,
            "man": 0.1111,
        },
        8,
    )

    bad_profiles = ["goodbye", (), None, 9, 9.34, True, [None], [], {"name": 1}]
    for bad_profile in bad_profiles:
        assert detect_language_by_top_n(bad_profile, de_profile, de_profile, 2) is None
        assert detect_language_by_top_n(unknown_profile, bad_profile, de_profile, 2) is None
        assert detect_language_by_top_n(unknown_profile, de_profile, bad_profile, 2) is None


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_detect_language_by_top_n_invalid_compare_profiles_by_top_n(
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Bad function output scenario, invalid compare_profiles_by_top_n

    Args:
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock compare_profiles_by_top_n
    """
    monkeypatch.setattr(
        "lab_1_classify_profile.main.compare_profiles_by_top_n", lambda *args, **kwargs: None
    )

    unknown_profile = (
        "unk",
        {"happy": 0.5, "she": 0.2, "man": 0.1},
        3,
    )
    en_profile = (
        "en",
        {
            "happy": 0.3333,
            "he": 0.1667,
            "man": 0.1667,
        },
        3,
    )
    de_profile = (
        "de",
        {
            "ich": 0.3333,
            "weiß": 0.1111,
            "nicht": 0.1111,
            "machen": 0.1111,
            "möchte": 0.1111,
            "vielleicht": 0.1111,
            "man": 0.1111,
        },
        7,
    )

    actual = detect_language_by_top_n(unknown_profile, en_profile, de_profile, 2)
    assert actual is None
