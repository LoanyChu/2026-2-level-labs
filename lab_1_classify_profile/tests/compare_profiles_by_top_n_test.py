"""
Checks the first lab language comparison function
"""

# pylint: disable=duplicate-code,assignment-from-no-return

import pytest
from pytest import MonkeyPatch

from admin_utils.constants import FLOAT_TOLERANCE
from lab_1_classify_profile.main import compare_profiles_by_top_n


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_top_n_ideal() -> None:
    """
    Ideal scenario
    """
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
            "man": 0.3111,
        },
        4,
    )

    actual = compare_profiles_by_top_n(en_profile, de_profile, 3)
    assert 0.3333 == pytest.approx(actual, abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_top_n_no_intersections_ideal() -> None:
    """
    Ideal scenario with no intersections
    """
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
        },
        6,
    )

    actual = compare_profiles_by_top_n(en_profile, de_profile, 4)
    assert 0.0 == pytest.approx(actual, abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_top_n_identical() -> None:
    """
    Ideal scenario with identical profiles
    """
    first_profile = (
        "en",
        {
            "happy": 0.2,
            "he": 0.1,
            "man": 0.1,
        },
        3,
    )

    second_profile = (
        "en",
        {
            "happy": 0.2,
            "he": 0.1,
            "man": 0.1,
        },
        3,
    )

    expected = 1.0
    actual = compare_profiles_by_top_n(first_profile, second_profile, 2)
    assert expected == pytest.approx(actual, abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_top_n_bad_input() -> None:
    """
    Bad input scenario
    """
    bad_profiles = [{}, (), [], 3.14, -9, "this is a profile, trust"]

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
        },
        7,
    )

    for bad_profile in bad_profiles:
        assert compare_profiles_by_top_n(bad_profile, de_profile, 4) is None
        assert compare_profiles_by_top_n(de_profile, bad_profile, 4) is None


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_top_n_bad_input_top_n() -> None:
    """
    Bad input scenario
    """
    en_profile = (
        "en",
        {
            "happy": 0.2,
            "he": 0.1,
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
        },
        7,
    )

    expected = None
    actual = compare_profiles_by_top_n(en_profile, de_profile, {})
    assert expected == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_top_n_bad_input_freq_type() -> None:
    """
    Bad input scenario
    """
    en_profile = (
        "en",
        {
            "happy": 0.5,
            "he": 0.25,
            "man": 0.25,
        },
        3,
    )

    bad_profile = ("de", [], 0)

    expected = None
    actual = compare_profiles_by_top_n(en_profile, bad_profile, 2)
    assert expected == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_top_n_invalid_top_n_words_output(monkeypatch: MonkeyPatch) -> None:
    """
    Bad function output scenario, invalid get_top_n_words

    Args:
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock get_top_n_words
    """

    monkeypatch.setattr("lab_1_classify_profile.main.get_top_n_words", lambda *args, **kwargs: None)

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
            "man": 0.3111,
        },
        4,
    )

    expected = None
    actual = compare_profiles_by_top_n(en_profile, de_profile, 3)
    assert expected == actual
