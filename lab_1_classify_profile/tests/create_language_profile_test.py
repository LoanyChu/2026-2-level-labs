"""
Checks the first lab language profile creation function
"""

import pytest
from pytest import MonkeyPatch

from admin_utils.constants import FLOAT_TOLERANCE
from lab_1_classify_profile.main import create_language_profile

STOP_WORDS_EN = ["the", "a", "is"]
STOP_WORDS_DE = ["muss", "das", "was"]


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_create_profile_ideal() -> None:
    """
    Ideal scenario
    """
    expected = ("en", {"happy": 0.5, "he": 0.25, "man": 0.25}, 3)
    language_name = "en"
    text = "he is a happy happy man"
    actual = create_language_profile(language_name, text, STOP_WORDS_EN)
    assert isinstance(actual, tuple)
    assert actual[0] == expected[0]
    assert actual[2] == expected[2]
    for token, freq in actual[1].items():
        assert freq == pytest.approx(expected[1][token], abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_create_profile_no_stop_words() -> None:
    """
    Ideal scenario with no stop words
    """
    expected = (
        "en",
        {
            "happy": 0.3333,
            "he": 0.1667,
            "man": 0.1667,
            "is": 0.1667,
            "a": 0.1667,
        },
        5,
    )
    language_name = "en"
    text = "he is a happy happy man"
    actual = create_language_profile(language_name, text, [])
    assert isinstance(actual, tuple)
    assert actual[0] == expected[0]
    assert actual[2] == expected[2]
    for token, freq in actual[1].items():
        assert freq == pytest.approx(expected[1][token], abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_create_profile_german_ideal() -> None:
    """
    Ideal scenario with german profile
    """
    expected = (
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
    language_name = "de"
    text = "Ich weiß nicht was ich machen möchte. Vielleicht ich muss das überlegen"
    actual = create_language_profile(language_name, text, STOP_WORDS_DE)
    assert isinstance(actual, tuple)
    assert actual[0] == expected[0]
    assert actual[2] == expected[2]
    for token, freq in actual[1].items():
        assert freq == pytest.approx(expected[1][token], abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_create_profile_bad_input() -> None:
    """
    Bad input scenario
    """
    expected = None
    language_name = "de"
    text = []
    actual = create_language_profile(language_name, text, STOP_WORDS_DE)
    assert expected == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_create_profile_invalid_remove_stop_words_none_output(monkeypatch: MonkeyPatch) -> None:
    """
    Bad function output scenario, invalid remove_stop_words

    Args:
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock remove_stop_words
    """
    monkeypatch.setattr(
        "lab_1_classify_profile.main.remove_stop_words", lambda *args, **kwargs: None
    )
    expected = None
    language_name = "de"
    text = "Ich weiß nicht was ich machen möchte. Vielleicht ich muss das überlegen"
    actual = create_language_profile(language_name, text, STOP_WORDS_DE)
    assert expected == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_create_profile_invalid_calculate_frequencies_none_output(monkeypatch: MonkeyPatch) -> None:
    """
    Bad function output scenario, invalid calculate_frequencies

    Args:
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock calculate_frequencies
    """
    monkeypatch.setattr(
        "lab_1_classify_profile.main.calculate_frequencies", lambda *args, **kwargs: None
    )
    expected = None
    language_name = "de"
    text = "Ich weiß nicht was ich machen möchte. Vielleicht ich muss das überlegen"
    actual = create_language_profile(language_name, text, STOP_WORDS_DE)
    assert expected == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_create_profile_bad_input_lang_name() -> None:
    """
    Bad input scenario, language name
    """
    expected = None
    language_name = 123
    text = "Ich weiß nicht was ich machen möchte. Vielleicht ich muss das überlegen"
    actual = create_language_profile(language_name, text, STOP_WORDS_DE)
    assert expected == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_create_profile_bad_input_stop_words() -> None:
    """
    Bad input scenario, stop words
    """
    expected = None
    language_name = "de"
    text = "Ich weiß nicht was ich machen möchte. Vielleicht ich muss das überlegen"
    actual = create_language_profile(language_name, text, None)
    assert expected == actual
