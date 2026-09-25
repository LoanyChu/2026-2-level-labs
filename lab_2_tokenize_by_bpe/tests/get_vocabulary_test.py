"""
Checks the second lab's get vocabulary function
"""

# pylint: disable=redefined-outer-name, assignment-from-no-return
import pytest

from lab_2_tokenize_by_bpe.main import get_vocabulary


@pytest.fixture(scope="function", autouse=True)
def word_frequencies() -> dict:
    """
    Prepare test data.

    Returns:
        dict: Correct data dict.
    """
    return {
        ("Часовня</s>",): 1,
        ("окружена</s>",): 1,
        ("низким</s>",): 1,
        ("белым</s>",): 1,
        ("заборчиком,</s>",): 1,
        ("который</s>",): 1,
        ("должен</s>",): 1,
        ("бы</s>",): 1,
        ("преграждать</s>",): 1,
        ("сюда</s>",): 1,
        ("путь</s>",): 1,
        ("альбатросам.</s>",): 1,
    }


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark8
@pytest.mark.mark10
def test_get_vocabulary_ideal(word_frequencies: dict) -> None:
    """
    Ideal get vocabulary scenario.

    Args:
        word_frequencies (dict): Prepared test data.
    """
    expected = {
        "альбатросам.</s>": 0,
        "заборчиком,</s>": 1,
        "преграждать</s>": 2,
        "окружена</s>": 3,
        "Часовня</s>": 4,
        "который</s>": 5,
        "должен</s>": 6,
        "низким</s>": 7,
        "белым</s>": 8,
        "путь</s>": 9,
        "сюда</s>": 10,
        "бы</s>": 11,
        "<unk>": 12,
        ",": 13,
        ".": 14,
        "/": 15,
        "<": 16,
        ">": 17,
        "s": 18,
        "Ч": 19,
        "а": 20,
        "б": 21,
        "в": 22,
        "г": 23,
        "д": 24,
        "е": 25,
        "ж": 26,
        "з": 27,
        "и": 28,
        "й": 29,
        "к": 30,
        "л": 31,
        "м": 32,
        "н": 33,
        "о": 34,
        "п": 35,
        "р": 36,
        "с": 37,
        "т": 38,
        "у": 39,
        "ч": 40,
        "ы": 41,
        "ь": 42,
        "ю": 43,
        "я": 44,
    }

    actual = get_vocabulary(word_frequencies, unknown_token="<unk>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark8
@pytest.mark.mark10
def test_get_vocabulary_bad_input(word_frequencies: dict) -> None:
    """
    Get vocabulary invalid inputs check.

    Args:
        word_frequencies (dict): Prepared test data.
    """
    word_frequencies_bad_input = ["string", (), None, 1, 1.1, True, [None]]
    unknown_token_bad_input = [None, (), 1.1, True, [None], {}, 1]
    expected = None
    for index, bad_input in enumerate(word_frequencies_bad_input):
        actual = get_vocabulary(bad_input, "<unk>")
        assert expected == actual

        actual = get_vocabulary(word_frequencies, unknown_token_bad_input[index])
        assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark8
@pytest.mark.mark10
def test_get_vocabulary_return_value(word_frequencies: dict) -> None:
    """
    Get vocabulary return value check.

    Args:
        word_frequencies (dict): Prepared test data.
    """
    actual = get_vocabulary(word_frequencies, "<unk>")
    for key in actual:
        assert isinstance(actual[key], int)
        assert isinstance(key, str)
    assert isinstance(actual, dict)
