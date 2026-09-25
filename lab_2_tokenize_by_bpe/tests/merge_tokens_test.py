"""
Checks the second lab's merge tokens function
"""

# pylint: disable=redefined-outer-name, assignment-from-no-return
import pytest

from lab_2_tokenize_by_bpe.main import merge_tokens


@pytest.fixture(scope="function", autouse=True)
def setup() -> tuple:
    """
    Prepare test data.

    Returns:
        tuple: Correct data tuple.
    """
    word_frequencies = {
        ("В", "е", "з", "</s>"): 1,
        ("к", "о", "р", "а", "б", "л", "ь", "</s>"): 2,
        ("к", "а", "р", "а", "м", "е", "л", "ь", ",", "</s>"): 1,
        ("н", "а", "с", "к", "о", "ч", "и", "л", "</s>"): 1,
        ("н", "а", "</s>"): 2,
        ("м", "е", "л", "ь", ",", "</s>"): 1,
        ("м", "а", "т", "р", "о", "с", "ы", "</s>"): 1,
        ("д", "в", "е", "</s>"): 1,
        ("н", "е", "д", "е", "л", "и", "</s>"): 1,
        ("к", "а", "р", "а", "м", "е", "л", "ь", "</s>"): 1,
        ("м", "е", "л", "и", "</s>"): 1,
        ("е", "л", "и", ".", "</s>"): 1,
    }
    pair = ("е", "л")

    return word_frequencies, pair


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_merge_tokens_ideal(setup: tuple) -> None:
    """
    Ideal merge tokens scenario.

    Args:
        setup (tuple): Prepared test data.
    """
    word_frequencies, pair = setup

    expected = {
        ("В", "е", "з", "</s>"): 1,
        ("к", "о", "р", "а", "б", "л", "ь", "</s>"): 2,
        ("к", "а", "р", "а", "м", "ел", "ь", ",", "</s>"): 1,
        ("н", "а", "с", "к", "о", "ч", "и", "л", "</s>"): 1,
        ("н", "а", "</s>"): 2,
        ("м", "ел", "ь", ",", "</s>"): 1,
        ("м", "а", "т", "р", "о", "с", "ы", "</s>"): 1,
        ("д", "в", "е", "</s>"): 1,
        ("н", "е", "д", "ел", "и", "</s>"): 1,
        ("к", "а", "р", "а", "м", "ел", "ь", "</s>"): 1,
        ("м", "ел", "и", "</s>"): 1,
        ("ел", "и", ".", "</s>"): 1,
    }
    actual = merge_tokens(word_frequencies, pair)
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_merge_tokens_bad_input(setup: tuple) -> None:
    """
    Merge tokens invalid inputs check.

    Args:
        setup (tuple): Prepared test data.
    """
    word_frequencies, pair = setup

    word_frequencies_bad_input = ["string", (), None, 1, 1.1, True, [None]]
    pair_bad_input = ["string", None, 1, 1.1, True, [None], {}]
    expected = None
    for index, bad_input in enumerate(word_frequencies_bad_input):
        actual = merge_tokens(bad_input, pair)
        assert expected == actual

        actual = merge_tokens(word_frequencies, pair_bad_input[index])
        assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_merge_tokens_return_value(setup: tuple) -> None:
    """
    Merge tokens return value check.

    Args:
        setup (tuple): Prepared test data.
    """
    word_frequencies, pair = setup

    actual = merge_tokens(word_frequencies, pair)
    for key in actual:
        assert isinstance(actual[key], int)
        assert isinstance(key, tuple)
    assert isinstance(actual, dict)
