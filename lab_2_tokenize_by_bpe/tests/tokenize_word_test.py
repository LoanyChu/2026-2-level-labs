"""
Checks the second lab's tokenize word function
"""

# pylint: disable=redefined-outer-name, assignment-from-no-return
import json
from pathlib import Path

import pytest

from lab_2_tokenize_by_bpe.main import tokenize_word


@pytest.fixture(scope="function", autouse=True)
def setup() -> tuple:
    """
    Setup for test
    """
    path_to_tests_directory = Path(__file__).parent
    with open(path_to_tests_directory / "vocabulary.json", "r", encoding="utf-8") as json_file:
        vocabulary = json.load(json_file)

    ideal_word = ("а", "л", "ь", "б", "а", "т", "р", "о", "с", "ы", "</s>")
    word_with_unk = ("a", "l", "c", "a", "t", "r", "a", "z", "</s>")

    return vocabulary, ideal_word, word_with_unk


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_tokenize_word_ideal(setup: tuple) -> None:
    """
    Ideal tokenize word scenario.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, ideal_word, _ = setup
    expected = [0, 186, 29]
    actual = tokenize_word(ideal_word, vocabulary, "</s>", "<unk>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_tokenize_word_with_unk(setup: tuple) -> None:
    """
    Tokenize word scenario with unknown token.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, _, word_with_unk = setup
    expected = [129, 134, 130, 129, 140, 138, 129, 16, 34]
    actual = tokenize_word(word_with_unk, vocabulary, "</s>", "<unk>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_tokenize_word_bad_input(setup: tuple) -> None:
    """
    Tokenize word invalid inputs check.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, ideal_word, _ = setup
    word_bad_input = ["string", [None], {}, None, 1, 1.1, True]
    vocabulary_bad_input = [None, (), 1.1, True, [None], "string", 1]
    end_of_word_bad_input = [(), {}, 1, 1.1, True, [None]]
    unknown_bad_input = [(), {}, None, 1, 1.1, True, [None]]
    expected = None
    for index, bad_input in enumerate(word_bad_input):
        actual = tokenize_word(bad_input, vocabulary, "</s>", "<unk>")
        assert expected == actual

        actual = tokenize_word(ideal_word, vocabulary_bad_input[index], "</s>", "<unk>")
        assert expected == actual

        actual = tokenize_word(ideal_word, vocabulary, "</s>", unknown_bad_input[index])
        assert expected == actual

    for bad_input in end_of_word_bad_input:
        actual = tokenize_word(ideal_word, vocabulary, bad_input, "<unk>")
        assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_tokenize_word_return_value(setup: tuple) -> None:
    """
    Tokenize word return value check.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, ideal_word, _ = setup
    actual = tokenize_word(ideal_word, vocabulary, "</s>", "<unk>")
    assert isinstance(actual, list)
