"""
Checks the second lab's train function
"""

# pylint: disable=redefined-outer-name, assignment-from-no-return
from unittest import mock

import pytest

from lab_2_tokenize_by_bpe.main import collect_frequencies, train


@pytest.fixture(scope="function", autouse=True)
def word_frequencies() -> dict:
    """
    Prepare test data.

    Returns:
        dict: Correct frequency data dictionary.
    """
    return collect_frequencies(
        "Вез корабль карамель, наскочил корабль на мель, "
        "матросы две недели карамель на мели ели.",
        None,
        "</s>",
    )


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_train_ideal(word_frequencies: dict) -> None:
    """
    Ideal train scenario.

    Args:
        word_frequencies (dict): Prepared test data.
    """
    expected = {
        ("В", "е", "з", "</s>"): 1,
        ("корабль</s>",): 2,
        ("карамель,</s>",): 1,
        ("на", "с", "ко", "ч", "и", "л", "</s>"): 1,
        ("на</s>",): 2,
        ("мель,</s>",): 1,
        ("м", "а", "т", "р", "о", "с", "ы", "</s>"): 1,
        ("д", "ве</s>"): 1,
        ("недели</s>",): 1,
        ("карамель</s>",): 1,
        ("мели</s>",): 1,
        ("ели.</s>",): 1,
    }
    actual = train(word_frequencies, 30)
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_train_none_word_frequencies(word_frequencies: dict) -> None:
    """
    Train with None as word_frequencies' return value.

    Args:
        word_frequencies (dict): Prepared test data.
    """
    expected = None
    with mock.patch("lab_2_tokenize_by_bpe.main.merge_tokens", return_value=None):
        actual = train(word_frequencies, 30)
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_train_stops_when_no_pairs_left() -> None:
    """
    Train should stop when there are no more pairs to merge,
    even if num_merges is not reached.
    """
    word_frequencies = {
        ("a", "b", "c", "</s>"): 1,
    }
    actual = train(word_frequencies, 100)
    expected = {
        ("abc</s>",): 1,
    }
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_train_bad_input(word_frequencies: dict) -> None:
    """
    Train invalid inputs check.

    Args:
        word_frequencies (dict): Prepared test data.
    """
    word_frequencies_bad_input = ["string", (), 1, 1.1, True, [None]]
    num_merges_bad_input = ["string", None, (), 1.1, [None], {}]
    expected = None
    for bad_input in word_frequencies_bad_input:
        actual = train(bad_input, 30)
        assert expected == actual
    for bad_input in num_merges_bad_input:
        actual = train(word_frequencies, bad_input)
        assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_train_return_value(word_frequencies: dict) -> None:
    """
    Train return value check.

    Args:
        word_frequencies (dict): Prepared test data.
    """
    actual = train(word_frequencies, 100)
    for key in actual:
        assert isinstance(actual[key], int)
        assert isinstance(key, tuple)
    assert isinstance(actual, dict)
