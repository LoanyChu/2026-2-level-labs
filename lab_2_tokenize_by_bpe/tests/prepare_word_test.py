"""
Checks the second lab's prepare word function
"""

import pytest

from lab_2_tokenize_by_bpe.main import prepare_word


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark4
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_prepare_word_ideal() -> None:
    """
    Ideal prepare word scenario
    """
    expected = ("а", "л", "ь", "б", "а", "т", "р", "о", "с", "</s>")
    actual = prepare_word("альбатрос", None, "</s>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark4
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_prepare_word_with_start_of_word() -> None:
    """
    Prepare word scenario with start of word
    """
    expected = ("<sow>", "а", "л", "ь", "б", "а", "т", "р", "о", "с", "</s>")
    actual = prepare_word("альбатрос", "<sow>", "</s>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark4
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_prepare_word_bad_input() -> None:
    """
    Prepare word invalid inputs check
    """
    bad_raw_word_inputs = [{}, (), None, 1, 1.1, True, [None]]
    bad_start_end_inputs = [{}, (), 1, 1.1, True, [None]]
    expected = None
    for bad_input in bad_raw_word_inputs:
        actual = prepare_word(bad_input, None, "</s>")
        assert expected == actual

    for bad_input in bad_start_end_inputs:
        actual = prepare_word("альбатрос", bad_input, bad_input)
        assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark4
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_prepare_word_return_value() -> None:
    """
    Prepare word return value check
    """
    actual = prepare_word("альбатрос", None, "</s>")
    assert isinstance(actual, tuple)
