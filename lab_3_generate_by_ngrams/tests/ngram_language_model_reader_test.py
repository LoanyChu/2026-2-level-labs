# pylint: disable=protected-access
"""
Checks the third lab's NGramLanguageModelReader class
"""

import pytest
from pytest import MonkeyPatch

from lab_3_generate_by_ngrams.main import NGramLanguageModelReader
from lab_3_generate_by_ngrams.tests.constants import PATH_TO_ASSETS


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_language_model_reader_fields(language_model_reader: NGramLanguageModelReader) -> None:
    """
    Checks if NGramLanguageModelReader fields are created correctly

    Args:
        language_model_reader (NGramLanguageModelReader): An instance of NGramLanguageModelReader
    """
    assert language_model_reader._json_path == str(PATH_TO_ASSETS / "en_for_test.json")
    assert language_model_reader._eow_token == "_"


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_ngram_language_model_reader_load(language_model_reader: NGramLanguageModelReader) -> None:
    """
    Checks NGramLanguageModelReader load method ideal scenario

    Args:
        language_model_reader (NGramLanguageModelReader): An instance of NGramLanguageModelReader
    """
    actual = language_model_reader.load(2)
    expected_ngram_frequencies = {(0, 0): 1.0}
    assert 2 == actual.get_n_gram_size()
    for key, value in expected_ngram_frequencies.items():
        assert expected_ngram_frequencies.keys() == actual._n_gram_frequencies.keys()
        assert pytest.approx(value, abs=0.0001) == actual._n_gram_frequencies[key]


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_ngram_language_model_reader_load_invalid_input(
    language_model_reader: NGramLanguageModelReader,
) -> None:
    """
    Checks NGramLanguageModelReader load method with invalid input

    Args:
        language_model_reader (NGramLanguageModelReader): An instance of NGramLanguageModelReader
    """
    expected = None
    bad_inputs = [[None], {}, None, (), 1.1, "string"]
    for bad_input in bad_inputs:
        actual = language_model_reader.load(bad_input)
        assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_ngram_language_model_reader_load_unigram(
    language_model_reader: NGramLanguageModelReader,
) -> None:
    """
    Checks NGramLanguageModelReader load method with unigrams

    Args:
        language_model_reader (NGramLanguageModelReader): An instance of NGramLanguageModelReader
    """
    actual = language_model_reader.load(1)
    expected = None
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_ngram_language_model_reader_get_text_processor(
    language_model_reader: NGramLanguageModelReader,
) -> None:
    """
    Checks NGramLanguageModelReader get_text_processor method ideal scenario

    Args:
        language_model_reader (NGramLanguageModelReader): An instance of NGramLanguageModelReader
    """
    actual = language_model_reader.get_text_processor()
    expected = {
        "_": 0,
        "a": 3,
        "b": 10,
        "c": 5,
        "d": 8,
        "e": 2,
        "f": 9,
        "g": 4,
        "h": 15,
        "i": 16,
        "j": 1,
        "k": 17,
        "l": 11,
        "m": 12,
        "n": 13,
        "o": 14,
        "p": 23,
        "q": 22,
        "r": 6,
        "s": 7,
        "t": 19,
        "u": 18,
        "v": 21,
        "w": 20,
        "x": 25,
        "y": 24,
        "z": 26,
        "á": 29,
        "ä": 28,
        "é": 27,
        "ö": 31,
        "ü": 30,
    }
    assert expected == actual._storage


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_ngram_language_model_reader_load_none_token_id(
    language_model_reader: NGramLanguageModelReader, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks NGramLanguageModelReader load method with invalid input

    Args:
        language_model_reader (NGramLanguageModelReader): An instance of NGramLanguageModelReader
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock get_id
    """
    monkeypatch.setattr(
        language_model_reader._text_processor,
        "get_id",
        lambda *args, **kwargs: None,
    )

    actual = language_model_reader.load(2)
    assert actual is not None
    assert actual._n_gram_frequencies == {}
