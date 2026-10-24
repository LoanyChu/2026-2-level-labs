"""
Checks the third lab's NGramLanguageModel class
"""

# pylint: disable=protected-access, no-member

import pytest
from pytest import MonkeyPatch

from lab_3_generate_by_ngrams.main import NGramLanguageModel

EXTRACTED_NGRAMS = (
    (1, 2),
    (2, 3),
    (3, 1),
    (1, 0),
    (0, 4),
    (4, 2),
    (2, 5),
    (5, 2),
    (2, 6),
    (6, 7),
    (7, 1),
    (1, 8),
    (8, 9),
    (9, 5),
    (5, 0),
    (0, 8),
    (8, 10),
    (10, 0),
    (0, 1),
    (1, 11),
    (11, 2),
    (2, 0),
    (0, 1),
    (1, 7),
    (7, 10),
    (10, 12),
    (12, 0),
    (0, 9),
    (9, 13),
    (13, 0),
    (0, 4),
    (4, 2),
    (2, 5),
    (5, 2),
    (2, 6),
    (6, 7),
    (7, 1),
    (1, 8),
    (8, 5),
    (5, 4),
    (4, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 1),
    (1, 0),
    (0, 14),
    (14, 8),
    (8, 1),
    (1, 11),
    (11, 0),
    (0, 1),
    (1, 11),
    (11, 2),
    (2, 0),
    (0, 4),
    (4, 9),
    (9, 7),
    (7, 15),
    (15, 0),
    (0, 9),
    (9, 13),
    (13, 0),
    (0, 7),
    (7, 16),
    (16, 16),
    (16, 2),
    (2, 7),
    (7, 6),
    (6, 8),
    (8, 5),
    (5, 4),
    (4, 0),
    (0, 8),
    (8, 5),
    (5, 17),
    (17, 8),
    (8, 10),
    (10, 1),
    (1, 8),
    (8, 5),
    (5, 4),
    (4, 18),
    (18, 8),
    (8, 10),
    (10, 11),
    (11, 7),
    (7, 19),
    (19, 15),
    (15, 2),
    (2, 0),
    (0, 1),
    (1, 9),
    (9, 0),
    (0, 11),
    (11, 18),
    (18, 20),
    (20, 7),
    (7, 5),
    (5, 14),
    (14, 6),
    (6, 8),
    (8, 1),
    (1, 1),
    (1, 2),
    (2, 5),
    (5, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 1),
    (1, 0),
)

NGRAM_FREQUENCIES = {
    (1, 2): 0.25,
    (2, 3): 0.25,
    (3, 1): 1.0,
    (1, 0): 0.1875,
    (0, 4): 0.1875,
    (4, 2): 0.3333,
    (2, 5): 0.25,
    (5, 2): 0.2222,
    (2, 6): 0.1667,
    (6, 7): 0.5,
    (7, 1): 0.25,
    (1, 8): 0.1875,
    (8, 9): 0.1,
    (9, 5): 0.2,
    (5, 0): 0.2222,
    (0, 8): 0.125,
    (8, 10): 0.3,
    (10, 0): 0.25,
    (0, 1): 0.375,
    (1, 11): 0.1875,
    (11, 2): 0.4,
    (2, 0): 0.25,
    (1, 7): 0.0625,
    (7, 10): 0.125,
    (10, 12): 0.25,
    (12, 0): 1.0,
    (0, 9): 0.125,
    (9, 13): 0.4,
    (13, 0): 1.0,
    (8, 5): 0.4,
    (5, 4): 0.3333,
    (4, 0): 0.3333,
    (0, 14): 0.0625,
    (14, 8): 0.5,
    (8, 1): 0.2,
    (11, 0): 0.2,
    (4, 9): 0.1667,
    (9, 7): 0.2,
    (7, 15): 0.125,
    (15, 0): 0.5,
    (0, 7): 0.0625,
    (7, 16): 0.125,
    (16, 16): 0.5,
    (16, 2): 0.5,
    (2, 7): 0.0833,
    (7, 6): 0.125,
    (6, 8): 0.5,
    (5, 17): 0.1111,
    (17, 8): 1.0,
    (10, 1): 0.25,
    (4, 18): 0.1667,
    (18, 8): 0.5,
    (10, 11): 0.25,
    (11, 7): 0.2,
    (7, 19): 0.125,
    (19, 15): 1.0,
    (15, 2): 0.5,
    (1, 9): 0.0625,
    (9, 0): 0.2,
    (0, 11): 0.0625,
    (11, 18): 0.2,
    (18, 20): 0.5,
    (20, 7): 1.0,
    (7, 5): 0.125,
    (5, 14): 0.1111,
    (14, 6): 0.5,
    (1, 1): 0.0625,
}


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_fields(language_model: NGramLanguageModel) -> None:
    """
    Checks if NGramLanguageModel fields are created correctly

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
    """
    assert 2 == language_model._NGramLanguageModel__n_gram_size

    language_model.build()
    actual = language_model._n_gram_frequencies
    assert NGRAM_FREQUENCIES.keys() == actual.keys()
    for key in actual.keys():
        assert NGRAM_FREQUENCIES[key] == pytest.approx(actual[key], abs=0.0001)


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_ngram_language_model_set_n_grams(language_model: NGramLanguageModel) -> None:
    """
    Checks NGramLanguageModel set_n_grams method ideal scenario

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
    """
    language_model.set_n_grams(NGRAM_FREQUENCIES)
    assert NGRAM_FREQUENCIES == language_model._n_gram_frequencies


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_ngram_language_model_set_n_grams_invalid_input(language_model: NGramLanguageModel) -> None:
    """
    Checks NGramLanguageModel set_n_grams method with invalid inputs

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
    """
    bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
    expected = {}
    for bad_input in bad_inputs:
        language_model.set_n_grams(bad_input)
        assert expected == language_model._n_gram_frequencies


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_extract_n_grams(
    language_model: NGramLanguageModel, encoded_text: tuple[int, ...]
) -> None:
    """
    Checks NGramLanguageModel _extract_n_grams method ideal scenario

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
        encoded_text (tuple[int, ...]): Encoded corpus tokens
    """
    assert EXTRACTED_NGRAMS == language_model._extract_n_grams(encoded_text)


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_extract_n_grams_invalid_input(
    language_model: NGramLanguageModel,
) -> None:
    """
    Checks NGramLanguageModel _extract_n_grams method with invalid inputs

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
    """
    bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
    expected = None
    for bad_input in bad_inputs:
        actual = language_model._extract_n_grams(bad_input)
        assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_extract_n_grams_return_value(
    language_model: NGramLanguageModel, encoded_text: tuple[int, ...]
) -> None:
    """
    Checks NGramLanguageModel _extract_n_grams method return value

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
        encoded_text (tuple[int, ...]): Encoded corpus tokens
    """
    actual = language_model._extract_n_grams(encoded_text)
    assert isinstance(actual, tuple)


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_build(language_model: NGramLanguageModel) -> None:
    """
    Checks NGramLanguageModel build method ideal scenario

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
    """
    actual = language_model.build()
    assert actual is True


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_build_none_extract_n_grams(
    language_model: NGramLanguageModel, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks NGramLanguageModel build method with None as _extract_n_grams return value

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock _extract_n_grams
    """
    expected = False
    monkeypatch.setattr(
        language_model,
        "_extract_n_grams",
        lambda *args, **kwargs: None,
    )
    actual = language_model.build()
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_build_invalid_input(language_model: NGramLanguageModel) -> None:
    """
    Checks NGramLanguageModel build method with invalid inputs

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
    """
    bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
    expected = False
    for bad_input in bad_inputs:
        language_model = NGramLanguageModel(bad_input, 2)
        actual = language_model.build()
        assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_get_n_gram_size(language_model: NGramLanguageModel) -> None:
    """
    Checks NGramLanguageModel get_n_gram_size method ideal scenario

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
    """
    assert 2 == language_model.get_n_gram_size()


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_generate_next_token(language_model: NGramLanguageModel) -> None:
    """
    Checks NGramLanguageModel generate_next_token method ideal scenario

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
    """
    expected = {1: 0.375, 4: 0.1875, 9: 0.125, 8: 0.125, 14: 0.0625, 11: 0.0625, 7: 0.0625}
    language_model.build()
    actual = language_model.generate_next_token((0,))
    for key, value in expected.items():
        assert actual.keys() == expected.keys()
        assert value == pytest.approx(actual[key], abs=0.0001)


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_generate_next_token_invalid_input(
    language_model: NGramLanguageModel,
) -> None:
    """
    Checks NGramLanguageModel generate_next_token method with invalid inputs

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
    """
    bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
    expected = None
    for bad_input in bad_inputs:
        actual = language_model.generate_next_token(bad_input)
        assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_ngram_language_model_generate_next_token_long_context(
    language_model: NGramLanguageModel,
) -> None:
    """
    Checks NGramLanguageModel generate_next_token method with long context

    Args:
        language_model (NGramLanguageModel): An instance of NGramLanguageModel
    """
    language_model.build()
    expected = {2: 0.3333, 0: 0.3333, 18: 0.1667, 9: 0.1667}
    actual = language_model.generate_next_token(
        (
            1,
            2,
            3,
            4,
        )
    )
    for key, value in expected.items():
        assert actual.keys() == expected.keys()
        assert value == pytest.approx(actual[key], abs=0.0001)
