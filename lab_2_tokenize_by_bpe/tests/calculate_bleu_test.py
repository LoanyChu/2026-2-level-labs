"""
Checks the second lab's calculate blue function
"""

# pylint: disable=assignment-from-no-return
from unittest import mock

import pytest

from lab_2_tokenize_by_bpe.main import calculate_bleu


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_bleu_ideal() -> None:
    """
    Ideal calculate bleu scenario
    """
    expected = 68.78
    actual = calculate_bleu(
        "Добрый вечер! Как прошел Ваш день?", "Здравствуйте! Как прошел Ваш день?", 3
    )
    assert actual == pytest.approx(expected, abs=0.01)


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_bleu_identical() -> None:
    """
    Ideal identical calculate bleu scenario
    """
    expected = 100.0
    actual = calculate_bleu(
        "Добрый день! Какие у вас планы на вечер?",
        "Добрый день! Какие у вас планы на вечер?",
        3,
    )
    assert actual == expected


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_bleu_none_collect_ngrams() -> None:
    """
    Calculate bleu with None as collect_ngrams' return value
    """
    expected = None
    with mock.patch("lab_2_tokenize_by_bpe.main.collect_ngrams", return_value=None):
        actual = calculate_bleu(
            "Добрый день! Какие у вас планы на вечер?",
            "Добрый день! Какие у вас планы на вечер?",
            3,
        )
    assert actual == expected


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_bleu_none_calculate_precision() -> None:
    """
    Calculate bleu with None as calculate_precision's return value
    """
    expected = None
    with mock.patch("lab_2_tokenize_by_bpe.main.calculate_precision", return_value=None):
        actual = calculate_bleu(
            "Добрый день! Какие у вас планы на вечер?",
            "Добрый день! Какие у вас планы на вечер?",
            3,
        )
    assert actual == expected


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_bleu_none_geo_mean() -> None:
    """
    Calculate bleu with None as geo_mean's return value
    """
    expected = None
    with mock.patch("lab_2_tokenize_by_bpe.main.calculate_geo_mean", return_value=None):
        actual = calculate_bleu(
            "Добрый день! Какие у вас планы на вечер?",
            "Добрый день! Какие у вас планы на вечер?",
            3,
        )
    assert actual == expected


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_bleu_bad_input() -> None:
    """
    Calculate bleu invalid inputs check
    """
    actual_bad_input = [(), [None], {}, None, 1, 1.1, True]
    reference_bad_input = [(), [None], {}, None, 1, 1.1, True]
    max_order_bad_inputs = [None, (), 1.1, [None], "string", {}]
    expected = None
    for index, bad_input in enumerate(actual_bad_input):

        actual = calculate_bleu(bad_input, "Добрый день! Какие у вас планы на вечер?", 3)
        assert actual == expected

        actual = calculate_bleu(
            "Добрый день! Какие у вас планы на вечер?", reference_bad_input[index], 3
        )
        assert actual == expected

    for max_order_bad_input in max_order_bad_inputs:
        actual = calculate_bleu(
            "Добрый день! Какие у вас планы на вечер?",
            "Добрый день! Какие у вас планы на вечер?",
            max_order_bad_input,
        )
        assert actual == expected


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_bleu_return_value() -> None:
    """
    Calculate bleu return value check
    """
    actual = calculate_bleu(
        "В поисках пищи альбатросы способны преодолевать",
        "В поисках пищи альбатросы способны преодолевать",
        3,
    )
    assert isinstance(actual, float)
