"""
Checks the second lab's collect n-grams function
"""

# pylint: disable=assignment-from-no-return
import pytest

from lab_2_tokenize_by_bpe.main import collect_ngrams


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_collect_ngrams_ideal() -> None:
    """
    Ideal collect n-grams scenario
    """
    expected = [
        ("Д", "о", "б"),
        ("о", "б", "р"),
        ("б", "р", "ы"),
        ("р", "ы", "й"),
        ("ы", "й", " "),
        ("й", " ", "в"),
        (" ", "в", "е"),
        ("в", "е", "ч"),
        ("е", "ч", "е"),
        ("ч", "е", "р"),
        ("е", "р", "!"),
        ("р", "!", " "),
        ("!", " ", "К"),
        (" ", "К", "а"),
        ("К", "а", "к"),
        ("а", "к", " "),
        ("к", " ", "п"),
        (" ", "п", "р"),
        ("п", "р", "о"),
        ("р", "о", "ш"),
        ("о", "ш", "е"),
        ("ш", "е", "л"),
        ("е", "л", " "),
        ("л", " ", "В"),
        (" ", "В", "а"),
        ("В", "а", "ш"),
        ("а", "ш", " "),
        ("ш", " ", "д"),
        (" ", "д", "е"),
        ("д", "е", "н"),
        ("е", "н", "ь"),
        ("н", "ь", "?"),
    ]

    actual = collect_ngrams("Добрый вечер! Как прошел Ваш день?", 3)
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_collect_ngrams_bad_input() -> None:
    """
    Collect n-grams invalid inputs check
    """
    text_bad_input = [(), [None], {}, None, 1, 1.1, True]
    order_bad_input = [None, (), 1.1, [None], "string", {}]
    expected = None
    for bad_input in text_bad_input:
        actual = collect_ngrams(bad_input, 3)
        assert expected == actual

    for bad_input in order_bad_input:
        actual = collect_ngrams("Добрый вечер! Как прошел Ваш день?", bad_input)
        assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_collect_ngrams_return_value() -> None:
    """
    Collect n-grams return value check
    """
    actual = collect_ngrams("Добрый вечер! Как прошел Ваш день?", 3)
    assert isinstance(actual, list)
