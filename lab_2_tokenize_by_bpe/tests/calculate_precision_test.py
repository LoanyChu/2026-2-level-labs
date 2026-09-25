# pylint: disable=redefined-outer-name
"""
Checks the second lab's calculate precision function
"""
# pylint: disable=assignment-from-no-return
import pytest

from lab_2_tokenize_by_bpe.main import calculate_precision


@pytest.fixture(scope="function", autouse=True)
def actual_ngrams() -> list:
    """
    Prepare test data.

    Returns:
        list: Correct data list.
    """
    return [
        ("Д",),
        ("о",),
        ("б",),
        ("р",),
        ("ы",),
        ("й",),
        (" ",),
        ("в",),
        ("е",),
        ("ч",),
        ("е",),
        ("р",),
        ("!",),
        (" ",),
        ("К",),
        ("а",),
        ("к",),
        (" ",),
        ("п",),
        ("р",),
        ("о",),
        ("ш",),
        ("е",),
        ("л",),
        (" ",),
        ("В",),
        ("а",),
        ("ш",),
        (" ",),
        ("д",),
        ("е",),
        ("н",),
        ("ь",),
        ("?",),
    ]


@pytest.fixture(scope="function", autouse=True)
def reference_ngrams() -> list:
    """
    Prepare test data.

    Returns:
        list: Correct data list.
    """
    return [
        ("З",),
        ("д",),
        ("р",),
        ("а",),
        ("в",),
        ("с",),
        ("т",),
        ("в",),
        ("у",),
        ("й",),
        ("т",),
        ("е",),
        ("!",),
        (" ",),
        ("К",),
        ("а",),
        ("к",),
        (" ",),
        ("п",),
        ("р",),
        ("о",),
        ("ш",),
        ("е",),
        ("л",),
        (" ",),
        ("В",),
        ("а",),
        ("ш",),
        (" ",),
        ("д",),
        ("е",),
        ("н",),
        ("ь",),
        ("?",),
    ]


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_precision_ideal(actual_ngrams: list, reference_ngrams: list) -> None:
    """
    Ideal calculate precision scenario.

    Args:
        actual_ngrams (list): Prepared test data.
        reference_ngrams (list): Prepared test data.
    """
    expected = 0.8181818181818182
    actual = calculate_precision(actual_ngrams, reference_ngrams)
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_precision_missing_value(reference_ngrams: list) -> None:
    """
    Calculate precision missing value check.

    Args:
        reference_ngrams (list): Prepared test data.
    """
    expected = 0.0
    actual = calculate_precision([], reference_ngrams)
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_precision_bad_input(actual_ngrams: list, reference_ngrams: list) -> None:
    """
    Calculate precision invalid inputs check.

    Args:
        actual_ngrams (list): Prepared test data.
        reference_ngrams (list): Prepared test data.
    """
    actual_ngrams_bad_input = [(), "string", {}, None, 1, 1.1, True]
    reference_ngrams_bad_input = [None, (), 1.1, True, 1, "string", {}]
    expected = None
    for index, bad_input in enumerate(actual_ngrams_bad_input):
        actual = calculate_precision(bad_input, reference_ngrams)
        assert expected == actual

        actual = calculate_precision(actual_ngrams, reference_ngrams_bad_input[index])
        assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_calculate_precision_return_value(actual_ngrams: list, reference_ngrams: list) -> None:
    """
    Calculate precision return value check.

    Args:
        actual_ngrams (list): Prepared test data.
        reference_ngrams (list): Prepared test data.
    """
    actual = calculate_precision(actual_ngrams, reference_ngrams)
    assert isinstance(actual, float)
