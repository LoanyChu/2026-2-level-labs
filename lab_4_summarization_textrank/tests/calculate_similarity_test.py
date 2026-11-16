"""
Checks the fourth lab similarity calculation function
"""

import pytest

from lab_4_summarization_textrank.main import calculate_similarity, IncorrectArgumentError


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark8
@pytest.mark.mark10
def test_calculate_similarity_ideal() -> None:
    """
    Checks similarity calculation for valid sequences
    """
    assert 0.5714285714285714 == calculate_similarity([1, 2, 3, 4, 5], [5, 8, 3, 7, 2, 1, 8])
    assert calculate_similarity([1, 2, 3, 4, 5], [5, 8, 3, 7, 2, 1, 8]) == calculate_similarity(
        [5, 8, 3, 7, 2, 1, 8], [1, 2, 3, 4, 5]
    )
    assert 0 == calculate_similarity([], [5, 8, 3, 7, 2, 1, 8])
    assert 0 == calculate_similarity([], [])
    assert 0 == calculate_similarity([1, 2, 3], [4, 5, 6])
    assert 1 == calculate_similarity([1, 1], [1])


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark8
@pytest.mark.mark10
def test_calculate_similarity_non_iterable() -> None:
    """
    Checks that non-iterable inputs raise ValueError
    """
    sequence = [1, 2, 3, 4]
    for wrong_input in [4, 4.0, True]:
        with pytest.raises(IncorrectArgumentError):
            calculate_similarity(wrong_input, sequence)
        with pytest.raises(IncorrectArgumentError):
            calculate_similarity(sequence, wrong_input)
