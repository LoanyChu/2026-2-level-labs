"""
Checks the fourth lab Sentence abstraction
"""

import pytest

from lab_4_summarization_textrank.main import IncorrectArgumentError, Sentence


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_sentence_initialization_ideal(sentence: Sentence) -> None:
    """
    Checks Sentence initialization and its attributes

    Args:
        sentence (Sentence): An instance of Sentence
    """
    for attribute in ["_text", "_preprocessed", "_encoded", "_position"]:
        assert hasattr(sentence, attribute), f"Sentence misses {attribute} attribute"


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_sentence_initialization_value_error() -> None:
    """
    Checks that invalid initialization arguments raise ValueError
    """
    text_inputs = [
        1,
        True,
        (
            1,
            2,
        ),
        [True, 8],
        4.0,
    ]
    position_inputs = [
        "ааа",
        True,
        (
            1,
            2,
        ),
        [True, 8],
    ]
    for wrong_input in text_inputs:
        with pytest.raises(IncorrectArgumentError):
            Sentence(wrong_input, 0)
    for wrong_input in position_inputs:
        with pytest.raises(IncorrectArgumentError):
            Sentence("Мама мыла раму", wrong_input)
