"""
Checks the fourth lab SentencePreprocessor abstraction
"""

# pylint: disable=protected-access, duplicate-code
import string

import pytest

from lab_4_summarization_textrank.main import IncorrectArgumentError, SentencePreprocessor

TEXT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
    "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit "
    "anim id est laborum."
)

SENTENCES = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
    "commodo consequat.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat "
    "nulla pariatur.",
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit "
    "anim id est laborum.",
]

PREPROCESSED_SENTENCES = [
    (
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "eiusmod",
        "tempor",
        "incididunt",
        "labore",
        "dolore",
        "magna",
        "aliqua",
    ),
    (
        "enim",
        "ad",
        "minim",
        "veniam",
        "quis",
        "nostrud",
        "exercitation",
        "ullamco",
        "laboris",
        "nisi",
        "aliquip",
        "commodo",
        "consequat",
    ),
    (
        "duis",
        "aute",
        "irure",
        "dolor",
        "reprehenderit",
        "voluptate",
        "velit",
        "cillum",
        "dolore",
        "eu",
        "fugiat",
        "nulla",
        "pariatur",
    ),
    (
        "excepteur",
        "sint",
        "occaecat",
        "cupidatat",
        "proident",
        "sunt",
        "culpa",
        "officia",
        "deserunt",
        "mollit",
        "anim",
        "id",
        "laborum",
    ),
]


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_sentence_preprocessor_initialization_ideal() -> None:
    """
    Checks SentencePreprocessor initialization and its attributes
    """
    stop_words = ("and", "of", "the")
    punctuation = (",", "!", "?")
    preprocessor = SentencePreprocessor(stop_words, punctuation)

    for attribute in ["_stop_words", "_punctuation"]:
        assert hasattr(
            preprocessor, attribute
        ), f"Sentence preprocessor misses {attribute} attribute"


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_sentence_preprocessor_initialization_value_error() -> None:
    """
    Checks that invalid initialization arguments raise ValueError
    """
    wrong_inputs = [
        1,
        (
            1,
            2,
        ),
        True,
        [1, 2],
    ]
    for wrong_input in wrong_inputs:
        with pytest.raises(IncorrectArgumentError):
            SentencePreprocessor(wrong_input, ("1", "2", "3"))
        with pytest.raises(IncorrectArgumentError):
            SentencePreprocessor(("1", "2", "3"), wrong_input)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_split_by_sentence_ideal() -> None:
    """
    Checks splitting text into sentences
    """
    preprocessor = SentencePreprocessor((), tuple(string.punctuation))
    output = preprocessor._split_by_sentence(TEXT)
    assert isinstance(output, tuple)
    for index, sentence in enumerate(output):
        assert SENTENCES[index] == sentence.get_text()


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_split_by_sentence_value_error() -> None:
    """
    Checks that invalid input to _split_by_sentence raises ValueError
    """
    preprocessor = SentencePreprocessor((), tuple(string.punctuation))

    wrong_inputs = [
        1,
        (
            1,
            2,
        ),
        True,
        ["d", "a"],
        1.0,
    ]
    for wrong_input in wrong_inputs:
        with pytest.raises(IncorrectArgumentError):
            preprocessor._split_by_sentence(wrong_input)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_get_sentences_value_error() -> None:
    """
    Checks that invalid input to get_sentences raises ValueError
    """
    stop_words = ("sed", "ex", "ut", "est", "non", "in", "qui", "esse", "ea", "et", "do")
    preprocessor = SentencePreprocessor(stop_words, tuple(string.punctuation))
    wrong_inputs = [
        1,
        (
            1,
            2,
        ),
        True,
        ["d", "a"],
        1.0,
    ]
    for wrong_input in wrong_inputs:
        with pytest.raises(IncorrectArgumentError):
            preprocessor.get_sentences(wrong_input)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_preprocess_sentences_value_error() -> None:
    """
    Checks that invalid input to _preprocess_sentences raises ValueError
    """
    stop_words = ("sed", "ex", "ut", "est", "non", "in", "qui", "esse", "ea", "et", "do")
    preprocessor = SentencePreprocessor(stop_words, tuple(string.punctuation))
    with pytest.raises(IncorrectArgumentError):
        preprocessor._preprocess_sentences(SENTENCES)
