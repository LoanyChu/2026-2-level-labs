"""
Pytest fixtures for the fourth lab's classes
"""

# pylint: disable=redefined-outer-name
import string
from pathlib import Path

import pytest

from lab_4_summarization_textrank.main import (
    Sentence,
    SentenceEncoder,
    SentencePreprocessor,
    TextGraph,
)

PATH_TO_ASSETS = Path(__file__).parent.parent / "assets"


@pytest.fixture
def sentence() -> Sentence:
    """
    Returns an instance of Sentence

    Returns:
        Sentence: An instance of Sentence
    """
    return Sentence("Мама мыла раму", 0)


@pytest.fixture
def sentence_encoder() -> SentenceEncoder:
    """
    Returns an instance of SentenceEncoder

    Returns:
        SentenceEncoder: An instance of SentenceEncoder
    """
    return SentenceEncoder()


@pytest.fixture
def encoder_sentences() -> tuple[Sentence, ...]:
    """
    Returns preprocessed sentences for SentenceEncoder tests

    Returns:
        tuple[Sentence, ...]: Preprocessed sentences
    """
    text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
        "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
        "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit "
        "anim id est laborum."
    )
    preprocessor = SentencePreprocessor(
        ("sed", "ex", "ut", "est", "non", "in", "qui", "esse", "ea", "et", "do"),
        tuple(string.punctuation),
    )
    return preprocessor.get_sentences(text)


@pytest.fixture
def matrix_sentences() -> list[Sentence]:
    """
    Returns preprocessed and encoded sentences for TextGraph tests

    Returns:
        list[Sentence]: Preprocessed and encoded sentences
    """
    text = "Мама мыла раму? Раму долго мыла. Идти можно долго! Можно долго использовать раму?"
    preprocessor = SentencePreprocessor((), tuple(string.punctuation))
    sentences_list = list(preprocessor.get_sentences(text))
    encoder = SentenceEncoder()
    encoder.encode_sentences(sentences_list)
    return sentences_list


@pytest.fixture
def graph() -> TextGraph:
    """
    Returns a filled TextGraph instance

    Returns:
        TextGraph: Filled similarity graph
    """
    with open(PATH_TO_ASSETS / "texts" / "article_701.txt", "r", encoding="utf-8") as file:
        text = file.read()
    with open(PATH_TO_ASSETS / "stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = tuple(file.read().split("\n"))
    preprocessor = SentencePreprocessor(stop_words, tuple(string.punctuation))
    sentences = preprocessor.get_sentences(text)
    encoder = SentenceEncoder()
    encoder.encode_sentences(sentences)
    sim_graph = TextGraph()
    sim_graph.build(sentences)
    return sim_graph
