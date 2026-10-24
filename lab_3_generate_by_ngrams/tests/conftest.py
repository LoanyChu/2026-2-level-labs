"""
Pytest fixtures for the third lab's classes
"""

# pylint: disable=redefined-outer-name
import pytest

from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearcher,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
    WordProcessor,
)
from lab_3_generate_by_ngrams.tests.constants import PATH_TO_ASSETS


@pytest.fixture
def text_processor() -> TextProcessor:
    """
    Returns an instance of TextProcessor
    """
    return TextProcessor("_")


@pytest.fixture
def word_processor() -> WordProcessor:
    """
    Returns an instance of WordProcessor
    """
    return WordProcessor("<eos>")


@pytest.fixture
def encoded_text(text_processor: TextProcessor) -> tuple[int, ...] | None:
    """
    Encodes test text for NGramLanguageModel tests
    """
    return text_processor.encode(
        "Text Generation is the task of generating text"
        " with the goal of appearing indistinguishable "
        "to human-written text."
    )


@pytest.fixture
def language_model(encoded_text: tuple[int, ...]) -> NGramLanguageModel:
    """
    Returns an instance of NGramLanguageModel
    """
    return NGramLanguageModel(encoded_text, 2)


@pytest.fixture
def language_model_reader() -> NGramLanguageModelReader:
    """
    Returns an instance of NGramLanguageModelReader
    """
    return NGramLanguageModelReader(str(PATH_TO_ASSETS / "en_for_test.json"), eow_token="_")


@pytest.fixture
def greedy_setup(text_processor: TextProcessor) -> tuple[TextProcessor, NGramLanguageModel]:
    """
    Returns setup for greedy text generator tests
    """
    text = """Most unpleasant of all was the first minute when, on coming, happy and
        good-humored, from the theater, with a huge pear in his hand for his
        wife, he had not found his wife in the drawing-room, to his surprise
        had not found her in the study either, and saw her at last in her
        bedroom with the unlucky letter that revealed everything in her hand."""
    encoded = text_processor.encode(text)
    language_model = NGramLanguageModel(encoded, 3)
    language_model.build()
    return (text_processor, language_model)


@pytest.fixture
def greedy_text_generator(
    greedy_setup: tuple[TextProcessor, NGramLanguageModel],
) -> GreedyTextGenerator:
    """
    Returns an instance of GreedyTextGenerator
    """
    text_processor, language_model = greedy_setup
    return GreedyTextGenerator(language_model, text_processor)


@pytest.fixture
def beam_setup(text_processor: TextProcessor) -> tuple[TextProcessor, NGramLanguageModel]:
    """
    Returns setup for beam text generator tests
    """
    with open(PATH_TO_ASSETS / "Anna Karenina - Chapter 1.txt", "r", encoding="utf-8") as file:
        text = file.read()
    encoded = text_processor.encode(text)
    language_model = NGramLanguageModel(encoded, 3)
    language_model.build()
    return (text_processor, language_model)


@pytest.fixture
def beam_searcher(beam_setup: tuple[TextProcessor, NGramLanguageModel]) -> BeamSearcher:
    """
    Returns an instance of BeamSearcher
    """
    _, language_model = beam_setup
    return BeamSearcher(3, language_model)


@pytest.fixture
def beam_text_generator(
    beam_setup: tuple[TextProcessor, NGramLanguageModel],
) -> BeamSearchTextGenerator:
    """
    Returns an instance of BeamSearchTextGenerator
    """
    text_processor, language_model = beam_setup
    return BeamSearchTextGenerator(language_model, text_processor, 3)


@pytest.fixture
def backoff_setup() -> tuple[TextProcessor, tuple[NGramLanguageModel, ...]]:
    """
    Returns setup for backoff text generator tests
    """
    reader = NGramLanguageModelReader(str(PATH_TO_ASSETS / "en.json"), "_")
    models = tuple(m for n in (2, 3, 4) if (m := reader.load(n)) is not None)
    return (reader.get_text_processor(), models)


@pytest.fixture
def backoff_text_generator(
    backoff_setup: tuple[TextProcessor, tuple[NGramLanguageModel, ...]],
) -> BackOffGenerator:
    """
    Returns an instance of BackOffGenerator.
    """
    text_processor, models = backoff_setup
    return BackOffGenerator(models, text_processor)
