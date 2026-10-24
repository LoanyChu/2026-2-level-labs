# pylint: disable=protected-access
"""
Checks the third lab's WordProcessor class
"""

import pytest

from lab_3_generate_by_ngrams.main import (
    GreedyTextGenerator,
    NGramLanguageModel,
    TextProcessor,
    WordProcessor,
)


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_fields(word_processor: WordProcessor) -> None:
    """
    Checks if WordProcessor fields are created correctly

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    assert isinstance(word_processor, TextProcessor)
    assert "<eos>" == word_processor._end_of_word_token
    assert {"<eos>": 0} == word_processor._storage


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_tokenize(word_processor: WordProcessor) -> None:
    """
    Checks WordProcessor _tokenize method ideal scenario

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    expected = ("hello", "world", "<eos>", "its", "oclock", "<eos>", "bye")
    actual = word_processor._tokenize("Hello, world! It's 2 o'clock... Bye")
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_tokenize_no_words(word_processor: WordProcessor) -> None:
    """
    Checks WordProcessor _tokenize method with text without words

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    assert word_processor._tokenize("12 ... 34!") is None


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_tokenize_invalid_input(word_processor: WordProcessor) -> None:
    """
    Checks WordProcessor _tokenize method with invalid inputs

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    bad_inputs = [1, [None], {}, None, (), 1.1, True, ""]
    for bad_input in bad_inputs:
        assert word_processor._tokenize(bad_input) is None


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_put(word_processor: WordProcessor) -> None:
    """
    Checks WordProcessor _put method ideal scenario

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    word_processor._put("harry")
    word_processor._put("potter")
    word_processor._put("harry")
    assert {"<eos>": 0, "harry": 1, "potter": 2} == word_processor._storage


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_put_invalid_input(word_processor: WordProcessor) -> None:
    """
    Checks WordProcessor _put method with invalid inputs

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    bad_inputs = [1, [None], {}, None, (), 1.1, True, ""]
    for bad_input in bad_inputs:
        word_processor._put(bad_input)
    assert {"<eos>": 0} == word_processor._storage


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_postprocess_decoded_text(word_processor: WordProcessor) -> None:
    """
    Checks WordProcessor _postprocess_decoded_text method ideal scenario

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    decoded = ("hello", "world", "<eos>", "its", "late", "<eos>", "bye")
    actual = word_processor._postprocess_decoded_text(decoded)
    assert "Hello world. Its late. Bye." == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_postprocess_decoded_text_no_words(
    word_processor: WordProcessor,
) -> None:
    """
    Checks WordProcessor _postprocess_decoded_text method with special tokens only

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    assert word_processor._postprocess_decoded_text(("<eos>", "<eos>")) is None


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_postprocess_decoded_text_invalid_input(
    word_processor: WordProcessor,
) -> None:
    """
    Checks WordProcessor _postprocess_decoded_text method with invalid inputs

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
    for bad_input in bad_inputs:
        assert word_processor._postprocess_decoded_text(bad_input) is None


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_encode_decode(word_processor: WordProcessor) -> None:
    """
    Checks WordProcessor encode and decode methods inherited from TextProcessor

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    encoded = word_processor.encode("The cat sat. The cat ran!")
    assert (1, 2, 3, 0, 1, 2, 4, 0) == encoded
    assert "The cat sat. The cat ran." == word_processor.decode(encoded)


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_word_processor_greedy_generation(word_processor: WordProcessor) -> None:
    """
    Checks that generators work with WordProcessor without any changes

    Args:
        word_processor (WordProcessor): An instance of WordProcessor
    """
    encoded = word_processor.encode("The cat sat on the mat. The cat sat on the sofa.")
    language_model = NGramLanguageModel(encoded, 3)
    language_model.build()
    generator = GreedyTextGenerator(language_model, word_processor)
    assert "The cat sat on the sofa." == generator.run("The cat", 6)
