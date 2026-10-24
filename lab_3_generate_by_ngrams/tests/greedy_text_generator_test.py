"""
Checks the third lab's GreedyTextGenerator class
"""

# pylint: disable=protected-access, no-member

import pytest
from pytest import MonkeyPatch

from lab_3_generate_by_ngrams.main import GreedyTextGenerator, NGramLanguageModel, TextProcessor


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_greedy_text_generator_fields(
    greedy_text_generator: GreedyTextGenerator,
    greedy_setup: tuple[TextProcessor, NGramLanguageModel],
) -> None:
    """
    Checks if GreedyTextGenerator fields are created correctly

    Args:
        greedy_text_generator (GreedyTextGenerator): An instance of GreedyTextGenerator
        greedy_setup (tuple[TextProcessor, NGramLanguageModel]): Fixture setup with
            text processor and language model
    """
    text_processor, language_model = greedy_setup
    assert language_model == greedy_text_generator._model
    assert text_processor == greedy_text_generator._text_processor


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_greedy_text_generator_run(greedy_text_generator: GreedyTextGenerator) -> None:
    """
    Checks GreedyTextGenerator run method ideal scenario

    Args:
        greedy_text_generator (GreedyTextGenerator): An instance of GreedyTextGenerator
    """
    actual = greedy_text_generator.run("He", 4)
    assert "He had." == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_greedy_text_generator_run_none_encode(
    greedy_text_generator: GreedyTextGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks GreedyTextGenerator run method with None as encode return value

    Args:
        greedy_text_generator (GreedyTextGenerator): An instance of GreedyTextGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock encode
    """
    monkeypatch.setattr(
        greedy_text_generator._text_processor,
        "encode",
        lambda *args, **kwargs: None,
    )
    actual = greedy_text_generator.run("He", 4)
    expected = None
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_greedy_text_generator_run_none_generate_next_token(
    greedy_text_generator: GreedyTextGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks GreedyTextGenerator run method with None
    as generate_next_token return value

    Args:
        greedy_text_generator (GreedyTextGenerator): An instance of GreedyTextGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock generate_next_token
    """
    monkeypatch.setattr(
        greedy_text_generator._model,
        "generate_next_token",
        lambda *args, **kwargs: None,
    )
    actual = greedy_text_generator.run("He", 4)
    expected = "He."
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_greedy_text_generator_run_empty_next_candidate(
    greedy_text_generator: GreedyTextGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks GreedyTextGenerator run method with empty next candidate

    Args:
        greedy_text_generator (GreedyTextGenerator): An instance of GreedyTextGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock generate_next_token
    """
    monkeypatch.setattr(
        greedy_text_generator._model,
        "generate_next_token",
        lambda *args, **kwargs: {},
    )
    actual = greedy_text_generator.run("He", 4)
    expected = "He."
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_greedy_text_generator_run_invalid_input(
    greedy_text_generator: GreedyTextGenerator,
) -> None:
    """
    Checks GreedyTextGenerator run method with invalid inputs

    Args:
        greedy_text_generator (GreedyTextGenerator): An instance of GreedyTextGenerator
    """
    expected = None
    seq_len_bad_input = [[None], {}, None, (), 1.1, "string", 0, -1]
    prompt_bad_input = [1, [None], {}, None, (), 1.1, "", 0]
    for bad_input in seq_len_bad_input:
        actual = greedy_text_generator.run("He", bad_input)
        assert expected == actual
    for bad_input in prompt_bad_input:
        actual = greedy_text_generator.run(bad_input, 4)
        assert expected == actual
