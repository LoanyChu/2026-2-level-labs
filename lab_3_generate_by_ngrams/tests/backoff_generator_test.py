"""
Checks the third lab's BackOffGenerator class
"""

# pylint: disable=protected-access, no-member
import pytest
from pytest import MonkeyPatch

from lab_3_generate_by_ngrams.main import BackOffGenerator, NGramLanguageModel, TextProcessor


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_fields(
    backoff_text_generator: BackOffGenerator,
    backoff_setup: tuple[TextProcessor, tuple[NGramLanguageModel, ...]],
) -> None:
    """
    Checks if BackOffGenerator fields are created correctly

    Args:
        backoff_text_generator (BackOffGenerator): An instance of BackOffGenerator
        backoff_setup (tuple[TextProcessor, tuple[NGramLanguageModel, ...]]): Fixture setup with
            text processor and language models
    """
    backoff_text_processor, language_models = backoff_setup
    assert backoff_text_processor == backoff_text_generator._text_processor
    expected = {
        model.get_n_gram_size(): model
        for model in sorted(
            language_models, key=lambda model: model.get_n_gram_size(), reverse=True
        )
    }
    assert expected == backoff_text_generator._language_models


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_get_next_token(backoff_text_generator: BackOffGenerator) -> None:
    """
    Checks BackOffGenerator _get_next_token method ideal scenario

    Args:
        backoff_text_generator (BackOffGenerator): An instance of BackOffGenerator
    """
    expected = {0: 1.0}
    actual = backoff_text_generator._get_next_token((4, 9, 7))
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_run_empty_get_next_token(
    backoff_text_generator: BackOffGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BackOffGenerator run method with empty _get_next_token return value

    Args:
        backoff_text_generator (BackOffGenerator): An instance of BackOffGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock _get_next_token
    """
    monkeypatch.setattr(
        backoff_text_generator,
        "_get_next_token",
        lambda *args, **kwargs: [],
    )
    actual = backoff_text_generator.run("The", 50)
    expected = "The."
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_get_next_token_none_generate_next_token(
    backoff_text_generator: BackOffGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BackOffGenerator _get_next_token method with None
    as generate_next_token return value

    Args:
        backoff_text_generator (BackOffGenerator): An instance of BackOffGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock generate_next_token
    """
    for index in (2, 3, 4):
        monkeypatch.setattr(
            backoff_text_generator._language_models[index],
            "generate_next_token",
            lambda *args, **kwargs: None,
        )
    actual = backoff_text_generator._get_next_token((4, 9, 7))
    expected = None
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_get_next_token_empty_generate_next_token(
    backoff_text_generator: BackOffGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BackOffGenerator _get_next_token method with empty generate_next_token

    Args:
        backoff_text_generator (BackOffGenerator): An instance of BackOffGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock generate_next_token
    """
    for index in (2, 3, 4):
        monkeypatch.setattr(
            backoff_text_generator._language_models[index],
            "generate_next_token",
            lambda *args, **kwargs: {},
        )
    actual = backoff_text_generator._get_next_token((4, 9, 7))
    expected = None
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_get_next_token_without_models(
    backoff_setup: tuple[TextProcessor, tuple[NGramLanguageModel, ...]],
) -> None:
    """
    Checks BackOffGenerator _get_next_token method without models

    Args:
        backoff_setup (tuple[TextProcessor, tuple[NGramLanguageModel, ...]]): Fixture setup with
            text processor and language models
    """
    text_processor, _ = backoff_setup
    backoff_generator = BackOffGenerator((), text_processor)
    actual = backoff_generator._get_next_token((4, 9, 7))
    expected = None
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_get_next_token_invalid_input(
    backoff_text_generator: BackOffGenerator,
) -> None:
    """
    Checks BackOffGenerator _get_next_token method with invalid inputs

    Args:
        backoff_text_generator (BackOffGenerator): An instance of BackOffGenerator
    """
    bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
    expected = None
    for bad_input in bad_inputs:
        actual = backoff_text_generator._get_next_token(bad_input)
        assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_run(backoff_text_generator: BackOffGenerator) -> None:
    """
    Checks BackOffGenerator run method ideal scenario

    Args:
        backoff_text_generator (BackOffGenerator): An instance of BackOffGenerator
    """
    actual = backoff_text_generator.run("The", 50)
    assert "The the the the the the the the the the the the the t." == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_run_invalid_input(backoff_text_generator: BackOffGenerator) -> None:
    """
    Checks BackOffGenerator run method with invalid inputs

    Args:
        backoff_text_generator (BackOffGenerator): An instance of BackOffGenerator
    """
    bad_len_inputs = [[None], {}, None, (), 1.1, "string", 0, -1]
    bad_prompt_inputs = [1, [None], {}, None, (), 1.1, True, ""]
    expected = None
    for bad_input in bad_len_inputs:
        actual = backoff_text_generator.run("The", bad_input)
        assert expected == actual

    for bad_input in bad_prompt_inputs:
        actual = backoff_text_generator.run(bad_input, 50)
        assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_run_none_encode(
    backoff_text_generator: BackOffGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BackOffGenerator run method with None as encode return value

    Args:
        backoff_text_generator (BackOffGenerator): An instance of BackOffGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock encode
    """
    monkeypatch.setattr(
        backoff_text_generator._text_processor,
        "encode",
        lambda *args, **kwargs: None,
    )
    actual = backoff_text_generator.run("The", 50)
    assert actual is None


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark10
def test_backoff_generator_run_none_get_next_token(
    backoff_text_generator: BackOffGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BackOffGenerator run method with None and _get_next_token return value

    Args:
        backoff_text_generator (BackOffGenerator): An instance of BackOffGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock _get_next_token
    """
    monkeypatch.setattr(
        backoff_text_generator,
        "_get_next_token",
        lambda *args, **kwargs: None,
    )
    actual = backoff_text_generator.run("The", 50)
    assert actual is None
