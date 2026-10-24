"""
Checks the third lab's BeamSearchTextGenerator class
"""

# pylint: disable=protected-access, no-member

import pytest
from pytest import MonkeyPatch

from lab_3_generate_by_ngrams.main import BeamSearchTextGenerator, NGramLanguageModel, TextProcessor


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_search_text_generator_fields(
    beam_text_generator: BeamSearchTextGenerator,
    beam_setup: tuple[TextProcessor, NGramLanguageModel],
) -> None:
    """
    Checks if BeamSearchTextGenerator fields are created correctly

    Args:
        beam_text_generator (BeamSearchTextGenerator): An instance of BeamSearchTextGenerator
        beam_setup (tuple[TextProcessor, NGramLanguageModel]): Fixture setup with
            text processor and language model
    """
    text_processor, language_model = beam_setup
    assert language_model == beam_text_generator._language_model
    assert text_processor == beam_text_generator._text_processor
    assert 3 == beam_text_generator._beam_width


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_search_text_generator_run(beam_text_generator: BeamSearchTextGenerator) -> None:
    """
    Checks BeamSearchTextGenerator run method ideal scenario

    Args:
        beam_text_generator (BeamSearchTextGenerator): An instance of BeamSearchTextGenerator
    """
    actual = beam_text_generator.run("The", 70)
    assert actual == "The his the his the his the his the his the his the his the his the his t."


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_search_text_generator_run_invalid_input(
    beam_text_generator: BeamSearchTextGenerator,
) -> None:
    """
    Checks BeamSearchTextGenerator run method with invalid inputs

    Args:
        beam_text_generator (BeamSearchTextGenerator): An instance of BeamSearchTextGenerator
    """
    bad_seq_len_inputs = [[None], {}, None, (), 1.1, "string", 0]
    expected = None
    for bad_input in bad_seq_len_inputs:
        actual = beam_text_generator.run("The", bad_input)
        assert expected == actual

    bad_prompt_inputs = [[None], 1, True, {}, None, (), 1.1, ""]
    expected = None
    for bad_input in bad_prompt_inputs:
        actual = beam_text_generator.run(bad_input, 70)
        assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_search_text_generator_run_none_encode(
    beam_text_generator: BeamSearchTextGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BeamSearchTextGenerator run method with None as encode return value

    Args:
        beam_text_generator (BeamSearchTextGenerator): An instance of BeamSearchTextGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock encode
    """
    monkeypatch.setattr(
        beam_text_generator._text_processor,
        "encode",
        lambda *args, **kwargs: None,
    )
    actual = beam_text_generator.run("The", 70)
    expected = None
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_search_text_generator_run_none_get_next_token(
    beam_text_generator: BeamSearchTextGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BeamSearchTextGenerator run method with None
    as _get_next_token return value

    Args:
        beam_text_generator (BeamSearchTextGenerator): An instance of BeamSearchTextGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock _get_next_token
    """
    monkeypatch.setattr(
        beam_text_generator,
        "_get_next_token",
        lambda *args, **kwargs: None,
    )
    actual = beam_text_generator.run("The", 70)
    expected = None
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_search_text_generator_run_none_continue_sequence(
    beam_text_generator: BeamSearchTextGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BeamSearchTextGenerator run method with None
    as _get_next_tokens return value

    Args:
        beam_text_generator (BeamSearchTextGenerator): An instance of BeamSearchTextGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock continue_sequence
    """
    monkeypatch.setattr(
        beam_text_generator.beam_searcher,
        "continue_sequence",
        lambda *args, **kwargs: None,
    )
    actual = beam_text_generator.run("The", 70)
    expected = "The."
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_search_text_generator_run_none_prune_sequence_candidates(
    beam_text_generator: BeamSearchTextGenerator, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BeamSearchTextGenerator run method with None
    as prune_sequence_candidates return value

    Args:
        beam_text_generator (BeamSearchTextGenerator): An instance of BeamSearchTextGenerator
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock prune_sequence_candidates
    """
    monkeypatch.setattr(
        beam_text_generator.beam_searcher,
        "prune_sequence_candidates",
        lambda *args, **kwargs: None,
    )
    actual = beam_text_generator.run("The", 70)
    expected = None
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_search_text_generator_get_next_tokens(
    beam_text_generator: BeamSearchTextGenerator,
) -> None:
    """
    Checks BeamSearchTextGenerator _get_next_tokens method ideal scenario

    Args:
        beam_text_generator (BeamSearchTextGenerator): An instance of BeamSearchTextGenerator
    """
    expected = [(0, 0.6617), (3, 0.1955), (6, 0.0376)]
    actual = beam_text_generator._get_next_token((4, 9, 7))
    for expected_tuple, actual_tuple in zip(expected, actual):
        assert expected_tuple[0] == actual_tuple[0]
        assert pytest.approx(expected_tuple[1], abs=0.0001) == actual_tuple[1]


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_search_text_generator_get_next_tokens_invalid_input(
    beam_text_generator: BeamSearchTextGenerator,
) -> None:
    """
    Checks BeamSearchTextGenerator _get_next_tokens method with invalid inputs

    Args:
        beam_text_generator (BeamSearchTextGenerator): An instance of BeamSearchTextGenerator
    """
    expected = None
    bad_inputs = [1, [None], {}, None, (), 1.1, "string", 0]
    for bad_input in bad_inputs:
        actual = beam_text_generator._get_next_token(bad_input)
        assert expected == actual
