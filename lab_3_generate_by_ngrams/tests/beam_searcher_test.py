"""
Checks the third lab's BeamSearcher class
"""

# pylint: disable=protected-access, no-member

import pytest
from pytest import MonkeyPatch

from lab_3_generate_by_ngrams.main import BeamSearcher, NGramLanguageModel, TextProcessor


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_searcher_fields(
    beam_searcher: BeamSearcher, beam_setup: tuple[TextProcessor, NGramLanguageModel]
) -> None:
    """
    Checks if BeamSearcher fields are created correctly

    Args:
        beam_searcher (BeamSearcher): An instance of BeamSearcher
        beam_setup (tuple[TextProcessor, NGramLanguageModel]): Fixture setup with
            text processor and language model
    """
    _, language_model = beam_setup
    assert 3 == beam_searcher._beam_width
    assert language_model == beam_searcher._model


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_searcher_get_next_token(beam_searcher: BeamSearcher) -> None:
    """
    Checks BeamSearcher get_next_token method ideal scenario

    Args:
        beam_searcher (BeamSearcher): An instance of BeamSearcher
    """
    expected = [(0, 0.6617), (3, 0.1955), (6, 0.0376)]
    actual = beam_searcher.get_next_token((4, 9, 7))
    for expected_tuple, actual_tuple in zip(expected, actual):
        assert expected_tuple[0] == actual_tuple[0]
        assert expected_tuple[1] == pytest.approx(actual_tuple[1], abs=0.0001)


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_searcher_get_next_token_invalid_input(beam_searcher: BeamSearcher) -> None:
    """
    Checks BeamSearcher get_next_token method with invalid inputs

    Args:
        beam_searcher (BeamSearcher): An instance of BeamSearcher
    """
    bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
    expected = None
    for bad_input in bad_inputs:
        actual = beam_searcher.get_next_token(bad_input)
        assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_searcher_get_next_token_none_generate_next_token(
    beam_searcher: BeamSearcher, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BeamSearcher run method with None
    as generate_next_token return value

    Args:
        beam_searcher (BeamSearcher): An instance of BeamSearcher
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock generate_next_token
    """
    monkeypatch.setattr(
        beam_searcher._model,
        "generate_next_token",
        lambda *args, **kwargs: None,
    )
    actual = beam_searcher.get_next_token((4, 9, 7))
    expected = None
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_searcher_get_next_token_empty_generate_next_token(
    beam_searcher: BeamSearcher, monkeypatch: MonkeyPatch
) -> None:
    """
    Checks BeamSearcher run method with empty generate_next_token

    Args:
        beam_searcher (BeamSearcher): An instance of BeamSearcher
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock generate_next_token
    """
    monkeypatch.setattr(
        beam_searcher._model,
        "generate_next_token",
        lambda *args, **kwargs: [],
    )
    actual = beam_searcher.get_next_token((4, 9, 7))
    expected = []
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_searcher_continue_sequence(beam_searcher: BeamSearcher) -> None:
    """
    Checks BeamSearcher continue_sequence method ideal scenario

    Args:
        beam_searcher (BeamSearcher): An instance of BeamSearcher
    """
    expected = {(4, 9, 7, 0): 4.5469, (4, 9, 7, 3): 5.9915, (4, 9, 7, 13): 7.4186}
    actual = beam_searcher.continue_sequence(
        (4, 9, 7), [(0, 0.0106), (3, 0.0025), (13, 0.0006)], {(4, 9, 7): 0.0}
    )
    assert actual == pytest.approx(expected, abs=0.0001)


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_searcher_continue_sequence_invalid_input(beam_searcher: BeamSearcher) -> None:
    """
    Checks BeamSearcher continue_sequence method with invalid inputs

    Args:
        beam_searcher (BeamSearcher): An instance of BeamSearcher
    """
    bad_sequence_inputs = [[None], 1, True, {}, None, (), 1.1, "string"]
    bad_next_tokens_inputs = [1, True, {}, None, [], 1.1, "string"]
    long_next_token_input = [(0, 0.0106), (3, 0.0025), (13, 0.0006), (20, 0.0001)]
    bad_sequence_candidates_input = [[None], 1, True, {}, None, (), 1.1, "string"]
    expected = None
    for bad_input in bad_sequence_inputs:
        actual = beam_searcher.continue_sequence(
            bad_input, [(0, 0.0106), (3, 0.0025), (13, 0.0006)], {(4, 9, 7): 0.0}
        )

        assert expected == actual

    for bad_input in bad_next_tokens_inputs:
        actual = beam_searcher.continue_sequence((4, 9, 7), bad_input, {(4, 9, 7): 0.0})
        assert expected == actual

    for bad_input in bad_sequence_candidates_input:
        actual = beam_searcher.continue_sequence(
            (4, 9, 7), [(0, 0.0106), (3, 0.0025), (13, 0.0006)], bad_input
        )
        assert expected == actual

    actual_with_long_token = beam_searcher.continue_sequence(
        (4, 9, 7), long_next_token_input, {(4, 9, 7): 0.0}
    )
    assert expected == actual_with_long_token

    actual_without_sequence_in_continue = beam_searcher.continue_sequence(
        (4, 9, 7), [(0, 0.0106), (3, 0.0025), (13, 0.0006)], {(2, 2, 2): 0.0}
    )
    assert expected == actual_without_sequence_in_continue


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_searcher_prune_sequence_candidates(beam_searcher: BeamSearcher) -> None:
    """
    Checks BeamSearcher prune_sequence_candidates method ideal scenario

    Args:
        beam_searcher (BeamSearcher): An instance of BeamSearcher
    """
    expected = {(4, 9, 7, 0): 4.551, (4, 9, 7, 3): 5.9934, (4, 9, 7, 13): 7.4597}
    actual = beam_searcher.prune_sequence_candidates(
        {(4, 9, 7, 0): 4.551, (4, 9, 7, 3): 5.9934, (4, 9, 7, 13): 7.4597}
    )
    assert expected == actual


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_searcher_prune_sequence_candidates_invalid_inputs(
    beam_searcher: BeamSearcher,
) -> None:
    """
    Checks BeamSearcher prune_sequence_candidates method with invalid inputs

    Args:
        beam_searcher (BeamSearcher): An instance of BeamSearcher
    """
    bad_inputs = [1, [None], {}, None, (), 1.1, True, "string"]
    expected = None
    for bad_input in bad_inputs:
        actual = beam_searcher.prune_sequence_candidates(bad_input)
        assert expected == actual
