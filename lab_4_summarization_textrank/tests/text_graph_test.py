# pylint: disable=protected-access
"""
Checks the fourth lab TextGraph abstraction
"""

import pytest

from lab_4_summarization_textrank.main import (
    IncorrectArgumentError,
    Sentence,
    TextGraph,
)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark8
@pytest.mark.mark10
def test_similarity_graph_initialization() -> None:
    """
    Checks TextGraph initialization and its attributes
    """
    matrix = TextGraph()
    assert hasattr(matrix, "_matrix")
    assert isinstance(matrix._matrix, list)
    assert len(matrix) == 0
    assert str(matrix) == "TextGraph(vertices=0)"


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark8
@pytest.mark.mark10
def test_fill_from_single_sentence() -> None:
    """Checks that a sentence without graph edges is still included as a vertex."""
    sentence = Sentence("only sentence", 0)
    sentence.set_encoded((1000,))
    graph = TextGraph()

    graph.build([sentence])

    assert graph.get_vertices() == (sentence,)
    assert graph.calculate_inout_score(sentence) == 0


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark8
@pytest.mark.mark10
def test_fill_from_duplicate_sentences() -> None:
    """Checks that identical sentences remain distinct positional vertices."""
    first = Sentence("same", 0)
    second = Sentence("same", 1)
    first.set_encoded((1000,))
    second.set_encoded((1000,))
    graph = TextGraph()

    graph.build([first, second])

    assert graph.get_vertices() == (first, second)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark8
@pytest.mark.mark10
def test_build_invalid_input() -> None:
    """
    Checks that invalid input to build raises ValueError
    """
    invalid_inputs = [True, 1, 4.0, "мама", (), []]
    matrix = TextGraph()
    for invalid_input in invalid_inputs:
        with pytest.raises(IncorrectArgumentError):
            matrix.build(invalid_input)
