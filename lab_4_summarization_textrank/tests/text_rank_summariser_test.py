# pylint: disable=protected-access
"""
Checks the fourth lab TextRankSummarizer abstraction
"""

import pytest

from lab_4_summarization_textrank.main import (
    IncorrectArgumentError,
    NaiveSummarizer,
    Sentence,
    SentenceEncoder,
    SentencePreprocessor,
    TextGraph,
    TextRankSummarizer,
)

EXTRACT = (
    "Он родился 9 марта 1934 года в городе Гжатске (ныне Гагарин) Смоленской области.\n",
    "В июне Гагарин с отличием окончил Саратовский индустриальный техникум, в "
    + "июле – совершил первый самостоятельный полет на самолете Як-18 и 10 октября "
    + "того же года окончил Саратовский аэроклуб.\n",
    "Юрий Алексеевич 7 октября 1955 года был призван в ряды Советской Армии и "
    + "направлен в город Оренбург на учебу в 1-е Чкаловское военно-авиационное "
    + "училище летчиков имени К.Е.Ворошилова.\n",
    "В конце 1957 года Гагарин прибыл к месту своего назначения – истребительный "
    + "авиационный полк Северного флота.\n",
    "Гагарин 9 декабря 1959 года написал заявление с просьбой зачислить его в "
    + "группу кандидатов в космонавты.\n",
    "В 9 часов 7 минут по московскому времени 12 апреля 1961 года с космодрома "
    + 'Байконур стартовал космический корабль "Восток" с пилотом–космонавтом Юрием '
    + "Алексеевичем Гагариным на борту.\n",
    "С 23 мая 1961 года Гагарин – командир отряда космонавтов.\n",
    "Но сделано это было уже без Юрия Гагарина. 17 февраля 1968 года он защитил "
    + "диплом в Академии имени Н.Е.Жуковского.\n",
    "С большим трудом добился разрешения самостоятельно пилотировать самолет. 27 "
    + "марта 1968 года был первый такой полет.\n",
    "Но чтобы не произошло в тот день, ясно только одно – погиб первый космонавт "
    + "планеты Земля Юрий Алексеевич Гагарин.",
)

SUMMARY = "".join(EXTRACT)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark10
def test_text_rank_inheritance() -> None:
    """
    Checks that TextRankSummarizer inherits from NaiveSummarizer
    """
    assert issubclass(TextRankSummarizer, NaiveSummarizer)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark10
def test_text_rank_initialization() -> None:
    """
    Checks TextRankSummarizer initialization
    """
    sim_graph = TextGraph()
    text_rank = TextRankSummarizer(sim_graph)
    for attribute in [
        "_graph",
        "_damping_factor",
        "_convergence_threshold",
        "_max_iter",
        "_scores",
    ]:
        assert hasattr(text_rank, attribute)
    assert text_rank._damping_factor == 0.85
    assert text_rank._convergence_threshold == 0.0001
    assert text_rank._max_iter == 50


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark10
def test_text_rank_initialization_value_error() -> None:
    """
    Checks TextRankSummarizer initialization with invalid input
    """
    wrong_inputs = [True, 1, "dlsk;l ", SentenceEncoder(), SentencePreprocessor((), ())]
    for wrong_input in wrong_inputs:
        with pytest.raises(IncorrectArgumentError):
            TextRankSummarizer(wrong_input)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark10
def test_text_rank_update_vertex_score_uses_documented_degree() -> None:
    """Checks the TextRank update against the documented degree denominator."""
    vertex = Sentence("vertex", 0)
    neighbor = Sentence("neighbor", 1)
    vertex.set_encoded((1, 2))
    neighbor.set_encoded((1, 3))
    graph = TextGraph()
    graph.build([vertex, neighbor])
    summarizer = TextRankSummarizer(graph)

    summarizer._update_vertex_score(vertex, [neighbor], {vertex: 1.0, neighbor: 2.0})

    assert summarizer._scores[vertex] == pytest.approx(1.85)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark10
def test_text_rank_does_not_count_self_as_neighbor() -> None:
    """Checks that a lone vertex receives only the base TextRank score."""
    vertex = Sentence("vertex", 0)
    vertex.set_encoded((1,))
    graph = TextGraph()
    graph.build([vertex])
    summarizer = TextRankSummarizer(graph)

    summarizer.train()

    assert summarizer._scores[vertex] == pytest.approx(0.15)
