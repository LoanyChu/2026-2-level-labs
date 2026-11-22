# pylint: disable=protected-access
"""
Checks the fourth lab NaiveSummarizer abstraction
"""

import pytest

from lab_4_summarization_textrank.main import (
    IncorrectArgumentError,
    NaiveSummarizer,
    SentenceEncoder,
    SentencePreprocessor,
)

EXTRACT = (
    "Юрий Алексеевич Гагарин является первым космонавтом земли.\n",
    "Он родился 9 марта 1934 года в городе Гжатске (ныне Гагарин) Смоленской области.\n",
    "В июне Гагарин с отличием окончил Саратовский индустриальный техникум, в "
    + "июле – совершил первый самостоятельный полет на самолете Як-18 и 10 октября "
    + "того же года окончил Саратовский аэроклуб.\n",
    "В конце 1957 года Гагарин прибыл к месту своего назначения – истребительный "
    + "авиационный полк Северного флота.\n",
    "Гагарин 9 декабря 1959 года написал заявление с просьбой зачислить его в "
    + "группу кандидатов в космонавты.\n",
    "Уже в конце апреля Юрий Гагарин отправился в свою первую зарубежную поездку.\n",
    "С 23 мая 1961 года Гагарин – командир отряда космонавтов.\n",
    "Одним из тех, кто стал готовиться к полету на Луну, стал и Гагарин.\n",
    "К нему готовились Владимир Михайлович Комаров и Юрий Алексеевич Гагарин.\n",
    "Но чтобы не произошло в тот день, ясно только одно – погиб первый космонавт "
    + "планеты Земля Юрий Алексеевич Гагарин.",
)

SUMMARY = "".join(EXTRACT)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark8
@pytest.mark.mark10
def test_naive_summarizer_initialization_value_error() -> None:
    """
    Checks NaiveSummarizer initialization with invalid input
    """
    wrong_inputs = [True, 1, "dlsk;l", SentenceEncoder(), SentencePreprocessor((), ())]
    for wrong_input in wrong_inputs:
        with pytest.raises(IncorrectArgumentError):
            NaiveSummarizer(wrong_input)
