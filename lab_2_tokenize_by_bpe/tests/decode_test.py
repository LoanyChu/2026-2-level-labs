"""
Checks the second lab's decode function
"""

# pylint: disable=redefined-outer-name, assignment-from-no-return
import json
from pathlib import Path

import pytest

from lab_2_tokenize_by_bpe.main import decode


@pytest.fixture(scope="function", autouse=True)
def setup() -> tuple:
    """
    Prepare test data.

    Returns:
        tuple: Correct data tuple.
    """
    path_to_tests_directory = Path(__file__).parent
    with open(path_to_tests_directory / "vocabulary.json", "r", encoding="utf-8") as json_file:
        vocabulary = json.load(json_file)
    with open(path_to_tests_directory / "encoded_text.json", "r", encoding="utf-8") as json_file:
        loaded_dict = json.load(json_file)
        encoded_ideal = loaded_dict["ideal_encoded_text"]
        encoded_with_unk = loaded_dict["encoded_text_with_unk"]
        encoded_without_end = loaded_dict["encoded_text_without_end"]

    return vocabulary, encoded_ideal, encoded_with_unk, encoded_without_end


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark8
@pytest.mark.mark10
def test_decode_ideal(setup: tuple) -> None:
    """
    Ideal decode scenario.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, encoded_ideal, _, _ = setup

    expected = (
        "В поисках пищи альбатросы способны преодолевать "
        "значительные расстояния при малой затрате сил, "
        "используя наклонное либо динамическое парение. "
        "Их крылья устроены так, что птица может долго "
        "зависать в воздухе, но не осиливает длительный "
        "маховый полет. Активный взмах крыльями альбатрос "
        "делает только при взлете, полагаясь далее на "
        "силу и направление ветра. "
    )

    actual = decode(encoded_ideal, vocabulary, "</s>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark8
@pytest.mark.mark10
def test_decode_with_unk(setup: tuple) -> None:
    """
    Decode scenario with unknown tokens.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, _, encoded_with_unk, _ = setup

    expected = (
        "Это интересно! Слово «альбатрос» произошло "
        "от арабского al-<unk>a<unk><unk><unk>s («ныряльщик»), "
        "которое на португальском наречии стало звучать как "
        "alcatra<unk>, перекочевав затем в английский и русский "
        "языки. Под влиянием латинского al<unk><unk>s («белый») "
        "alcatra<unk> чуть позднее превратился в al<unk>atross. "
        "Алькатрас – так назван остров в Калифорнии, где "
        "содержались особо опасные преступники. "
    )
    actual = decode(encoded_with_unk, vocabulary, "</s>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark8
@pytest.mark.mark10
def test_decode_without_end_token(setup: tuple) -> None:
    """
    Decode scenario without end token.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, _, _, encoded_without_end = setup

    expected = (
        "Впоискахпищиальбатросыспособныпреодолеватьзначительныерасстояния"
        "прималойзатратесил,используянаклонноелибодинамическоепарение.Их"
        "крыльяустроенытак,чтоптицаможетдолгозависатьввоздухе,нонеосиливает"
        "длительныймаховыйполет.Активныйвзмахкрыльямиальбатросделаеттолькопри"
        "взлете,полагаясьдалеенасилуинаправлениеветра."
    )
    actual = decode(encoded_without_end, vocabulary, None)
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark8
@pytest.mark.mark10
def test_decode_bad_input(setup: tuple) -> None:
    """
    Decode invalid inputs check.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, encoded_ideal, _, _ = setup

    encoded_text_bad_input = ["string", (), {}, None, 1, 1.1, True]
    vocabulary_bad_input = [None, (), 1.1, True, [None], "string", 1]
    end_of_word_bad_input = [(), {}, 1, 1.1, True, [None]]
    expected = None
    for index, bad_input in enumerate(encoded_text_bad_input):
        actual = decode(bad_input, vocabulary, "</s>")
        assert expected == actual

        actual = decode(encoded_ideal, vocabulary_bad_input[index], "</s>")
        assert expected == actual

    for bad_input in end_of_word_bad_input:
        actual = decode(encoded_ideal, vocabulary, bad_input)
        assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark8
@pytest.mark.mark10
def test_decode_return_value(setup: tuple) -> None:
    """
    Decode return value check.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, encoded_ideal, _, _ = setup

    actual = decode(encoded_ideal, vocabulary, "</s>")
    assert isinstance(actual, str)
