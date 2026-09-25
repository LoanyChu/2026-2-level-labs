"""
Checks the second lab's encode function
"""

# pylint: disable=redefined-outer-name, assignment-from-no-return
import json
from pathlib import Path
from unittest import mock

import pytest

from lab_2_tokenize_by_bpe.main import encode


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

    ideal_original_text = (
        "Активный взмах крыльями альбатрос делает только при взлете, "
        "полагаясь далее на силу и направление ветра."
    )
    original_text_with_unk = (
        "Под влиянием латинского albus («белый») alcatraz"
        + " чуть позднее превратился в albatross."
    )
    arbitrary_text = "你好！我是俄罗斯人。我住在下诺夫哥罗德。我喜欢程序设计。"

    return vocabulary, ideal_original_text, original_text_with_unk, arbitrary_text


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_encode_ideal(setup: tuple) -> None:
    """
    Ideal encode scenario.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, ideal_original_text, _, _ = setup

    expected = [
        144,
        179,
        95,
        171,
        79,
        21,
        171,
        176,
        181,
        169,
        28,
        179,
        89,
        72,
        200,
        7,
        0,
        25,
        55,
        68,
        4,
        96,
        72,
        179,
        24,
        184,
        185,
        20,
        171,
        176,
        180,
        60,
        174,
        14,
        83,
        68,
        172,
        169,
        200,
        186,
        30,
        54,
        69,
        19,
        8,
        91,
        180,
        27,
        20,
        75,
        184,
        84,
        171,
        180,
        37,
        19,
        171,
        60,
        84,
        15,
    ]
    actual = encode(ideal_original_text, vocabulary, None, "</s>", "<unk>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_encode_with_unk(setup: tuple) -> None:
    """
    Encode scenario with unknown tokens.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, _, original_text_with_unk, _ = setup

    expected = [
        158,
        81,
        34,
        171,
        70,
        200,
        77,
        174,
        23,
        180,
        47,
        63,
        43,
        3,
        129,
        134,
        16,
        16,
        139,
        34,
        101,
        142,
        170,
        174,
        180,
        196,
        178,
        143,
        102,
        34,
        129,
        134,
        130,
        129,
        140,
        138,
        129,
        16,
        34,
        192,
        188,
        187,
        30,
        83,
        176,
        173,
        76,
        19,
        184,
        85,
        171,
        185,
        47,
        177,
        180,
        11,
        18,
        129,
        134,
        16,
        129,
        140,
        138,
        137,
        139,
        139,
        15,
    ]
    actual = encode(original_text_with_unk, vocabulary, None, "</s>", "<unk>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_encode_arbitrary_text(setup: tuple) -> None:
    """
    Encode scenario for absolutely arbitrary text.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, _, _, arbitrary_text = setup

    expected = [
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        34,
    ]
    actual = encode(arbitrary_text, vocabulary, None, "</s>", "<unk>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_encode_none_prepare_word(setup: tuple) -> None:
    """
    Encode with None as prepare_word's return value.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, _, _, arbitrary_text = setup

    expected = None
    with mock.patch("lab_2_tokenize_by_bpe.main.prepare_word", return_value=None):
        actual = encode(arbitrary_text, vocabulary, None, "</s>", "<unk>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_encode_none_tokenize_word(setup: tuple) -> None:
    """
    Encode with None as tokenize_word's return value.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, _, _, arbitrary_text = setup

    expected = None
    with mock.patch("lab_2_tokenize_by_bpe.main.tokenize_word", return_value=None):
        actual = encode(arbitrary_text, vocabulary, None, "</s>", "<unk>")
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_encode_bad_input(setup: tuple) -> None:
    """
    Encode invalid inputs check.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, _, _, _ = setup

    original_text_bad_input = [(), [None], {}, None, 1, 1.1, True]
    vocabulary_bad_input = [None, (), 1.1, True, [None], "string", 1]
    start_end_bad_input = [(), {}, 1, 1.1, True, [None]]
    unknown_bad_input = [(), {}, None, 1, 1.1, True, [None]]
    expected = None
    for index, bad_input in enumerate(original_text_bad_input):
        actual = encode(bad_input, vocabulary, None, "</s>", "<unk>")
        assert expected == actual

        actual = encode(
            "Активный взмах крыльями альбатрос делает только при взлете, "
            "полагаясь далее на силу и направление ветра.",
            vocabulary_bad_input[index],
            None,
            "</s>",
            "<unk>",
        )
        assert expected == actual

        actual = encode(
            "Активный взмах крыльями альбатрос делает только при взлете, "
            "полагаясь далее на силу и направление ветра.",
            vocabulary,
            None,
            "</s>",
            unknown_bad_input[index],
        )
        assert expected == actual

    for bad_input in start_end_bad_input:
        actual = encode(
            "Активный взмах крыльями альбатрос делает только при взлете, "
            "полагаясь далее на силу и направление ветра.",
            vocabulary,
            bad_input,
            "</s>",
            "<unk>",
        )
        assert expected == actual

        actual = encode(
            "Активный взмах крыльями альбатрос делает только при взлете, "
            "полагаясь далее на силу и направление ветра.",
            vocabulary,
            None,
            bad_input,
            "<unk>",
        )
        assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_encode_return_value(setup: tuple) -> None:
    """
    Encode return value check.

    Args:
        setup (tuple): Prepared test data.
    """
    vocabulary, ideal_original_text, _, _ = setup

    actual = encode(ideal_original_text, vocabulary, None, "</s>", "<unk>")
    assert isinstance(actual, list)
