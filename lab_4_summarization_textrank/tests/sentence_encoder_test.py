# pylint: disable=protected-access
"""
Checks the fourth lab SentenceEncoder abstraction
"""

import pytest

from lab_4_summarization_textrank.main import IncorrectArgumentError, Sentence, SentenceEncoder

ENCODED_SENTENCES = [
    (1011, 1014, 1010, 1015, 1006, 1008, 1007, 1002, 1003, 1001, 1012, 1009, 1004, 1005, 1013),
    (1027, 1028, 1025, 1016, 1019, 1017, 1020, 1023, 1021, 1022, 1026, 1024, 1018),
    (1036, 1038, 1037, 1010, 1029, 1033, 1031, 1039, 1004, 1034, 1032, 1035, 1030),
    (1041, 1046, 1051, 1052, 1050, 1047, 1042, 1044, 1048, 1040, 1045, 1043, 1049),
]


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_sentence_encoder_initialization_ideal(sentence_encoder: SentenceEncoder) -> None:
    """
    Checks SentenceEncoder initialization and its attributes

    Args:
        sentence_encoder (SentenceEncoder): An instance of SentenceEncoder
    """
    for attribute in ["_word2id", "_id2word"]:
        assert hasattr(
            sentence_encoder, attribute
        ), f"Sentence encoder misses {attribute} attribute"


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_encode_sentences_value_error(sentence_encoder: SentenceEncoder) -> None:
    """
    Checks that invalid input to encode_sentences raises IncorrectArgumentError

    Args:
        sentence_encoder (SentenceEncoder): An instance of SentenceEncoder
    """
    for wrong_input in [
        "text",
        1,
        4.0,
        ["word1", "word2"],
    ]:
        with pytest.raises(IncorrectArgumentError):
            sentence_encoder.encode_sentences(wrong_input)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_encode_sentences_with_empty_preprocessed_sentence(
    sentence_encoder: SentenceEncoder,
) -> None:
    """Checks that a valid sentence with no tokens is encoded as an empty tuple

    Args:
        sentence_encoder (SentenceEncoder): An instance of SentenceEncoder
    """
    sentence = Sentence("", 0)
    sentence_encoder.encode_sentences([sentence])
    assert sentence.get_encoded() == ()


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_encode_sentences_repeated_tokens(sentence_encoder: SentenceEncoder) -> None:
    """Checks that repeated unseen tokens share a single identifier.

    Args:
        sentence_encoder (SentenceEncoder): An instance of SentenceEncoder.
    """
    sentence = Sentence("repeat", 0)
    sentence.set_preprocessed(("same", "same"))
    sentence_encoder.encode_sentences([sentence])

    assert sentence.get_encoded()[0] == sentence.get_encoded()[1]
    assert len(sentence_encoder._word2id) == len(sentence_encoder._id2word)


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark6
@pytest.mark.mark8
@pytest.mark.mark10
def test_learn_indices_return_value(sentence_encoder: SentenceEncoder) -> None:
    """
    Checks that invalid input to _learn_indices raises ValueError

    Args:
        sentence_encoder (SentenceEncoder): An instance of SentenceEncoder
    """
    invalid_inputs = [[123], True, 1, 0.0, {6: 5}]
    for invalid_input in invalid_inputs:
        with pytest.raises(IncorrectArgumentError):
            sentence_encoder._learn_indices(invalid_input)
