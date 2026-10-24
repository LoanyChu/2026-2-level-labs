"""
Checks the correctness of Beam Search Algorithm
"""

# pylint: disable=too-few-public-methods, no-member, redefined-outer-name

import pytest

from lab_3_generate_by_ngrams.main import BeamSearcher


class FakeNGramLanguageModel:
    """
    Store language model by n_grams, predict the next letter.
    Attributes:
        __n_gram_size (int): A size of n-grams to use for language modelling
        _n_gram_frequencies (dict): Frequencies for n-grams
        _encoded_corpus (tuple): Encoded text
    """

    def __init__(self, encoded_corpus: tuple | None, n_gram_size: int) -> None:
        """
        Initialize an instance of NGramLanguageModel.

        Args:
            encoded_corpus (tuple | None): Encoded text
            n_gram_size (int): A size of n-grams to use for language modelling
        """
        self.__n_gram_size = n_gram_size
        self._encoded_corpus = encoded_corpus
        self._n_gram_frequencies = {}

    def generate_next_token(self, sequence: tuple[int, ...]) -> dict | None:
        """
        Rewritten generate_next_token method

        Args:
            sequence (tuple[int, ...]): sequence to continue

        Returns:
            dict | None: next tokens with their probabilities or None
        """
        sequence = tuple(sequence[-(self.__n_gram_size - 1) :])
        next_tokens = {
            (1,): {2: 0.5, 3: 0.3, 4: 0.2},
            (2,): {5: 0.5, 6: 0.2, 7: 0.1},
            (3,): {8: 0.9, 9: 0.7, 10: 0.2},
            (4,): {11: 1.5, 12: 0.5, 13: 0.5},
        }
        return next_tokens.get(sequence)


@pytest.fixture
def fake_model() -> FakeNGramLanguageModel:
    """
    Returns an instance of FakeNGramLanguageModel
    """
    return FakeNGramLanguageModel(
        encoded_corpus=(0, 1, 2, 3, 4, 8, 6, 7, 8, 9, 10, 11, 12, 13), n_gram_size=2
    )


@pytest.mark.lab_3_generate_by_ngrams
@pytest.mark.mark8
@pytest.mark.mark10
def test_beam_searcher_algorithm(fake_model: FakeNGramLanguageModel) -> None:
    """
    Checks the operation of the algorithm

    Args:
        fake_model (FakeNGramLanguageModel): An instance of FakeNGramLanguageModel
    """
    expected = [(0, 1, 2, 5), (0, 1, 3, 8), (0, 1, 4, 11)]
    for beam_width in range(1, 4):
        fake_bm = BeamSearcher(beam_width=beam_width, language_model=fake_model)
        seq_candidates = {(0, 1): 0.0}
        sequences_to_continue = list(seq_candidates.keys())
        for _ in range(1, 3):
            for seq in sequences_to_continue:
                next_tokens = fake_bm.get_next_token(seq)
                seq_candidates = fake_bm.continue_sequence(
                    sequence=seq, next_tokens=next_tokens, sequence_candidates=seq_candidates
                )
            pruned_candidates = fake_bm.prune_sequence_candidates(seq_candidates)
            seq_candidates = pruned_candidates
            sequences_to_continue = list(seq_candidates)
        actual = min(seq_candidates.items(), key=lambda x: (x[1], x[0]))[0]
        assert expected[beam_width - 1] == actual
