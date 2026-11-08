"""
Lab 2.

BPE and machine translation evaluation
"""

import json
from typing import Sequence

# pylint:disable=unused-argument

def prepare_word(
    raw_word: str, start_of_word: str | None, end_of_word: str | None
) -> tuple[str, ...] | None:
    """
    Tokenize a word into characters and attach optional boundary tokens.

    Args:
        raw_word (str): Original word
        start_of_word (str | None): A token that signifies the start of word
        end_of_word (str | None): A token that signifies the end of word

    Returns:
        tuple[str, ...] | None: Preprocessed word

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(raw_word, str)
        or (start_of_word is not None and not isinstance(start_of_word, str))
        or (end_of_word is not None and not isinstance(end_of_word, str))
    ):
        return None

    result = []
    if start_of_word is not None:
        result.append(start_of_word)
    result.extend(tuple(raw_word))
    if end_of_word is not None:
        result.append(end_of_word)
    return tuple(result)


def collect_frequencies(
    text: str, start_of_word: str | None, end_of_word: str | None
) -> dict[tuple[str, ...], int] | None:
    """
    Count number of occurrences of each word.

    Args:
        text (str): Original text with no preprocessing
        start_of_word (str | None): A token that signifies the start of word
        end_of_word (str): A token that signifies the end of word

    Returns:
        dict[tuple[str, ...], int] | None: Dictionary where key - preprocessed word,
            value - number of occurrences

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if (
        not isinstance(text, str)
        or (start_of_word is not None and not isinstance(start_of_word, str))
        or (end_of_word is not None and not isinstance(end_of_word, str))
    ):
        return None

    counts = {}
    for w in text.split():
        prepared = prepare_word(w, start_of_word, end_of_word)
        if prepared is None:
            return None
        counts[prepared] = counts.get(prepared, 0) + 1
    return counts


def count_tokens_pairs(
    word_frequencies: dict[tuple[str, ...], int],
) -> dict[tuple[str, str], int] | None:
    """
    Count number of occurrences of each pair of subsequent tokens.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences

    Returns:
        dict[tuple[str, str], int] | None: A dictionary where
            key - token pair, value - number of occurrences

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(word_frequencies, dict):
        return None

    for key, value in word_frequencies.items():
        if (
            not isinstance(key, tuple)
            or not all(isinstance(t, str) for t in key)
            or not isinstance(value, int)
            or isinstance(value, bool)
        ):
            return None

    pair_counts = {}
    for word, freq in word_frequencies.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
    return pair_counts


def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Update word frequency dictionary by replacing a pair of tokens with a merged one.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences
        pair (tuple[str, str]): A pair of tokens to be merged

    Returns:
        dict[tuple[str, ...], int] | None: A dictionary where
            key - preprocessed word, value - number of occurrences

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(word_frequencies, dict)
        or not isinstance(pair, tuple)
        or len(pair) != 2
        or not isinstance(pair[0], str)
        or not isinstance(pair[1], str)
    ):
        return None

    for key, value in word_frequencies.items():
        if (
            not isinstance(key, tuple)
            or not all(isinstance(t, str) for t in key)
            or not isinstance(value, int)
            or isinstance(value, bool)
        ):
            return None

    merged_token = pair[0] + pair[1]
    new_frequencies = {}
    for word, count in word_frequencies.items():
        new_tokens = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(word[i])
                i += 1
        new_word = tuple(new_tokens)
        new_frequencies[new_word] = new_frequencies.get(new_word, 0) + count
    return new_frequencies


def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Create required number of new tokens by merging existing ones.

    Args:
        word_frequencies (dict[tuple[str, ...], int] | None): A dictionary where
            key - preprocessed word, value - number of occurrences
        num_merges (int): Required number of new tokens

    Returns:
        dict[tuple[str, ...], int] | None: A dictionary where
            key - preprocessed word, value - number of occurrences

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if (
        not isinstance(word_frequencies, dict)
        or not isinstance(num_merges, int)
        or isinstance(num_merges, bool)
        or num_merges < 0
    ):
        return None

    current = word_frequencies
    for _ in range(num_merges):
        pair_counts = count_tokens_pairs(current)
        if pair_counts is None:
            return None
        if not pair_counts:
            break

        best_pair = sorted(pair_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
        merged = merge_tokens(current, best_pair)
        if merged is None:
            return None
        current = merged
    return current

def get_vocabulary(
    word_frequencies: dict[tuple[str, ...], int], unknown_token: str
) -> dict[str, int] | None:
    """
    Establish correspondence between tokens and its integer identifier.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences
        unknown_token (str): A token to signify an unknown token

    Returns:
        dict[str, int] | None: A dictionary where key - token, value - identifier

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(word_frequencies, dict) or not isinstance(unknown_token, str):
        return None

    for key, value in word_frequencies.items():
        if (
            not isinstance(key, tuple)
            or not all(isinstance(t, str) for t in key)
            or not isinstance(value, int)
            or isinstance(value, bool)
        ):
            return None

    unique_tokens = set()
    for key in word_frequencies:
        unique_tokens.update(key)

    vocabulary = {unknown_token: 0}
    next_id = 1
    for token in sorted(unique_tokens):
        if token == unknown_token:
            continue
        vocabulary[token] = next_id
        next_id += 1
    return vocabulary


def decode(
    encoded_text: Sequence[int] | None,
    vocabulary: dict[str, int] | None,
    end_of_word_token: str | None,
) -> str | None:
    """
    Translate encoded sequence into decoded one.

    Args:
        encoded_text (Sequence[int] | None): A sequence of token identifiers
        vocabulary (dict[str, int] | None): A dictionary where
            key - token, value - identifier
        end_of_word_token (str | None): An end-of-word token

    Returns:
        str | None: Decoded sequence

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(encoded_text, (list, tuple))
        or not isinstance(vocabulary, dict)
        or (end_of_word_token is not None and not isinstance(end_of_word_token, str))
    ):
        return None

    if not all(isinstance(t, int) and not isinstance(t, bool) for t in encoded_text):
        return None

    for key, value in vocabulary.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
        ):
            return None

    id_to_token = {token_id: token for token, token_id in vocabulary.items()}
    result = []
    for token_id in encoded_text:
        # Если ID нет, просто пропускаем (или можно вернуть None, если тест требует)
        if token_id in id_to_token:
            token = id_to_token[token_id]
            if end_of_word_token is not None and token == end_of_word_token:
                result.append(" ")
            else:
                result.append(token)

    return "".join(result)


def tokenize_word(
    word: tuple[str, ...],
    vocabulary: dict[str, int],
    end_of_word: str | None,
    unknown_token: str,
) -> list[int] | None:
    """
    Split word into tokens.

    Args:
        word (tuple[str, ...]): Preprocessed word
        vocabulary (dict[str, int]): A dictionary where key - token, value - identifier
        end_of_word (str | None): An end-of-word token
        unknown_token (str): A token that signifies unknown sequence

    Returns:
        list[int] | None: A list of token identifiers

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(word, tuple)
        or not all(isinstance(t, str) for t in word)
        or not isinstance(vocabulary, dict)
    ):
        return None

    if (
        (end_of_word is not None and not isinstance(end_of_word, str))
        or not isinstance(unknown_token, str)
    ):
        return None

    for key, value in vocabulary.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
        ):
            return None

    if unknown_token not in vocabulary:
        return None

    unknown_id = vocabulary.get(unknown_token, 0)

    result = []
    i = 0
    while i < len(word):
        match = None
        match_len = 0
        for j in range(len(word), i, -1):
            candidate = "".join(word[i:j])
            if candidate in vocabulary:
                match = vocabulary[candidate]
                match_len = j - i
                break
        if match is None:
            result.append(unknown_id)
            i += 1
        else:
            result.append(match)
            i += match_len
    return result


def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Read and retrieve dictionary of type <token: identifier>.

    Args:
        vocab_path (str): A path to the saved vocabulary

    Returns:
        dict[str, int] | None: A dictionary where key - token, value - identifier

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(vocab_path, str):
        return None

    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    for key, value in data.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
        ):
            return None

    return data


def encode(
    original_text: str,
    vocabulary: dict[str, int] | None,
    start_of_word_token: str | None,
    end_of_word_token: str | None,
    unknown_token: str,
) -> list[int] | None:
    """
    Translate original text into a sequence of token identifiers.

    Args:
        original_text (str): Original text
        vocabulary (dict[str, int] | None): A dictionary where key - token, value - identifier
        start_of_word_token (str | None): A start-of-word token
        end_of_word_token (str | None): An end-of-word token
        unknown_token (str): A token that signifies unknown sequence

    Returns:
        list[int] | None: A list of token identifiers

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if (
        not isinstance(original_text, str)
        or not isinstance(vocabulary, dict)
        or not isinstance(unknown_token, str)
    ):
        return None

    if (
        start_of_word_token is not None and not isinstance(start_of_word_token, str)
    ) or (end_of_word_token is not None and not isinstance(end_of_word_token, str)):
        return None

    for key, value in vocabulary.items():
        if (
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
        ):
            return None

    result = []
    for w in original_text.split():
        tokens = list(w)
        if start_of_word_token is not None:
            tokens.insert(0, start_of_word_token)
        if end_of_word_token is not None:
            tokens.append(end_of_word_token)

        word = tuple(tokens)
        encoded_word = tokenize_word(word, vocabulary, end_of_word_token, unknown_token)
        if encoded_word is None:
            return None
        result.extend(encoded_word)
    return result


def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extract n-grams from the given sequence.

    Args:
        text (str): Original text
        order (int): Required number of elements in a single n-gram

    Returns:
        list[tuple[str, ...]] | None: A sequence of n-grams

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(text, str)
        or not isinstance(order, int)
        or isinstance(order, bool)
        or order <= 0
    ):
        return None

    tokens = text.split()
    return [tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1)]


def calculate_precision(
    actual: Sequence[tuple[str, ...]], reference: Sequence[tuple[str, ...]]
) -> float | None:
    """
    Compare two sequences by virtue of Precision metric.

    Args:
        actual (Sequence[tuple[str, ...]]): Predicted sequence of n-grams
        reference (Sequence[tuple[str, ...]]): Expected sequence of n-grams

    Returns:
        float | None: Value of Precision metric

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(actual, (list, tuple)):
        return None
    if not isinstance(reference, (list, tuple)):
        return None

    for ngram in actual:
        if not isinstance(ngram, tuple) or not all(isinstance(t, str) for t in ngram):
            return None

    for ngram in reference:
        if not isinstance(ngram, tuple) or not all(isinstance(t, str) for t in ngram):
            return None

    if len(actual) == 0:
        return 0.0

    reference_counts = {}
    for ngram in reference:
        reference_counts[ngram] = reference_counts.get(ngram, 0) + 1

    matches = 0
    for ngram in actual:
        if reference_counts.get(ngram, 0) > 0:
            matches += 1
            reference_counts[ngram] -= 1

    return matches / len(actual)


def calculate_geo_mean(
    precisions: Sequence[float], max_order: int
) -> float | None:
    """
    Compute geometric mean of sequence of values.

    Args:
        precisions (Sequence[float]): A sequence of Precision values
        max_order (int): Maximum length of n-gram considered

    Returns:
        float | None: A value of geometric mean of Precision metric

    In case of corrupt input arguments, None is returned
    """
    if (
        not isinstance(precisions, (list, tuple))
        or not isinstance(max_order, int)
        or isinstance(max_order, bool)
        or max_order <= 0
        or len(precisions) != max_order
    ):
        return None

    product = 1.0
    has_zero = False

    for value in precisions:
        # Если есть отрицательное число - сразу None
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
        ):
            return None

        if value == 0:
            has_zero = True
        else:
            product *= float(value)

    if has_zero:
        return 0.0

    return float(product ** (1.0 / max_order))


def calculate_bleu(
    actual: str | None, reference: str, max_order: int = 3
) -> float | None:
    """
    Compare two sequences by virtue of BLEU metric.

    Args:
        actual (str | None): Predicted sequence
        reference (str): Expected sequence
        max_order (int, optional): Max length of n-gram to consider for comparison. Defaults to 3

    Returns:
        float | None: A value of BLEU metric

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
    if (
        not isinstance(actual, str)
        or not isinstance(reference, str)
        or not isinstance(max_order, int)
        or isinstance(max_order, bool)
        or max_order <= 0
    ):
        return None

    precisions = []
    for order in range(1, max_order + 1):
        actual_ngrams = collect_ngrams(actual, order)
        reference_ngrams = collect_ngrams(reference, order)
        if actual_ngrams is None or reference_ngrams is None:
            return None

        precision = calculate_precision(actual_ngrams, reference_ngrams)
        if precision is None:
            return None
        precisions.append(precision)

    geo_mean = calculate_geo_mean(precisions, max_order)
    if geo_mean is None or geo_mean == 0.0:
        return geo_mean

    actual_len = len(actual.split())
    reference_len = len(reference.split())

    if actual_len == 0:
        return 0.0

    if actual_len >= reference_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = float(
            2.718281828459045 ** (1 - reference_len / actual_len)
        )

    return float(brevity_penalty * geo_mean * 100)
#o
