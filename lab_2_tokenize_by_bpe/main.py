"""
Lab 2.

BPE and machine translation evaluation
"""

# pylint:disable=unused-argument, too-many-return-statements, too-many-branches
from typing import Sequence


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
    if not isinstance(raw_word, str) or (start_of_word is not None and not isinstance(start_of_word, str)) or (end_of_word is not None and not isinstance(end_of_word, str)):
        return None

    result = []
    if start_of_word is not None:
        result.append(start_of_word)

    result.extend(tuple(raw_word))

    if end_of_word is not None:
        result.append(end_of_word)

    return tuple(result)


def collect_frequencies(
    text: str, start_of_word: str | None, end_of_word: str
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
    if not isinstance(raw_word, str) or (start_of_word is not None and not isinstance(start_of_word, str)) or (end_of_word is not None and not isinstance(end_of_word, str)):
        return None

    counts = {}
    for raw_word in text.split():
        prepared = prepare_word(raw_word, start_of_word, end_of_word)
        if prepared is None:
            return None
        if prepared in counts:
            counts[prepared] += 1
        else:
            counts[prepared] = 1

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
        if not isinstance(key, tuple):
            return None
        for token in key:
            if not isinstance(token, str):
                return None
        if not isinstance(value, int) or isinstance(value, bool):
            return None

    pair_counts = {}
    for word, freq in word_frequencies.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair in pair_counts:
                pair_counts[pair] += freq
            else:
                pair_counts[pair] = freq

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
    if not isinstance(word_frequencies, dict) or (not isinstance(pair, tuple) or len(pair) != 2) or (not isinstance(pair[0], str) or not isinstance(pair[1], str)):
        return None

    for key, value in word_frequencies.items():
        if not isinstance(key, tuple):
            return None
        for token in key:
            if not isinstance(token, str):
                return None
        if not isinstance(value, int) or isinstance(value, bool):
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
        if new_word in new_frequencies:
            new_frequencies[new_word] += count
        else:
            new_frequencies[new_word] = count

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
    if not isinstance(word_frequencies, dict) or (not isinstance(num_merges, int) or isinstance(num_merges, bool) or num_merges < 0):
        return None

    current = word_frequencies
    for _ in range(num_merges):
        pair_counts = count_tokens_pairs(current)
        if pair_counts is None:
            return None
        if not pair_counts:
            break

        sorted_pairs = sorted(pair_counts.items(), key=lambda x: (-x[1], x[0]))
        best_pair = sorted_pairs[0][0]

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
    if not isinstance(word_frequencies, dict) or (not isinstance(unknown_token, str)):
        return None

    for key, value in word_frequencies.items():
        if not isinstance(key, tuple):
            return None
        for token in key:
            if not isinstance(token, str):
                return None
        if not isinstance(value, int) or isinstance(value, bool):
            return None

    unique_tokens = set()
    for key in word_frequencies:
        for token in key:
            unique_tokens.add(token)

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
    if not isinstance(encoded_text, (list, tuple)) or (not isinstance(vocabulary, dict)) or (end_of_word_token is not None and not isinstance(end_of_word_token, str)):
        return None

    for token_id in encoded_text:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            return None

    for key, value in vocabulary.items():
        if not isinstance(key, str):
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            return None

    id_to_token = {}
    for token, token_id in vocabulary.items():
        id_to_token[token_id] = token

    result = []
    for token_id in encoded_text:
        if token_id not in id_to_token:
            return None

        token = id_to_token[token_id]

        if end_of_word_token is not None and token.endswith(end_of_word_token):
            piece = token[:-len(end_of_word_token)] if end_of_word_token else token
            result.append(piece)
            result.append(" ")
        else:
            result.append(token)

    return "".join(result)



def tokenize_word(
    word: tuple[str, ...], vocabulary: dict[str, int], end_of_word: str | None, unknown_token: str
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
    if not isinstance(word, tuple):
        return None
    for token in word:
        if not isinstance(token, str):
            return None
    if not isinstance(vocabulary, dict) or (end_of_word is not None and not isinstance(end_of_word, str)) or (not isinstance(unknown_token, str)):
        return None

    for key, value in vocabulary.items():
        if not isinstance(key, str) or (not isinstance(value, int) or isinstance(value, bool)):
            return None

    if unknown_token not in vocabulary:
        return None

    if end_of_word is not None and end_of_word != "" and end_of_word in word:
        if word[-1] != end_of_word:
            return None

    unknown_id = vocabulary[unknown_token]
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
            if (
                end_of_word is not None
                and end_of_word != ""
                and i < len(word)
                and word[i] == end_of_word
            ):
                return None
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
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return None

    vocabulary = {}
    for line in lines:
        if not line:
            continue

        if ":" in line:
            token, _, id_str = line.rpartition(":")
            id_str = id_str.strip()
        else:
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                return None
            token, id_str = parts
            id_str = id_str.strip()

        if token == "":
            return None

        try:
            token_id = int(id_str)
        except ValueError:
            return None

        vocabulary[token] = token_id

    return vocabulary


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
    if not isinstance(original_text, str):
        return None
    if not isinstance(vocabulary, dict):
        return None
    if start_of_word_token is not None and not isinstance(start_of_word_token, str):
        return None
    if end_of_word_token is not None and not isinstance(end_of_word_token, str):
        return None
    if not isinstance(unknown_token, str):
        return None

    for key, value in vocabulary.items():
        if not isinstance(key, str):
            return None
        if not isinstance(value, int) or isinstance(value, bool):
            return None

    if start_of_word_token is not None and start_of_word_token != "":
        if not any(token.startswith(start_of_word_token) for token in vocabulary):
            return None

    if end_of_word_token is not None and end_of_word_token != "":
        if not any(token.endswith(end_of_word_token) for token in vocabulary):
            return None

    result = []
    for raw_word in original_text.split():
        tokens = list(raw_word)

        if start_of_word_token is not None and start_of_word_token != "":
            tokens.insert(0, start_of_word_token)

        if end_of_word_token is not None and end_of_word_token != "":
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
    if not isinstance(text, str):
        return None
    if not isinstance(order, int) or isinstance(order, bool) or order <= 0:
        return None

    tokens = text.split()
    ngrams = []
    for i in range(len(tokens) - order + 1):
        ngrams.append(tuple(tokens[i:i + order]))

    return ngrams


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
        if not isinstance(ngram, tuple):
            return None
        for token in ngram:
            if not isinstance(token, str):
                return None

    for ngram in reference:
        if not isinstance(ngram, tuple):
            return None
        for token in ngram:
            if not isinstance(token, str):
                return None

    if len(actual) == 0:
        return 0.0

    reference_counts = {}
    for ngram in reference:
        if ngram in reference_counts:
            reference_counts[ngram] += 1
        else:
            reference_counts[ngram] = 1

    matches = 0
    for ngram in actual:
        if reference_counts.get(ngram, 0) > 0:
            matches += 1
            reference_counts[ngram] -= 1

    return matches / len(actual)



def calculate_geo_mean(precisions: Sequence[float], max_order: int) -> float | None:
    """
    Compute geometric mean of sequence of values.

    Args:
        precisions (Sequence[float]): A sequence of Precision values
        max_order (int): Maximum length of n-gram considered

    Returns:
        float | None: A value of geometric mean of Precision metric

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(precisions, (list, tuple)):
        return None
    if not isinstance(max_order, int) or isinstance(max_order, bool) or max_order <= 0:
        return None
    if len(precisions) != max_order:
        return None

    product = 1.0
    has_zero = False

    for value in precisions:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        if value < 0:
            return None
        if value == 0:
            has_zero = True
        else:
            product *= float(value)

    if has_zero:
        return 0.0

    return float(product ** (1.0 / max_order))


def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
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
    if actual is None:
        return None
    if not isinstance(actual, str):
        return None
    if not isinstance(reference, str):
        return None
    if not isinstance(max_order, int) or isinstance(max_order, bool) or max_order <= 0:
        return None

    precisions:list[float]=[]
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
    if geo_mean is None:
        return None
    if geo_mean == 0.0:
        return 0.0

    actual_len = len(actual.split())
    reference_len = len(reference.split())

    if actual_len == 0:
        return 0.0

    if actual_len >= reference_len:
        brevity_penalty:float = 1.0
    else:
        brevity_penalty = float(2.718281828459045 ** (1 - reference_len / actual_len))

    return brevity_penalty * geo_mean * 100
