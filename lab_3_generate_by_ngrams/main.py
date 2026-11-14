"""
Lab 3.

Beam-search and natural language generation evaluation
"""

# pylint:disable=too-few-public-methods

import math

class TextProcessor:
    """
    Handle text tokenization, encoding and decoding.

    Attributes:
        _end_of_word_token (str): A token denoting word boundary
        _storage (dict): Dictionary in the form of <token: identifier>
    """

    def __init__(self, end_of_word_token: str) -> None:
        """
        Initialize an instance of TextProcessor.

        Args:
            end_of_word_token (str): A token denoting word boundary
        """
        self._end_of_word_token = end_of_word_token
        self._storage = {end_of_word_token: 0}

    def _tokenize(self, text: str) -> tuple[str, ...] | None:
        """
        Tokenize text into unigrams, separating words with special token.

        Punctuation and digits are removed. EoW token is appended after the last word in two cases:
        1. It is followed by punctuation
        2. It is followed by space symbol

        Args:
            text (str): Original text

        Returns:
            tuple[str, ...] | None: Tokenized text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(text, str) or not text:
            return None

        text_lower = text.lower()

        fragments = []
        current = []
        for char in text_lower:
            if char.isspace():
                if current:
                    fragments.append(current)
                    current = []
            else:
                current.append(char)
        if current:
            fragments.append(current)

        if not fragments:
            return None

        ends_special = not text[-1].isalnum()

        tokens = []
        total = len(fragments)
        for idx, frag in enumerate(fragments):
            letters = [c for c in frag if c.isalpha()]
            if not letters:
                continue
            tokens.extend(letters)
            if idx == total - 1:
                if ends_special:
                    tokens.append(self._end_of_word_token)
            else:
                tokens.append(self._end_of_word_token)

        if not tokens:
            return None
        return tuple(tokens)

    def get_id(self, element: str) -> int | None:
        """
        Retrieve a unique identifier of an element.

        Args:
            element (str): String element to retrieve identifier for

        Returns:
            int | None: Integer identifier that corresponds to the given element

        In case of corrupt input arguments or arguments not included in storage,
        None is returned
        """
        if not isinstance(element, str):
            return None
        return self._storage.get(element)

    def get_end_of_word_token(self) -> str:  # type: ignore[empty-body]
        """
        Retrieve value stored in self._end_of_word_token attribute.

        Returns:
            str: EoW token
        """
        return self._end_of_word_token

    def get_token(self, element_id: int) -> str | None:
        """
        Retrieve an element by unique identifier.

        Args:
            element_id (int): Identifier to retrieve identifier for

        Returns:
            str | None: Element that corresponds to the given identifier

        In case of corrupt input arguments or arguments not included in storage, None is returned
        """
        if not isinstance(element_id, int) or isinstance(element_id, bool):
            return None
        for token, token_id in self._storage.items():
            if token_id == element_id:
                return token
        return None

    def encode(self, text: str) -> tuple[int, ...] | None:
        """
        Encode text.

        Tokenize text, assign each symbol an integer identifier and
        replace letters with their ids.

        Args:
            text (str): An original text to be encoded

        Returns:
            tuple[int, ...] | None: Processed text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(text, str):
            return None
        tokens = self._tokenize(text)
        if tokens is None:
            return None

        result = []
        for token in tokens:
            self._put(token)
            token_id = self.get_id(token)
            if token_id is None:
                return None
            result.append(token_id)
        return tuple(result)

    def _put(self, element: str) -> None:
        """
        Put an element into the storage, assign a unique id to it.

        Args:
            element (str): An element to put into storage

        In case of corrupt input arguments or invalid argument length,
        an element is not added to storage
        """
        if not isinstance(element, str):
            return
        if len(element) != 1:
            return
        if element not in self._storage:
            self._storage[element] = len(self._storage)

    def decode(self, encoded_corpus: tuple[int, ...]) -> str | None:
        """
        Decode and postprocess encoded corpus by converting integer identifiers to string.

        Special symbols are replaced with spaces (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        decoded = self._decode(encoded_corpus)
        if decoded is None:
            return None
        return self._postprocess_decoded_text(decoded)

    def fill_from_ngrams(self, content: dict) -> None:
        """
        Fill internal storage with letters from external JSON.

        Args:
            content (dict): ngrams from external JSON
        """
        if not isinstance(content, dict):
            return
        freq = content.get("freq")
        if not isinstance(freq, dict):
            return
        for ngram in freq:
            if not isinstance(ngram, str):
                continue
            for char in ngram:
                if char.isalpha():
                    self._put(char.lower())

    def _decode(self, corpus: tuple[int, ...]) -> tuple[str, ...] | None:
        """
        Decode sentence by replacing ids with corresponding letters.

        Args:
            corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            tuple[str, ...] | None: Sequence with decoded tokens

        In case of corrupt input arguments, None is returned.
        In case any of methods used return None, None is returned.
        """
        if not isinstance(corpus, tuple):
            return None
        result = []
        for token_id in corpus:
            token = self.get_token(token_id)
            if token is None:
                return None
            result.append(token)
        return tuple(result)

    def _postprocess_decoded_text(self, decoded_corpus: tuple[str, ...]) -> str | None:
        """
        Convert decoded sentence into the string sequence.

        Special symbols are replaced with spaces (no multiple spaces in a row are allowed).
        The first letter is capitalized, resulting sequence must end with a full stop.

        Args:
            decoded_corpus (tuple[str, ...]): A tuple of decoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(decoded_corpus, tuple) or not decoded_corpus:
            return None

        parts = []
        for token in decoded_corpus:
            if not isinstance(token, str):
                return None
            if token == self._end_of_word_token:
                parts.append(" ")
            else:
                parts.append(token)

        text = " ".join("".join(parts).split())
        if not text:
            return None

        text = text[0].upper() + text[1:]
        if not text.endswith("."):
            text += "."
        return text


class WordProcessor(TextProcessor):
    """
    Handle text tokenization, encoding and decoding on the level of words.

    Attributes:
        _end_of_word_token (str): A token denoting sentence boundary
        _storage (dict): Dictionary in the form of <token: identifier>
    """

    def _tokenize(self, text: str) -> tuple[str, ...] | None:
        """
        Tokenize text into words, separating sentences with special token.

        Punctuation and digits are removed. Special token is appended after the word
        that ends a sentence, i.e. is followed by '.', '!' or '?'.

        Args:
            text (str): Original text

        Returns:
            tuple[str, ...] | None: Tokenized text

        In case of corrupt input arguments, None is returned.
        In case text does not contain any words, None is returned.
        """
        if not isinstance(text, str) or not text:
            return None

        fragments = text.split()
        if not fragments:
            return None

        tokens = []
        for fragment in fragments:
            letters = [char.lower() for char in fragment if char.isalpha()]
            if not letters:
                continue
            word = "".join(letters)
            tokens.append(word)

            ends_sentence = fragment[-1] in ".!?"
            if ends_sentence and (not tokens or tokens[-1] != self._end_of_word_token):
                tokens.append(self._end_of_word_token)

        if not tokens:
            return None
        return tuple(tokens)

    def _put(self, element: str) -> None:
        """
        Put a word into the storage, assign a unique id to it.

        Args:
            element (str): A word to put into storage

        In case of corrupt input arguments, an element is not added to storage
        """
        if not isinstance(element, str) or not element:
            return
        if element not in self._storage:
            self._storage[element] = len(self._storage)

    def _postprocess_decoded_text(self, decoded_corpus: tuple[str, ...]) -> str | None:
        """
        Convert decoded sequence of words into the string sequence.

        Words are separated with spaces, special tokens are replaced with full stops.
        Each sentence is capitalized, resulting sequence must end with a full stop.

        Args:
            decoded_corpus (tuple[str, ...]): A tuple of decoded tokens

        Returns:
            str | None: Resulting text

        In case of corrupt input arguments, None is returned.
        In case decoded corpus does not contain any words, None is returned.
        """
        if not isinstance(decoded_corpus, tuple) or not decoded_corpus:
            return None

        parts = []
        capitalize_next = True
        word_count = 0

        for token in decoded_corpus:
            if not isinstance(token, str):
                return None
            if token == self._end_of_word_token:
                if parts:
                    parts[-1] = parts[-1] + "."
                capitalize_next = True
            else:
                word = token
                if capitalize_next and word:
                    word = word[0].upper() + word[1:]
                capitalize_next = False
                parts.append(word)
                word_count += 1

        if word_count == 0:
            return None

        text = " ".join(parts)
        if not text.endswith("."):
            text += "."
        return text


class NGramLanguageModel:
    """
    Store language model by n_grams, predict the next token.

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
        self._encoded_corpus = encoded_corpus
        self.__n_gram_size = n_gram_size
        self._n_gram_frequencies = {}

    def set_n_grams(self, frequencies: dict) -> None:
        """
        Retrieve value stored in self.__n_gram_size attribute.

        Returns:
            int: Size of stored n_grams
        """
        return self.__n_gram_size

    def set_n_grams(self, frequencies: dict) -> None:
        """
        Setter method for n-gram frequencies.

        Args:
            frequencies (dict): Computed in advance frequencies for n-grams
        """
        if isinstance(frequencies, dict):
            self._n_gram_frequencies = frequencies

    def build(self) -> bool:
        """
        Fill attribute `_n_gram_frequencies` from encoded corpus.

        Encoded corpus is stored in the attribute `_encoded_corpus`

        Returns:
            bool: True if attribute is filled successfully, otherwise False

        In case of corrupt input arguments or methods used return None,
        False is returned
        """
        if not isinstance(self._encoded_corpus, tuple) or not self._encoded_corpus:
            return False

        n_grams = self._extract_n_grams(self._encoded_corpus)
        if n_grams is None:
            return False

        n_gram_counts = {}
        context_counts = {}
        for n_gram in n_grams:
            n_gram_counts[n_gram] = n_gram_counts.get(n_gram, 0) + 1
            context = n_gram[:-1]
            context_counts[context] = context_counts.get(context, 0) + 1

        frequencies = {}
        for n_gram, count in n_gram_counts.items():
            context = n_gram[:-1]
            frequencies[n_gram] = count / context_counts[context]

        self._n_gram_frequencies = frequencies
        return True

    def generate_next_token(self, sequence: tuple[int, ...]) -> dict | None:
        """
        Retrieve tokens that can continue the given sequence along with their probabilities.

        Args:
            sequence (tuple[int, ...]): A sequence to match beginning of NGrams for continuation

        Returns:
            dict | None: Possible next tokens with their probabilities

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(sequence, tuple):
            return None

        context_length = self.__n_gram_size - 1
        if len(sequence) < context_length:
            return None

        context = sequence[-context_length:] if context_length else tuple()
        result = {}
        for n_gram, probability in self._n_gram_frequencies.items():
            if n_gram[:-1] == context:
                result[n_gram[-1]] = probability
        return result

    def _extract_n_grams(
        self, encoded_corpus: tuple[int, ...]
    ) -> tuple[tuple[int, ...], ...] | None:
        """
        Split encoded sequence into n-grams.

        Args:
            encoded_corpus (tuple[int, ...]): A tuple of encoded tokens

        Returns:
            tuple[tuple[int, ...], ...] | None: A tuple of extracted n-grams

        In case of corrupt input arguments, None is returned
        """
        if not isinstance(encoded_corpus, tuple):
            return None

        n_gram_size = self.__n_gram_size
        if len(encoded_corpus) < n_gram_size:
            return tuple()

        n_grams = []
        for i in range(len(encoded_corpus) - n_gram_size + 1):
            n_grams.append(encoded_corpus[i:i + n_gram_size])
        return tuple(n_grams)

import math


class GreedyTextGenerator:
    """
    Greedy text generation by N-grams.

    Attributes:
        _model (NGramLanguageModel): A language model to use for text generation
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(self, language_model: NGramLanguageModel, text_processor: TextProcessor) -> None:
        """
        Initialize an instance of GreedyTextGenerator.

        Args:
            language_model (NGramLanguageModel): A language model to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
        """
        self._model = language_model
        self._text_processor = text_processor

    def run(self, prompt: str, seq_len: int) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            prompt (str): Beginning of sequence
            seq_len (int): Number of tokens to generate

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if not isinstance(prompt, str) or not isinstance(seq_len, int) or seq_len <= 0:
            return None

        encoded = self._text_processor.encode(prompt)
        if encoded is None:
            return None

        sequence = encoded
        for _ in range(seq_len):
            next_tokens = self._model.generate_next_token(sequence)
            if not next_tokens:
                break

            # Choose token with maximum probability; tie -> larger token id.
            best_token = None
            best_prob = -1.0
            for token, prob in next_tokens.items():
                if prob > best_prob or (prob == best_prob and (best_token is None or token > best_token)):
                    best_token = token
                    best_prob = prob

            if best_token is None:
                break
            sequence = sequence + (best_token,)

        result = self._text_processor.decode(sequence)
        return result


class BeamSearcher:
    """
    Beam Search algorithm for diverse text generation.

    Attributes:
        _beam_width (int): Number of candidates to consider at each step
        _model (NGramLanguageModel): A language model to use for next token prediction
    """

    def __init__(self, beam_width: int, language_model: NGramLanguageModel) -> None:
        """
        Initialize an instance of BeamSearcher.

        Args:
            beam_width (int): Number of candidates to consider at each step
            language_model (NGramLanguageModel): A language model to use for next token prediction
        """
        self._beam_width = beam_width
        self._model = language_model

    def get_next_token(self, sequence: tuple[int, ...]) -> list[tuple[int, float]] | None:
        """
        Retrieve candidate tokens for sequence continuation.

        Candidates are sorted by probability and then by token in descending order.
        Number of tokens retrieved must not be bigger than beam width parameter.

        The return value has the following format: [(token, probability), ...].

        Args:
            sequence (tuple[int, ...]): Base sequence to continue

        Returns:
            list[tuple[int, float]] | None: Tokens to use for base sequence continuation

        In case of corrupt input arguments or methods used return None.
        """
        if not isinstance(sequence, tuple):
            return None

        next_tokens = self._model.generate_next_token(sequence)
        if next_tokens is None:
            return None
        if not next_tokens:
            return []

        candidates = list(next_tokens.items())
        # Sort by probability descending, then by token descending.
        candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
        return candidates[: self._beam_width]

    def continue_sequence(
        self,
        sequence: tuple[int, ...],
        next_tokens: list[tuple[int, float]],
        sequence_candidates: dict[tuple[int, ...], float],
    ) -> dict[tuple[int, ...], float] | None:
        """
        Generate new sequences from the base sequence with next tokens provided.

        The base sequence is deleted after continued variations are added.

        Args:
            sequence (tuple[int, ...]): Base sequence to continue
            next_tokens (list[tuple[int, float]]): Token for sequence continuation
            sequence_candidates (dict[tuple[int, ...], float]):
                Storage with all sequences generated

        Returns:
            dict[tuple[int, ...], float] | None: Updated sequence candidates

        In case of corrupt input arguments or unexpected behaviour of methods used return None.
        """
        if not isinstance(sequence, tuple):
            return None
        if not isinstance(next_tokens, list):
            return None
        if not isinstance(sequence_candidates, dict):
            return None
        if not next_tokens:
            return None
        if len(next_tokens) > self._beam_width:
            return None
        if sequence not in sequence_candidates:
            return None

        # Copy to avoid side effects.
        updated = dict(sequence_candidates)
        current_penalty = updated.pop(sequence)

        for token, probability in next_tokens:
            if not isinstance(token, int) or not isinstance(probability, float):
                return None
            if probability <= 0.0:
                return None
            new_sequence = sequence + (token,)
            updated[new_sequence] = current_penalty - math.log(probability)

        return updated

    def prune_sequence_candidates(
        self, sequence_candidates: dict[tuple[int, ...], float]
    ) -> dict[tuple[int, ...], float] | None:
        """
        Remove those sequence candidates that do not make top-N most probable sequences.

        Args:
            sequence_candidates (dict[tuple[int, ...], float]): Current candidate sequences

        Returns:
            dict[tuple[int, ...], float] | None: Pruned sequences

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_candidates, dict):
            return None

        items = list(sequence_candidates.items())

        # Stable sort: first by sequence descending, then by penalty ascending.
        items.sort(key=lambda item: item[0], reverse=True)
        items.sort(key=lambda item: item[1])

        pruned = items[: self._beam_width]
        return dict(pruned)


import json


class BeamSearchTextGenerator:
    """
    Class for text generation with BeamSearch.

    Attributes:
        _language_model (NGramLanguageModel): Language model for next token prediction
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
        _beam_width (int): Beam width parameter for generation
        beam_searcher (BeamSearcher): Searcher instance for the language model
    """

    def __init__(
        self, language_model: NGramLanguageModel, text_processor: TextProcessor, beam_width: int
    ) -> None:
        """
        Initializes an instance of BeamSearchTextGenerator.

        Args:
            language_model (NGramLanguageModel): Language model to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
            beam_width (int): Beam width parameter for generation
        """
        self._language_model = language_model
        self._text_processor = text_processor
        self._beam_width = beam_width
        self.beam_searcher = BeamSearcher(beam_width, language_model)

    def run(self, prompt: str, seq_len: int) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            prompt (str): Beginning of sequence
            seq_len (int): Number of tokens to generate

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if not isinstance(prompt, str) or not isinstance(seq_len, int) or seq_len <= 0:
            return None

        encoded = self._text_processor.encode(prompt)
        if encoded is None:
            return None

        sequence_candidates = {encoded: 0.0}

        for _ in range(seq_len):
            base_sequences = list(sequence_candidates.keys())
            for base_sequence in base_sequences:
                if base_sequence not in sequence_candidates:
                    continue

                next_tokens = self._get_next_token(base_sequence)
                if next_tokens is None:
                    return None
                if not next_tokens:
                    continue

                updated = self.beam_searcher.continue_sequence(
                    base_sequence, next_tokens, sequence_candidates
                )
                if updated is not None:
                    sequence_candidates = updated

            pruned = self.beam_searcher.prune_sequence_candidates(sequence_candidates)
            if pruned is None:
                return None
            sequence_candidates = pruned

        if not sequence_candidates:
            return None

        best_sequence = min(sequence_candidates.items(), key=lambda item: item[1])[0]
        return self._text_processor.decode(best_sequence)

    def _get_next_token(
        self, sequence_to_continue: tuple[int, ...]
    ) -> list[tuple[int, float]] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            list[tuple[int, float]] | None: Next tokens for sequence
            continuation

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_to_continue, tuple):
            return None
        return self.beam_searcher.get_next_token(sequence_to_continue)


class NGramLanguageModelReader:
    """
    Factory for loading language models ngrams from external JSON.

    Attributes:
        _json_path (str): Local path to assets file
        _eow_token (str): Special token for text processor
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(self, json_path: str, eow_token: str) -> None:
        """
        Initialize reader instance.

        Args:
            json_path (str): Local path to assets file
            eow_token (str): Special token for text processor
        """
        self._json_path = json_path
        self._eow_token = eow_token
        self._text_processor = TextProcessor(eow_token)
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                self._content = json.load(file)
        except (OSError, json.JSONDecodeError):
            self._content = None
        if self._content is not None:
            self._text_processor.fill_from_ngrams(self._content)

    def load(self, n_gram_size: int) -> NGramLanguageModel | None:
        """
        Fill attribute `_n_gram_frequencies` from dictionary with N-grams.

        The N-grams taken from dictionary must be cleaned from digits and punctuation,
        their length must match n_gram_size, and spaces must be replaced with EoW token.

        Args:
            n_gram_size (int): Size of ngram

        Returns:
            NGramLanguageModel | None: Built language model.

        In case of corrupt input arguments or unexpected behaviour of methods used,
        None is returned.
        """
        if not isinstance(n_gram_size, int) or isinstance(n_gram_size, bool):
            return None
        if n_gram_size < 2:
            return None
        if not isinstance(self._content, dict):
            return None

        freq = self._content.get("freq")
        if not isinstance(freq, dict):
            return None

        aggregated: dict[tuple[int, ...], float] = {}
        for ngram_str, count in freq.items():
            if not isinstance(ngram_str, str):
                continue
            if not isinstance(count, (int, float)):
                continue

            tokens = []
            for char in ngram_str:
                if char.isspace():
                    tokens.append(self._eow_token)
                elif char.isalpha():
                    tokens.append(char.lower())

            if len(tokens) != n_gram_size:
                continue

            encoded_tokens = []
            valid = True
            for token in tokens:
                token_id = self._text_processor.get_id(token)
                if token_id is None:
                    valid = False
                    break
                encoded_tokens.append(token_id)
            if not valid:
                continue

            encoded_ngram = tuple(encoded_tokens)
            aggregated[encoded_ngram] = aggregated.get(encoded_ngram, 0) + count

        context_counts: dict[tuple[int, ...], float] = {}
        for ngram, count in aggregated.items():
            context = ngram[:-1]
            context_counts[context] = context_counts.get(context, 0) + count

        frequencies: dict[tuple[int, ...], float] = {}
        for ngram, count in aggregated.items():
            context = ngram[:-1]
            frequencies[ngram] = count / context_counts[context]

        model = NGramLanguageModel(None, n_gram_size)
        model.set_n_grams(frequencies)
        return model

    def get_text_processor(self) -> TextProcessor: # type: ignore[empty-body]
        """
        Get method for the processor created for the current JSON file.

        Returns:
            TextProcessor: processor created for the current JSON file.
        """
        return self._text_processor


class BackOffGenerator:
    """
    Language model for back-off based text generation.

    Attributes:
        _language_models (dict[int, NGramLanguageModel]): Language models for next token prediction
        _text_processor (TextProcessor): A TextProcessor instance to handle text processing
    """

    def __init__(
        self, language_models: tuple[NGramLanguageModel, ...], text_processor: TextProcessor
    ) -> None:
        """
        Initializes an instance of BackOffGenerator.

        Args:
            language_models (tuple[NGramLanguageModel, ...]):
                Language models to use for text generation
            text_processor (TextProcessor): A TextProcessor instance to handle text processing
        """
        self._language_models = {}
        for model in language_models:
            self._language_models[model.get_n_gram_size()] = model
        self._text_processor = text_processor

    def run(self, prompt: str, seq_len: int) -> str | None:
        """
        Generate sequence based on NGram language model and prompt provided.

        Args:
            prompt (str): Beginning of sequence
            seq_len (int): Number of tokens to generate

        Returns:
            str | None: Generated sequence

        In case of corrupt input arguments or methods used return None,
        None is returned
        """
        if not isinstance(prompt, str) or not isinstance(seq_len, int) or seq_len <= 0:
            return None

        encoded = self._text_processor.encode(prompt)
        if encoded is None:
            return None

        sequence = encoded
        for _ in range(seq_len):
            next_tokens = self._get_next_token(sequence)
            if next_tokens is None:
                return None
            if not next_tokens:
                break

            best_token = None
            best_prob = -1.0
            for token, prob in next_tokens.items():
                if prob > best_prob or (
                    prob == best_prob and (best_token is None or token > best_token)
                ):
                    best_token = token
                    best_prob = prob

            if best_token is None:
                break
            sequence = sequence + (best_token,)

        return self._text_processor.decode(sequence)

    def _get_next_token(self, sequence_to_continue: tuple[int, ...]) -> dict[int, float] | None:
        """
        Retrieve next tokens for sequence continuation.

        Args:
            sequence_to_continue (tuple[int, ...]): Sequence to continue

        Returns:
            dict[int, float] | None: Next tokens for sequence continuation

        In case of corrupt input arguments return None.
        """
        if not isinstance(sequence_to_continue, tuple):
            return None

        for n_gram_size in sorted(self._language_models.keys(), reverse=True):
            model = self._language_models[n_gram_size]
            next_tokens = model.generate_next_token(sequence_to_continue)
            if next_tokens is not None and next_tokens:
                return next_tokens
        return None
