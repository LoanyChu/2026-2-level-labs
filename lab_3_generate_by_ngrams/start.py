"""
Generation by NGrams starter
"""

# pylint:disable=too-many-locals, unused-import, unused-variable, protected-access

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from lab_3_generate_by_ngrams.main import (
    BackOffGenerator,
    BeamSearchTextGenerator,
    GreedyTextGenerator,
    NGramLanguageModel,
    NGramLanguageModelReader,
    TextProcessor,
)


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    lab_path = os.path.dirname(os.path.abspath(__file__))

    with open(
        os.path.join(lab_path, "assets", "Harry_Potter.txt"), "r", encoding="utf-8"
    ) as text_file:
        text = text_file.read()

    text_processor = TextProcessor("_")
    encoded_corpus = text_processor.encode(text)
    assert encoded_corpus, "Encoding failed"
    decoded_corpus = text_processor.decode(encoded_corpus)
    assert decoded_corpus, "Decoding failed"

    print("===TEXTPROCESSOR(encode/decode) ===")
    print(f"Encoded corpus: {len(encoded_corpus)} tokens, "
          f"{len(text_processor._storage)} unique tokens in storage")
    print(f"Decoded text:\n{decoded_corpus}")

    n_gram_size = 2
    language_model = NGramLanguageModel(encoded_corpus, n_gram_size)
    build_status = language_model.build()
    assert build_status, "Building the language model failed"

    print("How many unique n-grams:"
          f"{len(language_model._n_gram_frequencies)} unique n-grams")

    n_gram_size = 7
    greedy_model = NGramLanguageModel(encoded_corpus, n_gram_size)
    greedy_model.build()
    greedy_generator = GreedyTextGenerator(greedy_model, text_processor)
    greedy_result = greedy_generator.run("Vernon", 51)
    assert greedy_result, "Greedy generation failed"

    print("===GREEDYTEXT (n-grams = 7, 51 symbols, Vernon)===")
    print(f"Generated text:\n{greedy_result}")

    beam_width = 7
    beam_searcher_generator = BeamSearchTextGenerator(greedy_model, text_processor, beam_width)
    beam_search_result = beam_searcher_generator.run("Vernon", 56)
    assert beam_search_result, "Beam Search generation failed"

    print(f"===BEAMSEARCH (n-grams = 7, beam_width={beam_width}, 56 symbols, Vernon)===")
    print(f"Generated text:\n{beam_search_result}")

    reader = NGramLanguageModelReader(
        os.path.join(lab_path, "assets", "en_own.json"), "_"
    )
    language_models = []
    for size in range(2, 5):
        model = reader.load(size)
        assert model, f"Loading the language model of size {size} failed"
        language_models.append(model)

    backoff_generator = BackOffGenerator(tuple(language_models), reader.get_text_processor())
    generated_text = backoff_generator.run("Vernon", 100)
    assert generated_text, "BackOff generation failed"

    result = generated_text
    assert result

    print("===BACKOFF (100 symbols, Vernon)===")
    print(f"Generated text:\n{result}")



if __name__ == "__main__":
    main()
