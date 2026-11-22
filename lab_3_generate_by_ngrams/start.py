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
    WordProcessor,
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

    print("\n===TEXTPROCESSOR(encode/decode) ===")
    print(f"Encoded corpus: {len(encoded_corpus)} tokens, "
          f"{len(text_processor._storage)} unique tokens in storage")
    print(f"Decoded text:\n{decoded_corpus[:200]}")

    n_gram_size = 2
    language_model = NGramLanguageModel(encoded_corpus, n_gram_size)
    build_status = language_model.build()
    assert build_status, "Building the language model failed"

    print("\nHow many unique n-grams:"
          f"{len(language_model._n_gram_frequencies)} unique n-grams")

    n_gram_size = 7
    greedy_model = NGramLanguageModel(encoded_corpus, n_gram_size)
    greedy_model.build()
    greedy_generator = GreedyTextGenerator(greedy_model, text_processor)
    greedy_result = greedy_generator.run("Vernon", 51)
    assert greedy_result, "Greedy generation failed"

    print("\n===GREEDY (n-grams = 7, 51 symbols, Vernon)===")
    print(f"Generated text:\n{greedy_result}")

    beam_width = 7
    beam_searcher_generator = BeamSearchTextGenerator(greedy_model, text_processor, beam_width)
    beam_search_result = beam_searcher_generator.run("Vernon", 56)
    assert beam_search_result, "Beam Search generation failed"

    print(f"\n===BEAMSEARCH (n-grams = 7, beam_width={beam_width}, 56 symbols, Vernon)===")
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

    print("\n===BACKOFF (100 symbols, Vernon)===")
    print(f"Generated text:\n{result}")

    #word_processor = WordProcessor("<eos>")
    #word_encoded = word_processor.encode(text)
    #assert word_encoded, "Word encoding failed"

    #word_models = []
    #for size in (2, 3, 4):
        #word_model = NGramLanguageModel(word_encoded, size)
        #assert word_model.build(), f"Building word-level model of size {size} failed"
        #word_models.append(word_model)

    #prompt = "Harry looked at"

    #greedy_word_result = GreedyTextGenerator(word_models[2], word_processor).run(prompt, 12)
    #assert greedy_word_result, "Word-level greedy generation failed"

    #beam_word_result = BeamSearchTextGenerator(word_models[2], word_processor, 3).run(prompt, 12)
    #assert beam_word_result, "Word-level beam search generation failed"

    #word_backoff = BackOffGenerator(tuple(word_models), word_processor)
    #backoff_word_result = word_backoff.run(prompt, 12)
    #assert backoff_word_result, "Word-level backoff generation failed"

    #greedy_letter_result = GreedyTextGenerator(greedy_model, text_processor).run(prompt, 60)
    #assert greedy_letter_result, "Letter-level greedy generation failed"

    #beam_letter_result = BeamSearchTextGenerator(greedy_model, text_processor, 3).run(prompt, 60)
    #assert beam_letter_result, "Letter-level beam search generation failed"

    #backoff_letter_result = backoff_generator.run(prompt, 60)
    #assert backoff_letter_result, "Letter-level backoff generation failed"

    #print("\n===COMPARISON WORD V/S LETTER (Harry looked at)===")
    #print(f"\n GREEDY (words):\n{greedy_word_result}")
    #print(f"\n GREEDY (letters):\n{greedy_letter_result}")
    #print(f"\n BEAMSEARCH (words):\n{beam_word_result}")
    #print(f"\n BEAMSEARCH (letters):\n{beam_letter_result}")
    #print(f"\n BACKOFF (words):\n{backoff_word_result}")
    #print(f"\n BACKOFF (letters):\n{backoff_letter_result}")

if __name__ == "__main__":
    main()
