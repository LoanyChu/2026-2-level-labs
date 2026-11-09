"""
BPE Tokenizer starter
"""

import json

from lab_2_tokenize_by_bpe.main import calculate_bleu, decode

# pylint:disable=too-many-locals, unused-variable


def main() -> None:
    """
    Launches an implementation
    """
    with open("lab_2_tokenize_by_bpe/assets/text.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_raw.txt", "r", encoding="utf-8") as text_file:
        text_reference_translation = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/en_encoded.txt", "r", encoding="utf-8") as text_file:
        translation_encoded_raw = text_file.read()
    with open("lab_2_tokenize_by_bpe/assets/vocab.json", "r", encoding="utf-8") as vocab_file:
        vocabulary = json.load(vocab_file)

    print("-----VOCABULARY-----")
    print (dict(list(vocabulary.items())))

    encoded = [int(token) for token in translation_encoded_raw.split()]

    print("-----ENCODED TRANSLATION-----")
    print(encoded)

    decoded = decode(encoded, vocabulary, None)
    assert decoded is not None, "Decoding failed"
    print ("-----DECODED TRANSLATION-----")
    print (decoded)

    cleaned_translation = decoded.replace("\u2581", " ").strip()
    print("-----CLEANED TRANSLATION-----")
    print(cleaned_translation)

    result = calculate_bleu(cleaned_translation, text_reference_translation.strip())
    assert result is not None, "BLEU computation failed"

    print(f"BLEU is this: {result}")


if __name__ == "__main__":
    main()
