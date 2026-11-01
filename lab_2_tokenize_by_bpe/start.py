"""
BPE Tokenizer starter
"""
from lab_2_tokenize_by_bpe.main import (
    calculate_bleu,
    decode)
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

    encoded = [int(token) for token in translation_encoded_raw.split()]

    decoded = decode(encoded, vocabulary, None)
    assert decoded is not None, "Decoding failed"

    cleaned_translation = decoded.replace("\u2581", " ").strip()

    result = calculate_bleu(cleaned_translation, text_reference_translation.strip())
    assert result is not None, "BLEU computation failed"

    print(f"BLEU: {result}")


if __name__ == "__main__":
    main()
