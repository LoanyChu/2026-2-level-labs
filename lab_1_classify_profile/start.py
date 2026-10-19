"""
Language detection starter.
"""
# pylint: disable=unused-variable, duplicate-code
from main import (
    create_language_profile,
    detect_language_by_mse,
    detect_language_by_top_n,
)


#Mark 8
def main() -> None:
    """
    Launches an implementation.
    """
    with open("lab_1_classify_profile/assets/texts/de.txt", "r", encoding="utf-8") as file:
        de_text = file.read()
    with open("lab_1_classify_profile/assets/texts/unknown.txt", "r", encoding="utf-8") as file:
        unknown_text = file.read()
    with open("lab_1_classify_profile/assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    with open("lab_1_classify_profile/assets/texts/en.txt", "r", encoding="utf-8") as file:
        en_text = file.read()
    de_profile = create_language_profile("de", de_text, stopwords)
    en_profile = create_language_profile("en", en_text, stopwords)
    unknown_profile = create_language_profile("unknown", unknown_text, stopwords)

    result = detect_language_by_mse(unknown_profile, de_profile, en_profile)

    print("Определённый язык:", result)
    assert result, "Detection result is None"



if __name__ == "__main__":
    main()
