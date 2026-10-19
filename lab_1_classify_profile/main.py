"""
Lab 1.

Language detection
"""

# pylint:disable=unused-argument
from typing import Sequence

FreqDictType = dict[str, float]
"Frequency dictionary. Contains pairs of token and its frequency."
ProfileType = tuple[str, FreqDictType, int]
"Language profile of a text. Contains language name, frequency dictionary and number of tokens."
# Mark 4.


def tokenize(text: str) -> Sequence[str] | None:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation and other symbols from words

    Args:
       text (str): Text

    Returns:
        Sequence[str] | None: Sequence of lower-cased tokens without punctuation.
        Returns None if input text is not a string.
    """
    if not isinstance(text, str):
        return None
#Если не строковый аргумент, вернуть Нан.
# Это что бы убедиться, что программа работает именно с текстом
    vowels = set('aeiouAEIOU')
    tokens = []
    for raw_token in text.split():
        token = ''.join(ch.lower() for ch in raw_token if ch.isnumeric() or ch.isalpha() and ch not in vowels)
        if token:
            tokens.append(token)

    return tokens
print(tokenize("7plorjfksaALLlloaAJKJSla"))
#СОздается пустой список. Потом текст разбивается на слова по пробелам.
# Текст приводится к нижнему регистру, оставляет только цифры и буквы,
# после клеит всё снова в одну строку. Не пустые токены добавляются в список.

def remove_stop_words(tokens: Sequence[str], stop_words: Sequence[str]) -> Sequence[str] | None:
    """
    Removes stop words

    Args:
        tokens (Sequence[str]): Sequence of tokens
        stop_words (Sequence[str]): Sequence of stop words (can be empty)
    Returns:
        Sequence[str] | None: Sequence of tokens without stop words.
        Returns None in case of incorrect input types.
    """
    if not isinstance(tokens, (list, tuple)) or not isinstance(stop_words, (list, tuple)):
        return None
    #Токены и стоп слова являются списком или кортежом.
    # Причина та же, что и у пред. проверки.
    for t in tokens:
        if not isinstance(t, str):
            return None
    #Проверка 2, на то что все токены строки.
    for s in stop_words:
        if not isinstance(s, str):
            return None
    #Проверка 3, тоже на строки

    stop_set = set(stop_words)
    #Множество стоп-слов для быстрой проверки

    result = []
    for token in tokens:
        if token not in stop_set:
            result.append(token)
    #Снова пустой список, и если токен не входит в множество стоп слов,
    # его добавляют в список

    return result


def calculate_frequencies(tokens: Sequence[str]) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens

    Args:
        tokens (Sequence[str]): Sequence of tokens
    Returns:
        dict[str, float] | None: Dictionary with frequencies.
        Returns None in case of incorrect input types.
    """
    if not isinstance(tokens, (list, tuple)):
        return None

    for t in tokens:
        if not isinstance(t, str):
            return None

    if len(tokens) == 0:
        return {}
    #Если токенов нет, вернет пустой словарь

    counts = {}
    for t in tokens:
        if t in counts:
            counts[t] += 1
        else:
            counts[t] = 1
#Пустой словарь для кол-ва вхождений.
#После подсчитывается, сколько раз встретился каждый токен.
    total = len(tokens)
    frequencies = {}
    for t, c in counts.items():
        frequencies[t] = c / total
#Создается словарь для частот, а после количество вхождений делится на количество токенов.
    return frequencies


def get_top_n_words(freq_dict: dict[str, float], top_n: int) -> Sequence[str] | None:
    """
    Finds the most common words

    Args:
        freq_dict (dict[str, float]): Dictionary with frequencies
        top_n (int): Number of the most common words

    Returns:
        Sequence[str] | None: Sequence of the most common words.
        Returns None in case of incorrect input types or non-positive top_n.
    """
    if not isinstance(freq_dict, dict):
        return None
    for k, v in freq_dict.items():
        if not isinstance(k, str):
            return None
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        return None

    items = list(freq_dict.items())
    items.sort(key=lambda pair: (-pair[1], pair[0]))

    result = []
    for i in range(min(top_n, len(items))):
        result.append(items[i][0])
#Словарь в список пар, сортировка по частоте - по убыванию, а если частоты одинаковы
# по алфавиту (возрастанию). Потом берет слова из сортировки и добавляет в результат.

    return result


# Mark 6.


def create_language_profile(
    language: str, text: str, stop_words: Sequence[str]
) -> ProfileType | None:
    """
    Creates a language profile

    Args:
        language (str): Language name
        text (str): Text
        stop_words (Sequence[str]): Sequence of stop words (can be empty)

    Returns:
        ProfileType | None: Language profile.
        Returns None in case of incorrect input types.
    """
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    if not isinstance(stop_words, (list, tuple)):
        return None

    for w in stop_words:
        if not isinstance(w, str):
            return None

    stop_set = {w.lower() for w in stop_words}
    tokens = text.lower().split()
    filtered_tokens = [t for t in tokens if t not in stop_set]
    #Создает множество стоп слов, подготавливает текст (всё в нижнем регистре).
    #Потом убирает стоп слова.

    frequencies = calculate_frequencies(filtered_tokens)
    #Считает частоту того, что осталось.
    if frequencies is None:
        return None

    return (language, frequencies)


def check_profile(profile: ProfileType) -> bool:
    """
    Checks profile structure

    Args:
        profile (ProfileType): Profile to check

    Returns:
        bool: Returns True if the profile has right structure and types,
        otherwise returns False.
    """
    if not isinstance(profile, tuple) or len(profile) != 2:
        return False

    language, frequencies = profile
    #Распаковка, профайл представляет из себя два элемента

    if not isinstance(language, str):
        return False

    if not isinstance(frequencies, dict):
        return False

    for k, v in frequencies.items():
        if not isinstance(k, str):
            return False
        if not isinstance(v, float):
            return False

    return True
#Если всё окей, то возвращает тру


def compare_profiles_by_top_n(
    unknown_profile: ProfileType, profile_to_compare: ProfileType, top_n: int
) -> float | None:
    """
    Compares profiles and calculates the distance using top n words

    Args:
        unknown_profile (ProfileType): Unknown profile
        profile_to_compare (ProfileType): Profile of a known language
        top_n (int): Number of the most common words
    Returns:
        float | None: The distance between profiles.
        Returns None in case of incorrect input types.
    """
    if not check_profile(unknown_profile) or not check_profile(profile_to_compare):
        return None
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        return None

    _, freq1 = unknown_profile
    _, freq2 = profile_to_compare

    top_words1 = get_top_n_words(freq1, top_n)
    top_words2 = get_top_n_words(freq2, top_n)

    if top_words1 is None or top_words2 is None:
        return None

    words = set(top_words1) | set(top_words2)

    distance = 0.0
    for w in words:
        distance += abs(freq1.get(w, 0.0) - freq2.get(w, 0.0))

    return distance


def detect_language_by_top_n(
    unknown_profile: ProfileType, profile_1: ProfileType, profile_2: ProfileType, top_n: int
) -> str | None:
    """
    Detects the language of an unknown profile

    Args:
        unknown_profile (ProfileType): Unknown profile
        profile_1 (ProfileType): Profile for comparison
        profile_2 (ProfileType): Another profile for comparison
        top_n (int): Number of the most common words

    Returns:
        str | None: Unknown profile language.
        Returns None in case of incorrect input types.
    """
    if not check_profile(unknown_profile):
        return None
    if not check_profile(profile_1) or not check_profile(profile_2):
        return None
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        return None

    distance_1 = compare_profiles_by_top_n(unknown_profile, profile_1, top_n)
    distance_2 = compare_profiles_by_top_n(unknown_profile, profile_2, top_n)

    if distance_1 is None or distance_2 is None:
        return None

    if distance_1 <= distance_2:
        return profile_1[0]
    else:
        return profile_2[0]


# Mark 8


def calculate_mse(predicted: Sequence[float], actual: Sequence[float]) -> float | None:
    """
    Calculates mean squared error between predicted and actual values.

    Args:
        predicted (Sequence[float]): Sequence of predicted values
        actual (Sequence[float]): Sequence of actual values

    Returns:
        float | None: The score
        Returns None in case of incorrect input types or mismatched length.
        In case of empty inputs, returns 0.0.
    """
    if not isinstance(predicted, (list, tuple)) or not isinstance(actual, (list, tuple)):
        return None

    for v in predicted:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
    for v in actual:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None

    if len(predicted) != len(actual):
        return None

    if len(predicted) == 0:
        return 0.0

    total = 0.0
    for p, a in zip(predicted, actual):
        total += (p - a) ** 2

    return total / len(predicted)


def compare_profiles_by_mse(
    unknown_profile: ProfileType, profile_to_compare: ProfileType
) -> float | None:
    """
    Compares two language profiles using the MSE metric.

    Args:
        unknown_profile (ProfileType): Unknown profile
        profile_to_compare (ProfileType): Profile
            to compare the unknown profile with

    Returns:
        float | None: The distance between the profiles.
        In case of corrupt input arguments or invalid profile structure, None is returned.
    """
    if not check_profile(unknown_profile):
        return None
    if not check_profile(profile_to_compare):
        return None

    _, freq1 = unknown_profile
    _, freq2 = profile_to_compare

    words = sorted(set(freq1.keys()) | set(freq2.keys()))

    predicted = [freq1.get(w, 0.0) for w in words]
    actual = [freq2.get(w, 0.0) for w in words]

    return calculate_mse(predicted, actual)


def detect_language_by_mse(
    unknown_profile: ProfileType, profile_1: ProfileType, profile_2: ProfileType
) -> str | None:
    """
    Detects the language of an unknown profile.

    Args:
        unknown_profile (ProfileType): Profile
            to determine the language of
        profile_1 (ProfileType): Known profile
        profile_2 (ProfileType): Another known profile

    Returns:
        str | None: Unknown profile language.
        Returns None in case of incorrect input types.
    """
    if not check_profile(unknown_profile):
        return None
    if not check_profile(profile_1) or not check_profile(profile_2):
        return None

    mse_1 = compare_profiles_by_mse(unknown_profile, profile_1)
    mse_2 = compare_profiles_by_mse(unknown_profile, profile_2)

    if mse_1 is None or mse_2 is None:
        return None

    if mse_1 <= mse_2:
        return profile_1[0]
    else:
        return profile_2[0]

# Mark 10


def save_profile(profile: ProfileType, save_path: str) -> bool:
    """
    Saves a language profile

    Args:
        profile (ProfileType): Profile
        save_path (str): Path to the folder to save profile

    Returns:
        bool: False in case of incorrect input types or if the profile
        is missing obligatory keys. True if the profile is saved.
    """


def load_profile(path_to_file: str) -> ProfileType | None:
    """
    Loads a language profile.

    Args:
        path_to_file (str): Path to the language profile

    Returns:
        ProfileType | None: Loaded profile.
        Returns None in case of incorrect input types.
    """


def collect_profiles(paths_to_profiles: Sequence[str]) -> Sequence[ProfileType] | None:
    """
    Collects profiles for a given path.

    Args:
        paths_to_profiles (Sequence[str]): Sequence of paths to the profiles

    Returns:
        Sequence[ProfileType] | None: Sequence of loaded profiles.
        Returns None in case of incorrect input types.
    """


def detect_language_advanced(
    unknown_profile: ProfileType, known_profiles: Sequence[ProfileType], top_n: int
) -> Sequence[tuple[str, dict[str, float]]] | None:
    """
    Detects the language of an unknown profile.

    Args:
        unknown_profile (ProfileType): Profile
            to determine the language of
        known_profiles (Sequence[ProfileType]): Known profiles
        top_n (int): Number of popular words

    Returns:
        Sequence[tuple[str, dict[str, float]]] | None: Sorted sequence of tuples
        containing a language and a distance via both metrics.
        The sequence is sorted by best MSE value, then by best Top-N value.
        Returns None in case of incorrect input types.
    """


def print_report(
    unknown_profile: ProfileType, metrics_stats: Sequence[tuple[str, dict[str, float]]], top_n: int
) -> None:
    """
    Prints report for detection of language.

    Args:
        unknown_profile (ProfileType): Profile
        metrics_stats (Sequence[tuple[str, dict[str, float]]]): Sequence with distances for
            available language comparison and metrics
        top_n (int): Number of popular words

    In case of incorrect type inputs, does not print anything.
    """
