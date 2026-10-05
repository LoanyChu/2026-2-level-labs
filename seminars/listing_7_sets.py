"""
Programming 2026.

Seminar 7.

Data Type: set. frozenset.
"""

# pylint: disable=unused-argument,invalid-name,duplicate-value

# Common information about sets
#
# - A set is a collection of unique elements
# - Sets are unordered: elements have no index, so indexing and slicing are not supported
# - Sets are mutable (can be changed in-place)
# - Elements of a set must be immutable (hashable): int, str, tuple, etc.
# - Sets are useful for removing duplicates, testing membership
#   and performing mathematical operations

# Create a set
example_set = {1, 2, 3, 4, 4, 2}
print("Example set (duplicates removed):", example_set)
print("*" * 30)

# Create a set (second way): from any iterable
letters = set("hello")
print(letters)
print("*" * 30)

# Empty set: {} creates an empty dict, not a set!
empty_set = set()
print(type(empty_set), type({}))
print("*" * 30)

# Membership test is fast for sets
print(3 in example_set)
print(10 not in example_set)
print("*" * 30)

# Add and remove elements
example_set.add(5)
print(example_set)
example_set.remove(5)  # raises KeyError if the element is absent
example_set.discard(100)  # does nothing if the element is absent
print(example_set)
print("*" * 30)

# Remove duplicates from a list
duplicated = [1, 2, 2, 3, 3, 3]
print(list(set(duplicated)))
print("*" * 30)

# Basic operations
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print("Union:", a | b)  # {1, 2, 3, 4, 5, 6}
print("Intersection:", a & b)  # {3, 4}
print("Difference:", a - b)  # {1, 2}
print("Symmetric diff:", a ^ b)  # {1, 2, 5, 6}
print("*" * 30)

# Comparison of sets
print({1, 2} <= {1, 2, 3})  # subset
print({1, 2, 3} >= {1, 2})  # superset
print({1, 2} == {2, 1})  # order does not matter
print("*" * 30)

# Set methods (some of them)
# .add(item) -> add the item to the set
# .remove(item) -> remove the item, raise KeyError if it is absent
# .discard(item) -> remove the item if it is present
# .pop() -> remove and return an arbitrary item
# .clear() -> remove all items
# .union(other), .intersection(other), .difference(other),
# .symmetric_difference(other) -> same as |, &, -, ^ but accept any iterable
# .update(other) -> add all items from other to the set
# .issubset(other), .issuperset(other), .isdisjoint(other) -> comparisons

# Frozenset
#
# - frozenset is an immutable version of set
# - it can be used as a dict key or as an element of another set
frozen = frozenset([1, 2, 3])
print(frozen)
set_of_sets = {frozenset({1, 2}), frozenset({3, 4})}
print(set_of_sets)

print("*" * 15 + " TASKS " + "*" * 15)


# TASKS


# Task 1:
# easy level
def common_elements(original_list: list[int], secondary_list: list[int]) -> set[int]:
    """
    Find common elements of two lists.

    Args:
        original_list (list[int]): First list of numbers.
        secondary_list (list[int]): Second list of numbers.

    Returns:
        set[int]: Elements that are present in both lists.
    """
    # student realisation goes here


# Function calls with expected result:
# assert common_elements([1, 2, 3, 4], [3, 4, 5]) == {3, 4}
# assert common_elements([10, 20], [30, 40]) == set()


# Task 2:
# easy level
def unique_letters(word: str) -> set[str]:
    """
    Find all unique letters in a word (case-insensitive).

    Args:
        word (str): Input word.

    Returns:
        set[str]: Unique lowercase letters.
    """
    # student realisation goes here


# Function calls with expected result:
# assert unique_letters("Banana") == {"b", "a", "n"}
# assert unique_letters("Hello") == {"h", "e", "l", "o"}


# Task 3:
# easy level
def are_disjoint(original_set: set[int], secondary_set: set[int]) -> bool:
    """
    Check whether two sets are disjoint (no common elements).

    Args:
        original_set (set[int]): First set.
        secondary_set (set[int]): Second set.

    Returns:
        bool: True if sets have no elements in common, False otherwise.
    """
    # student realisation goes here


# Function calls with expected result:
# assert are_disjoint({1, 2, 3}, {4, 5, 6}) is True
# assert are_disjoint({1, 2, 3}, {3, 4, 5}) is False


# Task 4:
# easy level
def count_unique(numbers: list[int]) -> int:
    """
    Count the number of distinct elements in the list.

    Args:
        numbers (list[int]): List of numbers.

    Returns:
        int: Number of distinct elements.
    """
    # student realisation goes here


# Function calls with expected result:
# assert count_unique([1, 2, 2, 3, 3, 3]) == 3
# assert count_unique([]) == 0


# Task 5:
# easy level
def is_pangram(sentence: str) -> bool:
    """
    Check whether the sentence contains every letter of the English alphabet
    at least once (case-insensitive).

    Args:
        sentence (str): Input sentence.

    Returns:
        bool: True if the sentence is a pangram, False otherwise.
    """
    # student realisation goes here


# Function calls with expected result:
# assert is_pangram("The quick brown fox jumps over the lazy dog") is True
# assert is_pangram("Hello world") is False


# Task 6:
# medium level
def remove_duplicates_keep_order(items: list[int]) -> list[int]:
    """
    Remove duplicates from the list, keeping the order of the first occurrences.

    Args:
        items (list[int]): List of numbers.

    Returns:
        list[int]: List without duplicates.
    """
    # student realisation goes here


# Function calls with expected result:
# assert remove_duplicates_keep_order([3, 1, 3, 2, 1]) == [3, 1, 2]
# assert remove_duplicates_keep_order([5, 5, 5]) == [5]


# Task 7:
# medium level
def only_in_one(first: set[str], second: set[str]) -> set[str]:
    """
    Find the elements that are present in exactly one of the two sets.

    Args:
        first (set[str]): First set.
        second (set[str]): Second set.

    Returns:
        set[str]: Elements present only in one of the sets.
    """
    # student realisation goes here


# Function calls with expected result:
# assert only_in_one({"a", "b", "c"}, {"b", "c", "d"}) == {"a", "d"}
# assert only_in_one({"x"}, {"x"}) == set()


# Task 8:
# medium level
def find_missing_numbers(numbers: list[int], n: int) -> list[int]:
    """
    Find all numbers from 1 to n (inclusive) that are missing in the list.

    Args:
        numbers (list[int]): List of numbers.
        n (int): Upper bound of the range.

    Returns:
        list[int]: Sorted list of missing numbers.
    """
    # student realisation goes here


# Function calls with expected result:
# assert find_missing_numbers([1, 2, 4, 6], 6) == [3, 5]
# assert find_missing_numbers([1, 2, 3], 3) == []


# Task 9:
# medium level
def common_letters(words: list[str]) -> set[str]:
    """
    Find the letters that are present in every word of the list.

    Args:
        words (list[str]): List of lowercase words.

    Returns:
        set[str]: Letters common to all words. Empty set for an empty list.
    """
    # student realisation goes here


# Function calls with expected result:
# assert common_letters(["apple", "grape", "pear"]) == {"a", "p", "e"}
# assert common_letters(["cat", "dog"]) == set()


# Task 10:
# medium level
def has_pair_with_sum(numbers: list[int], target: int) -> bool:
    """
    Check whether there are two different elements in the list whose sum equals target.

    Args:
        numbers (list[int]): List of numbers.
        target (int): Target sum.

    Returns:
        bool: True if such pair exists, False otherwise.
    """
    # student realisation goes here


# Function calls with expected result:
# assert has_pair_with_sum([1, 4, 6, 9], 10) is True
# assert has_pair_with_sum([1, 2, 3], 7) is False


# Task 11:
# hard level
def group_anagrams(words: list[str]) -> set[frozenset[str]]:
    """
    Group words that are anagrams of each other.

    Args:
        words (list[str]): List of lowercase words.

    Returns:
        set[frozenset[str]]: Set of groups, each group is a frozenset of anagrams.
    """
    # student realisation goes here


# Function calls with expected result:
# assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == {
#     frozenset({"eat", "tea", "ate"}),
#     frozenset({"tan", "nat"}),
#     frozenset({"bat"}),
# }
