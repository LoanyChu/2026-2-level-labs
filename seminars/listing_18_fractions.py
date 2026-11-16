"""
Programming 2026.

Seminar 18.
Data Type: Fraction.
"""

# pylint: disable=invalid-name, unused-argument


# =============================================================================
# Fraction: implementing a custom rational number type
# =============================================================================

# A fraction represents a rational number as numerator / denominator.
#
# Our goal is to implement a custom Fraction class and make it behave like
# a built-in Python type.
#
# During this seminar, we will explore different Python protocols:
#
# - Object protocol: __init__, __str__, __repr__
# - Numeric protocol: __add__, __sub__, __mul__, __truediv__
# - Comparison protocol: __eq__, __lt__, __le__, ...
# - Conversion protocol: __int__, __float__, __bool__
# - Arithmetic protocol: __floordiv__, __mod__, __pow__

# Important properties:
# - Fractions represent rational numbers exactly.
# - The denominator cannot be zero.
# - Fractions should always be stored in their reduced form.
# - The denominator should always be positive.
# - Fraction objects should be immutable (optional advanced task).


# Original source:
# github.com/python/cpython/blob/main/Lib/fractions.py


from typing import Any


class Fraction:
    """
    Custom implementation of a rational number.
    """

    def __init__(self, numerator: int, denominator: int = 1) -> None:
        """
        Initialize a fraction.

        Examples:
            Fraction(1, 2)
            Fraction(3)
        """

    def __str__(self) -> str:
        """Return a human-readable representation."""

    def __repr__(self) -> str:
        """Return an unambiguous representation."""

    def __add__(self, other: Any) -> Any:
        """Return the sum of two fractions."""

    def __sub__(self, other: Any) -> Any:
        """Return the difference between two fractions."""

    def __mul__(self, other: Any) -> Any:
        """Return the product of two fractions."""

    def __truediv__(self, other: Any) -> Any:
        """Return the quotient of two fractions."""

    def __eq__(self, other: Any) -> Any:
        """Check whether two fractions are equal."""

    def __lt__(self, other: Any) -> Any:
        """Check whether this fraction is less than another: Any."""

    def __int__(self) -> int:
        """Convert the fraction to an integer."""

    def __float__(self) -> float:
        """Convert the fraction to a floating-point number."""

    def __bool__(self) -> bool:
        """Check whether the fraction is non-zero."""

    def __floordiv__(self, other: Any) -> Any:
        """Return the integer quotient."""

    def __mod__(self, other: Any) -> Any:
        """Return the remainder."""

    def __pow__(self, power: int) -> Any:
        """Raise the fraction to an integer power."""


# =============================================================================
# Testing our implementation
# =============================================================================

# Uncomment the following examples as the corresponding methods are implemented.

# --- Construction and representation ---

# f1 = Fraction(2, 4)
# f2 = Fraction(1, 2)
# f3 = Fraction(3)
#
# print(f1)       # 1/2
# print(repr(f1)) # Fraction(1, 2)
# print(f3)       # 3/1


# --- Arithmetic operations ---

# a = Fraction(1, 2)
# b = Fraction(1, 3)
#
# print(a + b)  # 5/6
# print(a - b)  # 1/6
# print(a * b)  # 1/6
# print(a / b)  # 3/2
# print(a // b) # 1
# print(a % b)  # 1/6
# print(a ** 2) # 1/4


# --- Comparisons ---

# a = Fraction(1, 2)
# b = Fraction(2, 4)
# c = Fraction(3, 4)
#
# print(a == b)  # True
# print(a < c)   # True
# print(a <= b)  # True (requires __le__ or suitable implementation)


# --- Conversions ---

# f = Fraction(3, 2)
#
# print(int(f))    # 1
# print(float(f))  # 1.5
# print(bool(f))   # True
# print(bool(Fraction(0)))  # False
