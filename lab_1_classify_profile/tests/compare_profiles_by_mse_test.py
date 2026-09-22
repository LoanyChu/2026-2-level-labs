"""
Checks the first lab language comparison function
"""

# pylint: disable=duplicate-code,disable=assignment-from-no-return

import pytest

from admin_utils.constants import FLOAT_TOLERANCE
from lab_1_classify_profile.main import compare_profiles_by_mse


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_mse_ideal() -> None:
    """
    Ideal scenario
    """
    unknown_profile = ("en", {"happy": 0.3333, "he": 0.1667, "man": 0.1667}, 3)

    profile_to_compare = (
        "de",
        {
            "ich": 0.3333,
            "weiß": 0.1111,
            "nicht": 0.1111,
            "man": 0.3111,
        },
        4,
    )

    actual = compare_profiles_by_mse(unknown_profile, profile_to_compare)
    assert 0.0493 == pytest.approx(actual, abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_mse_no_intersections_ideal() -> None:
    """
    Ideal scenario with no intersections
    """
    unknown_profile = ("en", {"happy": 0.3333, "he": 0.1667, "man": 0.1667}, 3)

    profile_to_compare = (
        "de",
        {
            "ich": 0.3333,
            "weiß": 0.1111,
            "nicht": 0.1111,
        },
        3,
    )

    actual = compare_profiles_by_mse(unknown_profile, profile_to_compare)
    assert 0.0504 == pytest.approx(actual, abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_mse_identical() -> None:
    """
    Ideal scenario with identical profiles
    """
    unknown_profile = ("unknown", {"happy": 0.3333, "he": 0.1667, "man": 0.1667}, 3)
    profile_to_compare = ("unknown", {"happy": 0.3333, "he": 0.1667, "man": 0.1667}, 3)

    actual = compare_profiles_by_mse(unknown_profile, profile_to_compare)
    assert 0.0 == pytest.approx(actual, abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark8
@pytest.mark.mark10
def test_compare_profiles_by_mse_bad_input_type() -> None:
    """
    Bad input scenario
    """
    bad_profiles = [{}, (), [], 3.14, -9, "this is a profile, trust"]

    profile_to_compare = (
        "de",
        {
            "ich": 0.3333,
            "weiß": 0.1111,
            "nicht": 0.1111,
            "man": 0.3111,
        },
        4,
    )

    for bad_profile in bad_profiles:
        assert compare_profiles_by_mse(bad_profile, profile_to_compare) is None
        assert compare_profiles_by_mse(profile_to_compare, bad_profile) is None
