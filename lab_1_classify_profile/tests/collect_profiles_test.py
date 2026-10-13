"""
Checks the first lab language profile collection function
"""

from pathlib import Path

import pytest

from admin_utils.constants import FLOAT_TOLERANCE
from lab_1_classify_profile.main import collect_profiles

PATH_TO_PROFILES_FOLDER = Path(__file__).parent / "assets"


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_collect_profiles_ideal() -> None:
    """
    Ideal scenario
    """

    expected = [
        (
            "de",
            {
                "hause": 0.1,
                "auslande": 0.1,
                "man": 0.6,
                "an": 0.4,
                "freunde": 0.1,
                "bin": 0.1,
                "gute": 0.1,
                "minuten": 0.1,
            },
            8,
        )
    ] * 3

    paths_to_profiles = [
        str(PATH_TO_PROFILES_FOLDER / "de.json"),
    ] * 3

    actual = collect_profiles(paths_to_profiles)
    for expected_profile, actual_profile in zip(expected, actual):
        assert expected_profile[0] == actual_profile[0]

        for token, frequency in expected_profile[1].items():
            assert actual_profile[1][token] == pytest.approx(frequency, abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_collect_profile_ideal_complex() -> None:
    """
    Ideal input scenario with corrupted and valid profiles
    """
    expected = [
        (
            "de",
            {
                "hause": 0.1,
                "auslande": 0.1,
                "man": 0.6,
                "an": 0.4,
                "freunde": 0.1,
                "bin": 0.1,
                "gute": 0.1,
                "minuten": 0.1,
            },
            8,
        )
    ]

    paths_to_profiles = [
        str(PATH_TO_PROFILES_FOLDER / "bad_profile_none.json"),
        str(PATH_TO_PROFILES_FOLDER / "bad_profile_missing_freq.json"),
        str(PATH_TO_PROFILES_FOLDER / "bad_profile_missing_name.json"),
        str(PATH_TO_PROFILES_FOLDER / "bad_profile_empty.json"),
        str(PATH_TO_PROFILES_FOLDER / "de.json"),
    ]

    actual = collect_profiles(paths_to_profiles)
    assert expected == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_collect_profile_bad_input_type() -> None:
    """
    Bad input scenario
    """
    expected = None
    bad_inputs = [{"a": 1}, None, 9, 9.34, True]
    for bad_input in bad_inputs:
        actual = collect_profiles(bad_input)
        assert expected == actual


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_collect_profile_bad_input_elements() -> None:
    """
    Bad input scenario, invalid element type in sequence
    """
    expected = None
    bad_inputs_wrong_elements = [
        [123],
        [None],
    ]
    for bad_input in bad_inputs_wrong_elements:
        actual = collect_profiles(bad_input)
        assert expected == actual
