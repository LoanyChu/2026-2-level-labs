"""
Checks the first lab profile loading function
"""

import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from admin_utils.constants import FLOAT_TOLERANCE
from lab_1_classify_profile.main import load_profile

PATH_TO_PROFILES_FOLDER = Path(__file__).parent / "assets"


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_load_profile_ideal() -> None:
    """
    Ideal scenario
    """
    path_to_profile = PATH_TO_PROFILES_FOLDER / "de.json"

    with open(path_to_profile, "r", encoding="utf-8") as file:
        expected = json.load(file)

    actual = load_profile(str(path_to_profile))
    assert actual[0] == expected["name"]
    assert isinstance(actual[1], dict)
    assert len(actual[1]) == len(expected["freq"])
    assert actual[2] == expected["n_words"]
    for token, value in expected["freq"].items():
        assert value == pytest.approx(actual[1][token], abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_load_profile_invalid_profile() -> None:
    """
    Bad input scenario, invalid profile
    """
    path_to_profile = PATH_TO_PROFILES_FOLDER / "bad_profile_none.json"
    actual = load_profile(str(path_to_profile))
    assert actual is None

    path_to_profile = PATH_TO_PROFILES_FOLDER / "bad_profile_empty.json"
    actual = load_profile(str(path_to_profile))
    assert actual is None

    path_to_profile = PATH_TO_PROFILES_FOLDER / "bad_profile_extra_keys.json"
    actual = load_profile(str(path_to_profile))
    assert actual is None

    path_to_profile = PATH_TO_PROFILES_FOLDER / "bad_profile_missing_freq.json"
    actual = load_profile(str(path_to_profile))
    assert actual is None

    path_to_profile = PATH_TO_PROFILES_FOLDER / "bad_profile_wrong_freq.json"
    actual = load_profile(str(path_to_profile))
    assert actual is None

    path_to_profile = PATH_TO_PROFILES_FOLDER / "bad_profile_missing_name.json"
    actual = load_profile(str(path_to_profile))
    assert actual is None

    path_to_profile = PATH_TO_PROFILES_FOLDER / "bad_profile_wrong_name.json"
    actual = load_profile(str(path_to_profile))
    assert actual is None


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_load_profile_bad_input_path_type() -> None:
    """
    Bad input scenario, wrong path type
    """
    actual = load_profile(None)
    assert actual is None


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_load_profile_invalid_check_profile(monkeypatch: MonkeyPatch) -> None:
    """
    Bad function output scenario, invalid check_profile

    Args:
        monkeypatch (MonkeyPatch): Pytest monkeypatch fixture to mock check_profile
    """
    monkeypatch.setattr("lab_1_classify_profile.main.check_profile", lambda *args, **kwargs: False)

    path_to_profile = PATH_TO_PROFILES_FOLDER / "de.json"
    actual = load_profile(str(path_to_profile))
    assert actual is None
