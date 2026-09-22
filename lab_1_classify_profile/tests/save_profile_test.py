"""
Checks the first profile saving function
"""

# pylint: disable=duplicate-code,assignment-from-no-return

import json
import shutil
from pathlib import Path
from typing import Generator

import pytest

from admin_utils.constants import FLOAT_TOLERANCE, PROJECT_ROOT
from lab_1_classify_profile.main import save_profile

TEST_DIRECTORY = PROJECT_ROOT / "pathstokeep"
JSON_PATH_TO_COMPARE = Path(__file__).parent / "assets" / "de.json"


@pytest.fixture(scope="function", autouse=True)
def pathlib_setup() -> Generator[None, None, None]:
    """
    Setting up
    """
    if TEST_DIRECTORY.exists():
        shutil.rmtree(TEST_DIRECTORY)
    TEST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    yield

    if TEST_DIRECTORY.exists():
        shutil.rmtree(TEST_DIRECTORY)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_save_profile_ideal() -> None:
    """
    Ideal scenario
    """
    profile = (
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
    actual = save_profile(profile, str(TEST_DIRECTORY))
    assert actual is True

    with open(f"{TEST_DIRECTORY}/de.json", "r", encoding="utf-8") as f:
        actual = json.load(f)

    with open(JSON_PATH_TO_COMPARE, "r", encoding="utf-8") as f:
        expected = json.load(f)

    assert expected["name"] == actual["name"]
    assert len(expected["freq"]) == len(actual["freq"])
    assert expected["n_words"] == actual["n_words"]
    for token, freq in expected["freq"].items():
        assert actual["freq"][token] == pytest.approx(freq, abs=FLOAT_TOLERANCE)


@pytest.mark.lab_1_classify_profile
@pytest.mark.mark10
def test_save_profile_bad_input_type() -> None:
    """
    Bad input scenario
    """
    bad_profiles = ["goodbye", (), None, 9, 9.34, True, [None], []]
    for bad_profile in bad_profiles:
        actual = save_profile(bad_profile, str(TEST_DIRECTORY))
        assert actual is False

    bad_paths = [(), None, 9, 9.34, True, [None], [], {}]
    for bad_path in bad_paths:
        actual = save_profile(("profile", {"a": 0.1}, 1), bad_path)
        assert actual is False
