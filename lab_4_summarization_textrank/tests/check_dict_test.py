"""
Checks the fourth lab check_dict helper
"""

import pytest

from lab_4_summarization_textrank.main import check_dict


@pytest.mark.lab_4_summarization_textrank
@pytest.mark.mark10
def test_check_dict() -> None:
    """
    Checks dictionary validation for valid and invalid inputs
    """
    assert check_dict({"a": 1.0}, str, float, False)
    assert check_dict({}, str, float, True)
    assert not check_dict({}, str, float, False)
    assert not check_dict(["a"], str, float, True)
    assert not check_dict({1: 1.0}, str, float, True)
    assert not check_dict({"a": "1"}, str, float, True)
