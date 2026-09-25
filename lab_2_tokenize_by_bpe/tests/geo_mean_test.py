"""
Checks the second lab's geo mean function
"""

# pylint: disable=assignment-from-no-return
import pytest

from lab_2_tokenize_by_bpe.main import calculate_geo_mean


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_geo_mean_ideal() -> None:
    """
    Ideal geo mean scenario
    """
    expected = 0.6878257000127244
    actual = calculate_geo_mean([0.8181818181818182, 0.6363636363636364, 0.625], 3)
    assert expected == pytest.approx(actual, abs=0.001)


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_geo_mean_negative_value() -> None:
    """
    Geo mean negative precisions check
    """
    expected = 0.0
    actual = calculate_geo_mean([-1, 0, 0.5], 3)
    assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_geo_mean_bad_input() -> None:
    """
    Geo mean invalid inputs check
    """
    precisions_bad_inputs = [(), "string", {}, None, 1, 1.1, True]
    max_order_bad_inputs = [None, (), 1.1, [None], "string", {}]
    expected = None
    for precision_bad_input in precisions_bad_inputs:
        actual = calculate_geo_mean(precision_bad_input, 3)
        assert expected == actual
    for max_order_bad_input in max_order_bad_inputs:
        actual = calculate_geo_mean([-1, 0, 0.5], max_order_bad_input)
        assert expected == actual


@pytest.mark.lab_2_tokenize_by_bpe
@pytest.mark.mark10
def test_geo_mean_return_value() -> None:
    """
    Geo mean return value check
    """
    actual = calculate_geo_mean([0.8181818181818182, 0.6363636363636364, 0.625], 3)
    assert isinstance(actual, float)
