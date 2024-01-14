"""Test cases for the util package."""

import pytest
import numpy as np
from itertools import product
from uncertainties.unumpy import nominal_values as unv
from uncertainties.unumpy import std_devs as usd
from uncertainties import unumpy as unp

from smplr.util import uncertain, unv_lambda, usd_lambda

_functions = [lambda x: np.zeros_like(x), lambda x: x, lambda x: x + 1, lambda x: x**2]
_arrays = [np.zeros(0), np.array(1), np.array([]), np.array([1, 2, 3, 4]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])]


@pytest.mark.parametrize(
    "fn,values",
    product(_functions + [np.sin, np.abs, unp.sin], _arrays),
)
def test_unv_lambda_no_unc(fn, values):
    """If a func without uncertainties is kept the same value."""
    expected = unv(fn(values))
    actual = unv_lambda(fn)(values)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "fn,values",
    product(_functions + [unp.sin], [unp.uarray(0 * a, 1) for a in _arrays]),
)
def test_unv_lambda_only_unc(fn, values):
    """If a func with uncertainties but no nominal values is set to zero."""
    expected = unv(fn(values))
    actual = unv_lambda(fn)(values)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "fn,values",
    product(_functions + [unp.sin], [unp.uarray(a, 1) for a in _arrays]),
)
def test_unv_lambda_mixed_uncertainties(fn, values):
    """If a func with unv and usd is stipped of the uncertainty."""
    expected = unv(fn(values))
    actual = unv_lambda(fn)(values)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "fn,values",
    product(_functions + [np.sin, np.abs, unp.sin], _arrays),
)
def test_usd_lambda_no_unc(fn, values):
    """If a func without uncertainties is kept the same value."""
    expected = usd(fn(values))
    actual = usd_lambda(fn)(values)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "fn,values",
    product(_functions + [unp.sin], [unp.uarray(0 * a, 1) for a in _arrays]),
)
def test_usd_lambda_only_unc(fn, values):
    """If a func with uncertainties but no nominal values is set to zero."""
    expected = usd(fn(values))
    actual = usd_lambda(fn)(values)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "fn,values",
    product(_functions + [unp.sin], [unp.uarray(a, 1) for a in _arrays]),
)
def test_usd_lambda_mixed_uncertainties(fn, values):
    """If a func with unv and usd is stipped of the uncertainty."""
    expected = usd(fn(values))
    actual = usd_lambda(fn)(values)
    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "data, mode, expected",
    [
        (np.array([1, 2, 3]), 'all', False),
        (np.array([1, 2, 3]), 'any', False),
        (unp.uarray([1, 2, 3], 0), 'all', False),
        (unp.uarray([1, 2, 3], 0), 'any', False),
        (unp.uarray([1, 2, 3], [0.1, 0, 0.3]), 'all', False),
        (unp.uarray([1, 2, 3], [0.1, 0, 0.3]), 'any', True),
        (unp.uarray([1, 2, 3], [0.1, 0.2, 0.3]), 'all', True),
        (unp.uarray([1, 2, 3], [0.1, 0.2, 0.3]), 'any', True),
    ],
)
def test_uncertain(data, mode, expected):
    """Test if ALL or ANY datapoints have uncertainties."""
    actual = uncertain(data, mode)
    assert actual == expected
