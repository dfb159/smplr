"""Generic utils and wrappers."""
from functools import wraps
from uncertainties.unumpy import nominal_values as unv
from uncertainties.unumpy import std_devs as usd



def unv_lambda(fn):
    """Return only the nominal values of ``fn``."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return unv(fn(*args, **kwargs))

    return wrapper


def usd_lambda(fn):
    """Return only the standard deviations of ``fn``."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return usd(fn(*args, **kwargs))

    return wrapper
