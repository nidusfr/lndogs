"""Helper functions for grain-size distribution analysis."""

from math import pi
from typing import Callable
import numpy as np
from scipy.special import erf


def differentiate(lst1, lst2):
    """Compute discrete differentiation of lst2 with respect to lst1.

    Args:
        lst1: Independent variable values
        lst2: Dependent variable values

    Returns:
        Tuple of (original lst1, differentiated lst2)
    """
    pairs1 = zip(lst1[:-1], lst1[1:])
    pairs2 = zip(lst2[:-1], lst2[1:])
    tmp1 = [(item[1] - item[0]) for item in pairs1]
    tmp2 = [(item[1] - item[0]) for item in pairs2]
    new2 = [item[1] / item[0] for item in zip(tmp1, tmp2)]
    tmp2 = [0] + new2
    tmp22 = [(a + b) / 2 for a, b in zip(tmp2, tmp2[1:] + [0])]
    return lst1, tmp22


def normalize(x: float, y: float, z: float) -> float:
    """Normalize three weights to percentage of their absolute sum.

    Args:
        x: First weight value
        y: Second weight value
        z: Third weight value

    Returns:
        Normalized percentage value of x
    """
    return 100 * abs(x) / (abs(x) + abs(y) + abs(z))


def phi(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian cumulative distribution function.

    Args:
        x: Input values
        mu: Mean parameter
        sigma: Standard deviation parameter

    Returns:
        Cumulative distribution values
    """
    return (1 + erf((x - mu) / abs(sigma) / np.sqrt(2))) / 2


def triple_phi(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Weighted sum of three Gaussian CDFs.

    Args:
        x: Input values
        params: Array of 9 parameters [μ1,σ1,w1, μ2,σ2,w2, μ3,σ3,w3]

    Returns:
        Combined weighted distribution values
    """
    return (normalize(params[2], params[5], params[8]) * phi(x, params[0], params[1]) +
            normalize(params[5], params[2], params[8]) * phi(x, params[3], params[4]) +
            normalize(params[8], params[2], params[5]) * phi(x, params[6], params[7]))


def gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Gaussian probability density function.

    Args:
        x: Input values
        mu: Mean parameter
        sigma: Standard deviation parameter

    Returns:
        Probability density values
    """
    return np.exp((-1 * (x - mu)**2) / (2 * abs(sigma)**2)) / np.sqrt(2 * pi * abs(sigma)**2)


def triple_gauss(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Weighted sum of three Gaussian PDFs.

    Args:
        x: Input values
        params: Array of 9 parameters [μ1,σ1,w1, μ2,σ2,w2, μ3,σ3,w3]

    Returns:
        Combined weighted density values
    """
    return (normalize(params[2], params[5], params[8]) * gauss(x, params[0], params[1]) +
            normalize(params[5], params[2], params[8]) * gauss(x, params[3], params[4]) +
            normalize(params[8], params[2], params[5]) * gauss(x, params[6], params[7]))


def sup(a: float) -> Callable[[float], float]:
    """Create lower bound constraint function.

    Args:
        a: Lower bound value

    Returns:
        Constraint function that returns x - a
    """
    def foo(x: float) -> float:
        return x - a
    return foo


def inf(a: float) -> Callable[[float], float]:
    """Create upper bound constraint function.

    Args:
        a: Upper bound value

    Returns:
        Constraint function that returns a - x
    """
    def foo(x: float) -> float:
        return a - x
    return foo


def supinf(a: float, b: float) -> Callable[[float], float]:
    """Create range constraint function.

    Args:
        a: Lower bound
        b: Upper bound

    Returns:
        Constraint function that returns (x-a)(b-x)
    """
    def foo(x: float) -> float:
        return (x - a) * (b - x)
    return foo


def c(x: float, func: Callable[[float], float]) -> float:
    """Apply constraint penalty when violated.

    Args:
        x: Value to check
        func: Constraint function

    Returns:
        0 if constraint satisfied, large penalty if violated
    """
    return (1 - np.sign(max(0, func(x)))) * 1000000


def compute_errors(p: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute error between model and data with parameter constraints.

    Args:
        p: Current parameters [μ1,σ1,w1, μ2,σ2,w2, μ3,σ3,w3]
        y: Target values
        x: Input values

    Returns:
        Error values with constraint penalties
    """
    err = ((y - triple_phi(x, p)) +
           c(p[0], supinf(-6, 6)) +
           c(p[3], supinf(-6, 6)) +
           c(p[6], supinf(-6, 6)))
    return err
