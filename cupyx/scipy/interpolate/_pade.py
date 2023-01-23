import cupy as cp
from cupy import poly1d
import scipy.interpolate

__all__ = ["pade"]


def pade(an, m, n=None):
    """
    Return Pade approximation to a polynomial as the ratio of two polynomials.

    Parameters
    ----------
    an : (N,) cupy.ndarray
        Taylor series coefficients.
    m : int
        The order of the returned approximating polynomial `q`.
    n : int, optional
        The order of the returned approximating polynomial `p`. By default,
        the order is ``len(an)-1-m``.

    Returns
    -------
    p, q : Polynomial class
        The Pade approximation of the polynomial defined by `an` is
        ``p(x)/q(x)``.

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyx.scipy.interpolate import pade
    >>> e_exp = [1.0, 1.0, 1.0/2.0, 1.0/6.0, 1.0/24.0, 1.0/120.0]
    >>> p, q = pade(e_exp, 2)

    >>> e_exp.reverse()
    >>> e_poly = cp.poly1d(e_exp)

    Compare ``e_poly(x)`` and the Pade approximation ``p(x)/q(x)``

    >>> e_poly(1)
    2.71666667

    >>> p(1)/q(1)
    2.71794872

    """
    p_cpu, q_cpu = scipy.interpolate.pade(an.get(), m, n)
    return poly1d(cp.array(p_cpu.coeffs)), poly1d(cp.array(q_cpu.coeffs))
