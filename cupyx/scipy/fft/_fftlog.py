'''Fast Hankel transforms using the FFTLog algorithm.
The implementation closely follows the Fortran code of Hamilton (2000).
'''

import math
from warnings import warn

import cupy
from cupyx.scipy.fft import _fft
from cupyx.scipy.special import loggamma, poch

try:
    # fht only exists in SciPy >= 1.7
    from scipy.fft import fht as _fht
    _scipy_fft = _fft._scipy_fft
    del _fht
except ImportError:
    class _DummyModule:
        def __getattr__(self, name):
            return None

    _scipy_fft = _DummyModule()

# Note scipy also defines fhtoffset but this only operates on scalars
__all__ = ['fht', 'ifht']


# constants
LN_2 = math.log(2)


@_fft._implements(_scipy_fft.fht)
def fht(a, dln, mu, offset=0.0, bias=0.0):
    """Compute the fast Hankel transform.

    Computes the discrete Hankel transform of a logarithmically spaced periodic
    sequence using the FFTLog algorithm [1]_, [2]_.

    Parameters
    ----------
    a : cupy.ndarray (..., n)
        Real periodic input array, uniformly logarithmically spaced.  For
        multidimensional input, the transform is performed over the last axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : float
        Order of the Hankel transform, any positive or negative real number.
    offset : float, optional
        Offset of the uniform logarithmic spacing of the output array.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.

    Returns
    -------
    A : cupy.ndarray (..., n)
        The transformed output array, which is real, periodic, uniformly
        logarithmically spaced, and of the same shape as the input array.

    See Also
    --------
    :func:`scipy.special.fht`
    :func:`scipy.special.fhtoffset` : Return an optimal offset for `fht`.

    References
    ----------
    .. [1] Talman J. D., 1978, J. Comp. Phys., 29, 35
    .. [2] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)

    """

    # size of transform
    n = a.shape[-1]

    # bias input array
    if bias != 0:
        # a_q(r) = a(r) (r/r_c)^{-q}
        j_c = (n-1)/2
        j = cupy.arange(n)
        a = a * cupy.exp(-bias*(j - j_c)*dln)

    # compute FHT coefficients
    u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)

    # transform
    A = _fhtq(a, u)

    # bias output array
    if bias != 0:
        # A(k) = A_q(k) (k/k_c)^{-q} (k_c r_c)^{-q}
        A *= cupy.exp(-bias*((j - j_c)*dln + offset))

    return A


@_fft._implements(_scipy_fft.ifht)
def ifht(A, dln, mu, offset=0.0, bias=0.0):
    """Compute the inverse fast Hankel transform.

    Computes the discrete inverse Hankel transform of a logarithmically spaced
    periodic sequence. This is the inverse operation to `fht`.

    Parameters
    ----------
    A : cupy.ndarray (..., n)
        Real periodic input array, uniformly logarithmically spaced.  For
        multidimensional input, the transform is performed over the last axis.
    dln : float
        Uniform logarithmic spacing of the input array.
    mu : float
        Order of the Hankel transform, any positive or negative real number.
    offset : float, optional
        Offset of the uniform logarithmic spacing of the output array.
    bias : float, optional
        Exponent of power law bias, any positive or negative real number.

    Returns
    -------
    a : cupy.ndarray (..., n)
        The transformed output array, which is real, periodic, uniformly
        logarithmically spaced, and of the same shape as the input array.

    See Also
    --------
    :func:`scipy.special.ifht`
    :func:`scipy.special.fhtoffset` : Return an optimal offset for `fht`.

    """

    # size of transform
    n = A.shape[-1]

    # bias input array
    if bias != 0:
        # A_q(k) = A(k) (k/k_c)^{q} (k_c r_c)^{q}
        j_c = (n - 1) / 2
        j = cupy.arange(n)
        A = A * cupy.exp(bias * ((j - j_c) * dln + offset))

    # compute FHT coefficients
    u = fhtcoeff(n, dln, mu, offset=offset, bias=bias)

    # transform
    a = _fhtq(A, u, inverse=True)

    # bias output array
    if bias != 0:
        # a(r) = a_q(r) (r/r_c)^{q}
        a /= cupy.exp(-bias * (j - j_c) * dln)

    return a


def fhtcoeff(n, dln, mu, offset=0.0, bias=0.0):
    '''Compute the coefficient array for a fast Hankel transform.
    '''

    lnkr, q = offset, bias

    # Hankel transform coefficients
    # u_m = (kr)^{-i 2m pi/(n dlnr)} U_mu(q + i 2m pi/(n dlnr))
    # with U_mu(x) = 2^x Gamma((mu+1+x)/2)/Gamma((mu+1-x)/2)
    xp = (mu + 1 + q)/2
    xm = (mu + 1 - q)/2
    y = cupy.linspace(0, math.pi * (n // 2) / (n * dln), n // 2 + 1)
    u = cupy.empty(n // 2 + 1, dtype=complex)
    v = cupy.empty(n // 2 + 1, dtype=complex)
    u.imag[:] = y
    u.real[:] = xm
    loggamma(u, out=v)
    u.real[:] = xp
    loggamma(u, out=u)
    y *= 2 * (LN_2 - lnkr)
    u.real -= v.real
    u.real += LN_2 * q
    u.imag += v.imag
    u.imag += y
    cupy.exp(u, out=u)

    # fix last coefficient to be real
    u.imag[-1] = 0

    # deal with special cases
    if not cupy.isfinite(u[0]):
        # write u_0 = 2^q Gamma(xp)/Gamma(xm) = 2^q poch(xm, xp-xm)
        # poch() handles special cases for negative integers correctly
        u[0] = 2**q * poch(xm, xp - xm)
        # the coefficient may be inf or 0, meaning the transform or the
        # inverse transform, respectively, is singular

    return u


def _fhtq(a, u, inverse=False):
    '''Compute the biased fast Hankel transform.

    This is the basic FFTLog routine.
    '''

    # size of transform
    n = a.shape[-1]

    # check for singular transform or singular inverse transform
    if cupy.isinf(u[0]) and not inverse:
        warn('singular transform; consider changing the bias')
        # fix coefficient to obtain (potentially correct) transform anyway
        u = u.copy()
        u[0] = 0
    elif u[0] == 0 and inverse:
        warn('singular inverse transform; consider changing the bias')
        # fix coefficient to obtain (potentially correct) inverse anyway
        u = u.copy()
        u[0] = cupy.inf

    # biased fast Hankel transform via real FFT
    A = _fft.rfft(a, axis=-1)
    if not inverse:
        # forward transform
        A *= u
    else:
        # backward transform
        A /= u.conj()
    A = _fft.irfft(A, n, axis=-1)
    A = A[..., ::-1]

    return A
