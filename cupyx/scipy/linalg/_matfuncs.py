from __future__ import annotations

import math
import cmath

import cupy
from cupy.linalg import _util


def khatri_rao(a, b):
    r"""
    Khatri-rao product

    A column-wise Kronecker product of two matrices

    Parameters
    ----------
    a : (n, k) array_like
        Input array
    b : (m, k) array_like
        Input array

    Returns
    -------
    c:  (n*m, k) ndarray
        Khatri-rao product of `a` and `b`.

    See Also
    --------
    .. seealso:: :func:`scipy.linalg.khatri_rao`

    """

    _util._assert_2d(a)
    _util._assert_2d(b)

    if a.shape[1] != b.shape[1]:
        raise ValueError("The number of columns for both arrays "
                         "should be equal.")

    c = a[..., :, cupy.newaxis, :] * b[..., cupy.newaxis, :, :]
    return c.reshape((-1,) + c.shape[2:])


# ### expm ###

# Backward-error thresholds theta_m for Bader-Blanes-Casas Taylor
# polynomial of degree m, for single and double precision.  Values
# transcribed from Lezcano/expm (same table is reproduced in PyTorch's
# aten/src/ATen/native/LinearAlgebra.cpp).
_BBC_DEGS = (1, 2, 4, 8, 12, 18)

_BBC_THETA_SINGLE = (
    1.192092800768788e-07,  # m=1
    5.978858893805233e-04,  # m=2
    5.116619363445086e-02,  # m=4
    5.800524627688768e-01,  # m=8
    1.461661507209034e+00,  # m=12
    3.010066362817634e+00,  # m=18
)

_BBC_THETA_DOUBLE = (
    2.220446049250313e-16,  # m=1
    2.580956802971767e-08,  # m=2
    3.397168839976962e-04,  # m=4
    4.991228871115323e-02,  # m=8
    2.996158913811580e-01,  # m=12
    1.090863719290036e+00,  # m=18
)

# Precomputed constants for the degree-8 polynomial decomposition of [1].
_BBC_T8_SQRT177 = math.sqrt(177.0)
_BBC_T8_X3 = 2.0 / 3.0
_BBC_T8_X1 = _BBC_T8_X3 * (1.0 + _BBC_T8_SQRT177) / 88.0
_BBC_T8_X2 = _BBC_T8_X3 * (1.0 + _BBC_T8_SQRT177) / 352.0
_BBC_T8_X4 = (-271.0 + 29.0 * _BBC_T8_SQRT177) / (315.0 * _BBC_T8_X3)
_BBC_T8_X5 = (-11.0 + 11.0 * _BBC_T8_SQRT177) / (1260.0 * _BBC_T8_X3)
_BBC_T8_X6 = (-99.0 + 11.0 * _BBC_T8_SQRT177) / (5040.0 * _BBC_T8_X3)
_BBC_T8_X7 = (89.0 - _BBC_T8_SQRT177) / (5040.0 * _BBC_T8_X3 * _BBC_T8_X3)
_BBC_T8_Y2 = (857.0 - 58.0 * _BBC_T8_SQRT177) / 630.0


def expm(a):
    """Compute the matrix exponential.

    Parameters
    ----------
    a : ndarray, 2D

    Returns
    -------
    matrix exponential of `a`

    Notes
    -----
    Uses the Bader-Blanes-Casas optimized Taylor polynomial algorithm
    [1]_ with degrees 1, 2, 4, 8, 12, or 18, selected by the
    trace-shifted matrix 1-norm, followed by scaling and squaring.
    Matrix balancing is not performed.

    References
    ----------
    .. [1] P. Bader, S. Blanes, and F. Casas, "Computing the Matrix
       Exponential with an Optimized Taylor Polynomial Algorithm",
       Mathematics 7(12), 1174 (2019).
       https://doi.org/10.3390/math7121174

    """
    if a.size == 0:
        return cupy.zeros((0, 0), dtype=a.dtype)

    n = a.shape[0]

    # follow scipy.linalg.expm dtype handling
    a_dtype = a.dtype if cupy.issubdtype(
        a.dtype, cupy.inexact) else cupy.float64

    # try reducing the norm
    mu = cupy.diag(a).sum() / n
    E = cupy.eye(n, dtype=a_dtype)
    A = a - E*mu

    thetas = (_BBC_THETA_SINGLE
              if a_dtype in (cupy.float32, cupy.complex64)
              else _BBC_THETA_DOUBLE)

    nrmA = cupy.linalg.norm(A, ord=1).item()
    s = 0
    if nrmA > thetas[-1]:
        s = int(math.ceil(math.log2(nrmA / thetas[-1])))
        A = A / (2 ** s)
        nrmA = nrmA / (2 ** s)

    deg = _BBC_DEGS[-1]
    for d, th in zip(_BBC_DEGS, thetas):
        if nrmA <= th:
            deg = d
            break

    x = _bbc_taylor(A, E, deg)

    for _ in range(s):
        x = x @ x

    # undo preprocessing
    emu = cmath.exp(mu) if cupy.issubdtype(
        mu.dtype, cupy.complexfloating) else math.exp(mu)
    x *= emu

    return x


def _bbc_taylor(A, E, deg):
    """Return the degree-`deg` BBC Taylor approximation of exp(A)."""
    if deg == 1:
        return E + A
    if deg == 2:
        A2 = A @ A
        return _bbc_t2(E, A, A2)
    if deg == 4:
        A2 = A @ A
        return _bbc_t4_assemble(E, A, A2 @ _bbc_t4_inner(E, A, A2))
    if deg == 8:
        A2 = A @ A
        A4 = A2 @ _bbc_t8_left1(A, A2)
        A8 = _bbc_t8_left2(A2, A4) @ _bbc_t8_right(E, A, A2, A4)
        return _bbc_t8_finalize(E, A, A2, A8)
    # TODO: in deg 12 and deg 18 the post-matmul adds could fold into
    # the matmul itself (C = A@B + βC) once nvmath-python's advanced
    # matmul or cupy.fuse's matmul support becomes available.
    if deg == 12:
        A2 = A @ A
        A3 = A @ A2
        q31, q32, q33, q34 = _bbc_t12_qs(E, A, A2, A3)
        qaux = q33 + q34 @ q34
        return q31 + (q32 + qaux) @ qaux
    if deg == 18:
        A2 = A @ A
        A3 = A @ A2
        A6 = A3 @ A3
        q31, q61, q62, q63, q64 = _bbc_t18_qs(E, A, A2, A3, A6)
        q91 = q31 @ q64 + q63
        return q61 + (q62 + q91) @ q91
    raise ValueError(f'unsupported BBC degree: {deg}')


@cupy.fuse
def _bbc_t2(E, A, A2):
    return E + A + 0.5 * A2


@cupy.fuse
def _bbc_t4_inner(E, A, A2):
    return 0.5 * E + A / 6.0 + A2 / 24.0


@cupy.fuse
def _bbc_t4_assemble(E, A, A2_inner):
    return E + A + A2_inner


@cupy.fuse
def _bbc_t8_left1(A, A2):
    return _BBC_T8_X1 * A + _BBC_T8_X2 * A2


@cupy.fuse
def _bbc_t8_left2(A2, A4):
    return _BBC_T8_X3 * A2 + A4


@cupy.fuse
def _bbc_t8_right(E, A, A2, A4):
    return (_BBC_T8_X4 * E + _BBC_T8_X5 * A + _BBC_T8_X6 * A2
            + _BBC_T8_X7 * A4)


@cupy.fuse
def _bbc_t8_finalize(E, A, A2, A8):
    return E + A + _BBC_T8_Y2 * A2 + A8


# Degree-12 coefficient matrix b_12 from PyTorch's compute_T12
# (aten/src/ATen/native/LinearAlgebra.cpp).  Row i = coefficients of
# q_i in powers [I, A, A2, A3].  The resulting p_12(A) is identical
# to Lezcano's factorisation, but this parametrisation preserves
# ~20% more accuracy in float32 / float64 / complex64 / complex128
# across the deg-12 norm band.
@cupy.fuse
def _bbc_t12_qs(E, A, A2, A3):
    q31 = (9.0198e-16 * E
           + 0.46932117595418237389 * A
           - 0.20099424927047284052 * A2
           - 0.04623946134063071740 * A3)
    q32 = (5.31597895759871264183 * E
           + 1.19926790417132231573 * A
           + 0.01179296240992997031 * A2
           + 0.01108844528519167989 * A3)
    q33 = (0.18188869982170434744 * E
           + 0.05502798439925399070 * A
           + 0.09351590770535414968 * A2
           + 0.00610700528898058230 * A3)
    q34 = (-2.0861320e-13 * E
           - 0.13181061013830184015 * A
           - 0.02027855540589259079 * A2
           - 0.00675951846863086359 * A3)
    return q31, q32, q33, q34


# Degree-18 coefficient matrix from [1] Table 3.  Row i = coefficients of
# q_i in powers [I, A, A2, A3, A6].
@cupy.fuse
def _bbc_t18_qs(E, A, A2, A3, A6):
    q31 = (-1.00365581030144618291e-01 * A
           - 8.02924648241156932449e-03 * A2
           - 8.92138498045729985177e-04 * A3)
    q61 = (3.97849749499645077844e-01 * A
           + 1.36783778460411720168e+00 * A2
           + 4.98289622525382669416e-01 * A3
           - 6.37898194594723280150e-04 * A6)
    q62 = (-1.09676396052962061844e+01 * E
           + 1.68015813878906206114e+00 * A
           + 5.71779846478865511061e-02 * A2
           - 6.98210122488052056106e-03 * A3
           + 3.34975017086070470649e-05 * A6)
    q63 = (-9.04316832390810593223e-02 * E
           - 6.76404519071381882256e-02 * A
           + 6.75961301770459654925e-02 * A2
           + 2.95552570429315521194e-02 * A3
           - 1.39180257516060693404e-05 * A6)
    q64 = (-9.23364619367118555360e-02 * A2
           - 1.69364939002081722752e-02 * A3
           - 1.40086798182036094347e-05 * A6)
    return q31, q61, q62, q63, q64
