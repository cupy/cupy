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

# Identity coefficients of q32, q33 (degree 12) and q62, q63 (degree 18).
# They are multiplied out by hand in _bbc_high_m1 instead of entering a
# matmul.
_BBC_T12_C32 = 5.31597895759871264183
_BBC_T12_C33 = 0.18188869982170434744
_BBC_T18_C62 = -1.09676396052962061844e+01
_BBC_T18_C63 = -9.04316832390810593223e-02

# Smallest n (single, double precision) at which the scaling is refined
# with the bound of [2]. Its two extra norm evaluations cost as much as
# several squarings while a matmul is not much dearer than a kernel launch.
_BBC_SHARP_MIN_N = (2048, 512)


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
    For large matrices the scaling is then reduced by using
    ``max(||A^2||^(1/2), ||A^3||^(1/3))`` in place of ``||A||`` [2]_.
    Both stages work on ``exp(A) - I`` and the identity is added at the
    end. Matrix balancing is not performed.

    References
    ----------
    .. [1] P. Bader, S. Blanes, and F. Casas, "Computing the Matrix
       Exponential with an Optimized Taylor Polynomial Algorithm",
       Mathematics 7(12), 1174 (2019).
       https://doi.org/10.3390/math7121174
    .. [2] A. H. Al-Mohy and N. J. Higham, "A New Scaling and Squaring
       Algorithm for the Matrix Exponential", SIAM J. Matrix Anal. Appl.
       31(3), 970-989 (2009).
       https://doi.org/10.1137/09074721X

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

    single = a_dtype in (cupy.float32, cupy.complex64)
    thetas = _BBC_THETA_SINGLE if single else _BBC_THETA_DOUBLE

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

    # x = exp(A) - I. An O(1) identity inside a matmul gets its diagonal
    # rounded relative to 1 on every accumulation; that error grows with n.
    if deg <= 8:
        x = _bbc_low_m1(A, E, deg)
    else:
        sharp = n >= _BBC_SHARP_MIN_N[0 if single else 1]
        x, s = _bbc_high_m1(A, E, deg, s, thetas, sharp)
    del A, E

    # (I + x) @ (I + x) - I
    for _ in range(s):
        x = _expm_square_m1(x, x @ x)

    d = x.diagonal()
    d += 1

    # undo preprocessing
    emu = cmath.exp(mu) if cupy.issubdtype(
        mu.dtype, cupy.complexfloating) else math.exp(mu)
    x *= emu

    return x


def _bbc_low_m1(A, E, deg):
    """Return T(A) - I for the BBC Taylor polynomial T of degree <= 8.

    The result may be `A` itself.
    """
    if deg == 1:
        return A
    A2 = A @ A
    if deg == 2:
        return _bbc_t2(A, A2)
    if deg == 4:
        return _bbc_t4_assemble(A, A2 @ _bbc_t4_inner(E, A, A2))
    if deg == 8:
        A4 = A2 @ _bbc_t8_left1(A, A2)
        A8 = _bbc_t8_left2(A2, A4) @ _bbc_t8_right(E, A, A2, A4)
        return _bbc_t8_finalize(A, A2, A8)
    raise ValueError(f'unsupported BBC degree: {deg}')


def _bbc_high_m1(A, E, deg, s, thetas, sharp):
    """Return ``(T(A) - I, s)`` for the BBC Taylor polynomial T of degree
    12 or 18.

    `A` must already be scaled by ``2**-s``. With `sharp`, part of that
    scaling may be undone in place (`s` is lowered to match) and the
    degree may drop from 18 to 12.
    """
    A2 = A @ A
    A3 = A @ A2
    if sharp and deg == 18:
        # ||A^k|| <= alpha^k for every k >= 2 (write k = 2i + 3j), so alpha
        # can replace ||A|| in the truncation bound behind thetas [2]. For
        # large dense matrices ||A|| is several times larger than alpha.
        alpha = max(math.sqrt(cupy.linalg.norm(A2, ord=1).item()),
                    cupy.linalg.norm(A3, ord=1).item() ** (1 / 3))
        # Trust alpha only if A2 and A3 cannot have underflowed.
        finfo = cupy.finfo(A.dtype)
        if alpha ** 3 * finfo.eps ** 2 > finfo.tiny:
            k = min(s, math.floor(math.log2(thetas[-1] / alpha)))
            if k > 0:
                # scaling by powers of two is exact
                A *= 2.0 ** k
                A2 *= 4.0 ** k
                A3 *= 8.0 ** k
                alpha *= 2.0 ** k
                s -= k
            if alpha <= thetas[-2]:
                deg = 12
    # Both schemes of [1] end in (c*I + left) @ (c'*I + right) with
    # c*c' = 1. The identity parts are multiplied out by hand, so only
    # left @ right goes through the matmul.
    # TODO: the post-matmul adds could fold into the matmul itself
    # (C = A@B + βC) once nvmath-python's advanced matmul or cupy.fuse's
    # matmul support becomes available.
    if deg == 12:
        p31, r32, r33, q34 = _bbc_t12_qs(E, A, A2, A3)
        del A2, A3
        right, left = _bbc_mid(r32, r33, q34 @ q34)
        del r32, r33, q34
        return _bbc_t12_finalize(p31, right, left, left @ right), s
    if deg == 18:
        A6 = A3 @ A3
        q31, q61, r62, r63, q64 = _bbc_t18_qs(A, A2, A3, A6)
        del A2, A3, A6
        right, left = _bbc_mid(r62, r63, q31 @ q64)
        del q31, r62, r63, q64
        return _bbc_t18_finalize(q61, right, left, left @ right), s
    raise ValueError(f'unsupported BBC degree: {deg}')


@cupy.fuse
def _expm_square_m1(x, x2):
    return x2 + 2.0 * x


@cupy.fuse
def _bbc_t2(A, A2):
    return A + 0.5 * A2


@cupy.fuse
def _bbc_t4_inner(E, A, A2):
    return 0.5 * E + A / 6.0 + A2 / 24.0


@cupy.fuse
def _bbc_t4_assemble(A, A2_inner):
    return A + A2_inner


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
def _bbc_t8_finalize(A, A2, A8):
    return A + _BBC_T8_Y2 * A2 + A8


# Degree-12 coefficient matrix b_12 from PyTorch's compute_T12
# (aten/src/ATen/native/LinearAlgebra.cpp).  Row i = coefficients of
# q_i in powers [I, A, A2, A3].  The resulting p_12(A) is identical
# to Lezcano's factorisation, but this parametrisation preserves
# ~20% more accuracy in float32 / float64 / complex64 / complex128
# across the deg-12 norm band.
# q32 = _BBC_T12_C32*I + r32 and q33 = _BBC_T12_C33*I + r33; the 9e-16
# identity term of q31 only completes c*c' to 1 and is dropped (p31).
@cupy.fuse
def _bbc_t12_qs(E, A, A2, A3):
    p31 = (0.46932117595418237389 * A
           - 0.20099424927047284052 * A2
           - 0.04623946134063071740 * A3)
    r32 = (1.19926790417132231573 * A
           + 0.01179296240992997031 * A2
           + 0.01108844528519167989 * A3)
    r33 = (0.05502798439925399070 * A
           + 0.09351590770535414968 * A2
           + 0.00610700528898058230 * A3)
    q34 = (-2.0861320e-13 * E
           - 0.13181061013830184015 * A
           - 0.02027855540589259079 * A2
           - 0.00675951846863086359 * A3)
    return p31, r32, r33, q34


@cupy.fuse
def _bbc_t12_finalize(p31, right, left, m):
    return (p31 + (_BBC_T12_C32 + _BBC_T12_C33) * right
            + _BBC_T12_C33 * left + m)


# Degree-18 coefficient matrix from [1] Table 3.  Row i = coefficients of
# q_i in powers [I, A, A2, A3, A6], with q62 = _BBC_T18_C62*I + r62 and
# q63 = _BBC_T18_C63*I + r63.
@cupy.fuse
def _bbc_t18_qs(A, A2, A3, A6):
    q31 = (-1.00365581030144618291e-01 * A
           - 8.02924648241156932449e-03 * A2
           - 8.92138498045729985177e-04 * A3)
    q61 = (3.97849749499645077844e-01 * A
           + 1.36783778460411720168e+00 * A2
           + 4.98289622525382669416e-01 * A3
           - 6.37898194594723280150e-04 * A6)
    r62 = (1.68015813878906206114e+00 * A
           + 5.71779846478865511061e-02 * A2
           - 6.98210122488052056106e-03 * A3
           + 3.34975017086070470649e-05 * A6)
    r63 = (-6.76404519071381882256e-02 * A
           + 6.75961301770459654925e-02 * A2
           + 2.95552570429315521194e-02 * A3
           - 1.39180257516060693404e-05 * A6)
    q64 = (-9.23364619367118555360e-02 * A2
           - 1.69364939002081722752e-02 * A3
           - 1.40086798182036094347e-05 * A6)
    return q31, q61, r62, r63, q64


@cupy.fuse
def _bbc_t18_finalize(q61, right, left, m):
    return (q61 + (_BBC_T18_C62 + _BBC_T18_C63) * right
            + _BBC_T18_C63 * left + m)


# right = r_b + m and left = r_a + right, i.e. the last two factors of the
# degree-12/18 schemes without their identity parts.
@cupy.fuse
def _bbc_mid(r_a, r_b, m):
    right = r_b + m
    return right, r_a + right
