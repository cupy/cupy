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
b = [64764752532480000.,
     32382376266240000.,
     7771770303897600.,
     1187353796428800.,
     129060195264000.,
     10559470521600.,
     670442572800.,
     33522128640.,
     1323241920.,
     40840800.,
     960960.,
     16380.,
     182.,
     1.,]

th13 = 5.371920351148152


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
    Uses Algorithm 2.3 of [1]_: a Pade approximant of order 3, 5, 7, 9,
    or 13 with scaling and squaring. Matrix balancing is not performed.

    References
    ----------
    .. [1] N. Higham, SIAM J. MATRIX ANAL. APPL. Vol. 26(4), p. 1179 (2005)
       https://doi.org/10.1137/04061101X

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

    # Double-precision thresholds from Table 2.3 of [1].
    nrmA = cupy.linalg.norm(A, ord=1).item()
    s = 0
    if nrmA > th13:
        s = int(math.ceil(math.log2(nrmA / th13)))
        A /= 2**s

    # Compute only the matrix powers needed by the selected Pade order.
    A2 = A @ A
    if nrmA < 1.495585217958292e-2:
        u, v = _expm_inner3(E, A2)
    else:
        A4 = A2 @ A2
        if nrmA < 2.539398330063230e-1:
            u, v = _expm_inner5(E, A2, A4)
        else:
            A6 = A2 @ A4
            if nrmA < 9.504178996162932e-1:
                u, v = _expm_inner7(E, A2, A4, A6)
            elif nrmA < 2.097847961257068:
                A8 = A4 @ A4
                u, v = _expm_inner9(E, A2, A4, A6, A8)
            else:
                bb = cupy.asarray(b, dtype=a_dtype)
                u1, u2, v1, v2 = _expm_inner(E, A2, A4, A6, bb)
                u = A6 @ u1 + u2
                v = A6 @ v1 + v2
    u = A @ u

    # squaring
    x = cupy.linalg.solve(-u + v, u + v)
    for _ in range(s):
        x = x @ x

    # undo preprocessing
    emu = cmath.exp(mu) if cupy.issubdtype(
        mu.dtype, cupy.complexfloating) else math.exp(mu)
    x *= emu

    return x


@cupy.fuse
def _expm_inner3(E, A2):
    return A2 + 60.*E, 12.*A2 + 120.*E


@cupy.fuse
def _expm_inner5(E, A2, A4):
    u = A4 + 420.*A2 + 15120.*E
    v = 30.*A4 + 3360.*A2 + 30240.*E
    return u, v


@cupy.fuse
def _expm_inner7(E, A2, A4, A6):
    u = A6 + 1512.*A4 + 277200.*A2 + 8648640.*E
    v = 56.*A6 + 25200.*A4 + 1995840.*A2 + 17297280.*E
    return u, v


@cupy.fuse
def _expm_inner9(E, A2, A4, A6, A8):
    u = A8 + 3960.*A6 + 2162160.*A4 + 302702400.*A2 + 8821612800.*E
    v = (90.*A8 + 110880.*A6 + 30270240.*A4 + 2075673600.*A2
         + 17643225600.*E)
    return u, v


@cupy.fuse
def _expm_inner(E, A2, A4, A6, b):
    u1 = b[13]*A6 + b[11]*A4 + b[9]*A2
    u2 = b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*E

    v1 = b[12]*A6 + b[10]*A4 + b[8]*A2
    v2 = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*E
    return u1, u2, v1, v2
