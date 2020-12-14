import cupy
from cupy.linalg import _util
from cupyx import lapack


def invh(a):
    """Compute the inverse of a Hermitian matrix.

    This function computes a inverse of a real symmetric or complex hermitian
    positive-definite matrix using Cholesky factorization. If matrix ``a`` is
    not positive definite, Cholesky factorization fails and it raises an error.

    Args:
        a (cupy.ndarray): Real symmetric or complex hermitian maxtix.

    Returns:
        cupy.ndarray: The inverse of matrix ``a``.
    """

    _util._assert_cupy_array(a)
    _util._assert_nd_squareness(a)

    # TODO: Remove this assert once cusolver supports nrhs > 1 for potrsBatched
    _util._assert_rank2(a)

    n = a.shape[-1]
    identity_matrix = cupy.eye(n, dtype=a.dtype)
    b = cupy.empty(a.shape, a.dtype)
    b[...] = identity_matrix

    return lapack.posv(a, b)
