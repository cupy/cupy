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
    # TODO: Use `_assert_stacked_2d` instead, once cusolver supports nrhs > 1
    # for potrsBatched
    _util._assert_2d(a)
    _util._assert_stacked_square(a)

    b = _util.stacked_identity_like(a)
    return lapack.posv(a, b)
