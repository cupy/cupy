from __future__ import annotations

import numpy

from cupy.linalg import _util
import cupy


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
    from cupyx import lapack

    _util._assert_cupy_array(a)
    # TODO: Use `_assert_stacked_2d` instead, once cusolver supports nrhs > 1
    # for potrsBatched
    _util._assert_2d(a)
    _util._assert_stacked_square(a)

    b = _util.stacked_identity_like(a)
    return lapack.posv(a, b)


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """
    Solves the linear system Ax = b using the Cholesky factorization of A.
    Batched input arrays are also supported.

    Args:
        c (tuple of cupy.ndarray and bool): The first tuple item is the
            Cholesky factor of the matrix `A`, typically obtained from
            `cupy.linalg.cholesky`. The second item is a bool that indicates
            whether the Cholesky factor is stored in the lower-triangular
            part (when True) or in the upper-triangular part (when False).
        b (cupy.ndarray): Right-hand side array.
        overwrite_b (bool, optional): If True, the contents of `b` may be
            overwritten for efficiency. Default is False.
        check_finite (bool, optional): If True, checks whether the input arrays
            contain only finite numbers. Disabling this may improve performance
            but can lead to crashes or non-termination if NaNs or infs are
            present. Default is True.

    Returns:
        cupy.ndarray: Solution to the linear system Ax = b.

    See Also:
        cupy.linalg.cho_factor: Computes the Cholesky factorization.

    Examples:
        >>> import cupy as cp
        >>> from cupy.linalg import cholesky, cho_solve
        >>> A = cp.array([[9, 3, 1, 5],
        ...               [3, 7, 5, 1],
        ...               [1, 5, 9, 2],
        ...               [5, 1, 2, 6]])
        >>> c= cholesky(A)
        >>> x = cho_solve((c, True), cp.ones(4))
        >>> cp.allclose(A @ x, cp.ones(4))
        True
    """
    from cupyx import lapack

    (c, lower) = c_and_lower

    # Check finiteness of input arrays
    if check_finite:
        indexes = (numpy.tril_indices(c.shape[-1], -1) if lower else
                   numpy.triu_indices(c.shape[-1], 1))
        if not cupy.isfinite(c[..., indexes[0], indexes[1]]).all():
            raise ValueError("Input array contains NaN or infinity.")
        if not cupy.isfinite(b).all():
            raise ValueError("Input array contains NaN or infinity.")

    # cupyx.lapack.potrs may overwrite b, so we need to ensure it will not
    # happen
    if not overwrite_b:
        if c.ndim == 2:
            # Handled by potrs (non-batched case)
            if b.flags.f_contiguous:
                b = cupy.asarray(b, order='F', copy=True)
        else:
            # Handled by potrs (batched case)
            if b.flags.c_contiguous:
                b = cupy.asarray(b, order='C', copy=True)

    return lapack.potrs(c, b, lower=lower)
