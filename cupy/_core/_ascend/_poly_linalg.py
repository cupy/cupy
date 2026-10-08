"""Ascend replacements for the ``cupy.linalg`` pieces used by polynomial code.

``cupy.linalg`` is not importable on the Ascend backend yet
(``cupy/linalg/_solve.py`` imports ``cupy.cuda.device``), but
``cupy.lib._routines_poly`` needs ``lstsq`` (polyfit) and ``inv`` (polyfit's
covariance branch). This module provides both as pure compositions of the
already-registered aclnn ops:

* ``ascend_svd``  (via ``cupy._core._routines_linalg._ascend_svd``)
* ``ascend_inverse`` (via ``..._ascend_inv``)
* ``ascend_matmul`` / ``ascend_dot`` (via the ``@`` operator)

The Vh orientation of ``_ascend_svd`` is interpreted exactly like the Ascend
branch of ``cupy/linalg/_decomposition.py::svd`` (``vh = v.transpose().conj()``)
so both call sites stay consistent; the aclnnSvd orientation needs L4
verification on real hardware (see docs/ascend/polynomial_note.md).
"""
from __future__ import annotations

import numpy
import cupy
from cupy._core import _routines_linalg as _linalg


def _svd_reduced(a):
    """Reduced SVD of a 2-D array, returning ``(u, s, vh)``."""
    u, sigma, v = _linalg._ascend_svd(a, False, True)
    return u, sigma, v.transpose().conj()


def inv(a):
    """``cupy.linalg.inv`` replacement (2-D only, via aclnnInverse)."""
    a = cupy.asarray(a)
    if a.ndim != 2:
        raise NotImplementedError(
            'inv: only 2-D input is supported on Ascend')
    return _linalg._ascend_inv(a)


def lstsq(a, b, rcond=None):
    """``cupy.linalg.lstsq`` replacement via the SVD pseudo-inverse.

    Returns ``(x, resids, rank, s)`` like ``numpy.linalg.lstsq``.
    NumPy semantics for `resides` is the sum of sequared residuals,
    returns only for over-determined ``m <= n`` and full rank ```rank == n`
    Otherwise, an empty array is returned
    """
    a = cupy.asarray(a)
    b = cupy.asarray(b)
    if a.ndim != 2:
        raise ValueError('lstsq: a must be a 2-d array')
    if b.ndim not in (1, 2):
        raise ValueError('lstsq: b must be 1-d or 2-d')
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            'lstsq: incompatible dimensions '
            '(a.shape[0]={}, b.shape[0]={})'.format(a.shape[0], b.shape[0]))

    m, n = a.shape
    one_d = b.ndim == 1
    bb = b.reshape(m, -1)

    try:
        u, s, vh = _svd_reduced(a)
    except RuntimeError:
        # aclnnSvd accaionally return ret=500003
        from cupy._core._ascend import cpu_fallback
        u_np, s_np, vh_np = cpu_fallback.call(
            numpy.linalg.svd, cupy.asnumpy(a), fullmatrices=False, compute_uv=True)
        u, s, vh = (cupy.asarray(u_np), cupy.asarray(s_np), cupy.asarray(vh_np))

    if rcond is None:
        eps = cupy.finfo(s.dtype).eps
        rcond = max(m, n) * eps
    cutoff = rcond * s[0] if s.size > 0 else 0

    keep = s > cutoff
    rank = int(cupy.count_nonzero(keep))
    s_safe = cupy.where(s > 0, s, cupy.array(1, dtype=s.dtype))
    sinv = cupy.where(keep, 1.0 / s_safe, 0.0)

    # minimum-norm solution: x = Vh^H diag(1/s) U^H b
    x = (vh.conj().T * sinv) @ (u.conj().T @ bb)   # (n, k)

    # NumPy semantics: resids is empty when not over-determined or
    # when the coeff matrix is rank-deficient.
    if m > n and rank == n:
        resids = ((a @ x - bb) ** 2).sum(axis=0)
    else:
        resids = cupy.empty((0,), dtype=s.dtype)
    return (x[:, 0] if one_d else x), resids, rank, s
