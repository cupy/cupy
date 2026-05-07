from __future__ import annotations

import numpy

import cupy
from cupy import _core
from cupy.linalg import _decomposition
from cupy.linalg import _util

import functools


def _multi_svd_norm(x, row_axis, col_axis, op):
    y = cupy.moveaxis(x, (row_axis, col_axis), (-2, -1))
    result = op(_decomposition.svd(y, compute_uv=False), axis=-1)
    return result


_norm_ord2 = _core.create_reduction_func(
    '_norm_ord2',
    ('?->l', 'b->l', 'B->L', 'h->l', 'H->L', 'i->l', 'I->L', 'l->l', 'L->L',
     'q->q', 'Q->Q',
     ('e->e', (None, None, None, 'float')),
     'f->f', 'd->d'),
    ('in0 * in0', 'a + b', 'out0 = sqrt(type_out0_raw(a))', None), 0)
_norm_ord2_complex = _core.create_reduction_func(
    '_norm_ord2_complex',
    ('F->f', 'D->d'),
    ('in0.real() * in0.real() + in0.imag() * in0.imag()',
     'a + b', 'out0 = sqrt(type_out0_raw(a))', None), 0)


def norm(x, ord=None, axis=None, keepdims=False):
    """Returns one of matrix norms specified by ``ord`` parameter.

    See numpy.linalg.norm for more detail.

    Args:
        x (cupy.ndarray): Array to take norm. If ``axis`` is None,
            ``x`` must be 1-D or 2-D.
        ord (non-zero int, inf, -inf, 'fro'): Norm type.
        axis (int, 2-tuple of ints, None): 1-D or 2-D norm is computed over
            ``axis``.
        keepdims (bool): If this is set ``True``, the axes which are normed
            over are left.

    Returns:
        cupy.ndarray

    """
    if not issubclass(x.dtype.type, numpy.inexact):
        x = x.astype(float)

    # Immediately handle some default, simple, fast, and common cases.
    if axis is None:
        ndim = x.ndim
        if (ord is None or (ndim == 1 and ord == 2) or
                (ndim == 2 and ord in ('f', 'fro'))):
            if x.dtype.kind == 'c':
                s = abs(x.ravel())
                s *= s
                ret = cupy.sqrt(s.sum())
            else:
                ret = cupy.sqrt((x * x).sum())
            if keepdims:
                ret = ret.reshape((1,) * ndim)
            return ret

    # Normalize the `axis` argument to a tuple.
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception:
            raise TypeError(
                '\'axis\' must be None, an integer or a tuple of integers')
        axis = (axis,)

    if len(axis) == 1:
        if ord == numpy.inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -numpy.inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            # Zero norm
            # Convert to Python float in accordance with NumPy
            return (x != 0).astype(x.real.dtype).sum(
                axis=axis, keepdims=keepdims)
        elif ord == 1:
            # special case for speedup
            return abs(x).sum(axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            # special case for speedup
            if x.dtype.kind == 'c':
                return _norm_ord2_complex(x, axis=axis, keepdims=keepdims)
            return _norm_ord2(x, axis=axis, keepdims=keepdims)
        else:
            try:
                float(ord)
            except TypeError:
                raise ValueError('Invalid norm order for vectors.')

            absx = abs(x)
            absx **= ord
            ret = absx.sum(axis=axis, keepdims=keepdims)
            ret **= cupy.reciprocal(ord, dtype=ret.dtype)
            return ret
    elif len(axis) == 2:
        row_axis, col_axis = axis
        if row_axis < 0:
            row_axis += nd
        if col_axis < 0:
            col_axis += nd
        if not (0 <= row_axis < nd and 0 <= col_axis < nd):
            raise ValueError('Invalid axis %r for an array with shape %r' %
                             (axis, x.shape))
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            op_max = functools.partial(cupy.take, indices=0)
            ret = _multi_svd_norm(x, row_axis, col_axis, op_max)
        elif ord == -2:
            op_min = functools.partial(cupy.take, indices=-1)
            ret = _multi_svd_norm(x, row_axis, col_axis, op_min)
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = abs(x).sum(axis=row_axis).max(axis=col_axis)
        elif ord == numpy.inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = abs(x).sum(axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = abs(x).sum(axis=row_axis).min(axis=col_axis)
        elif ord == -numpy.inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = abs(x).sum(axis=col_axis).min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            if x.dtype.kind == 'c':
                ret = _norm_ord2_complex(x, axis=axis)
            else:
                ret = _norm_ord2(x, axis=axis)
        elif ord == 'nuc':
            ret = _multi_svd_norm(x, row_axis, col_axis, cupy.sum)
        else:
            raise ValueError('Invalid norm order for matrices.')
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        raise ValueError('Improper number of dimensions to norm.')


def cond(x, p=None):
    """Returns the condition number of a matrix.

    This function computes the condition number using one of several norms,
    depending on the value of `p`.

    Args:
        x (cupy.ndarray): The matrix whose condition number is computed.
        p (str, int, optional): The norm type used.
            The following norms are supported:

            - None: 2-norm
            - 'fro': Frobenius norm.
            - inf: max(sum(abs(x), axis=1)).
            - -inf: min(sum(abs(x), axis=1)).
            - 1: max(sum(abs(x), axis=0)).
            - -1: min(sum(abs(x), axis=0)).
            - 2: 2-norm (largest singular value).
            - -2: smallest singular value.

    Returns:
        cupy.ndarray: The condition number of the matrix. May be infinite.
    """
    import cupyx
    # This function is implemented as a direct adaption of the numpy
    # implementation to cupy API. This includes comments.
    x = cupy.asarray(x)  # in case we have a matrix
    if x.size == 0:
        raise cupy.linalg.LinAlgError("cond is not defined on empty arrays")
    if p is None or p == 2 or p == -2:
        s = cupy.linalg.svd(x, compute_uv=False)
        with cupyx.errstate(linalg='ignore'):
            if p == -2:
                r = s[..., -1] / s[..., 0]
            else:
                r = s[..., 0] / s[..., -1]
    else:
        # Call inv(x) ignoring errors. The result array will
        # contain nans in the entries where inversion failed.

        _util._assert_cupy_array(x)
        _util._assert_stacked_2d(x)
        _util._assert_stacked_square(x)
        t, result_t = _util.linalg_common_type(x)
        with cupyx.errstate(linalg='ignore'):
            invx = cupy.linalg.inv(x)
            r = norm(x, p, axis=(-2, -1)) * norm(invx, p, axis=(-2, -1))
        r = r.astype(result_t, copy=False)

    # Convert nans to infs unless the original array had nan entries
    r = cupy.asarray(r)
    nan_mask = cupy.isnan(r)
    if nan_mask.any():
        nan_mask &= ~cupy.isnan(x).any(axis=(-2, -1))
        if r.ndim > 0:
            r[nan_mask] = cupy.inf
        elif nan_mask:
            r[()] = cupy.inf

    return r


def det(a):
    """Returns the determinant of an array.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(..., N, N)``.

    Returns:
        cupy.ndarray: Determinant of ``a``. Its shape is ``a.shape[:-2]``.

    .. seealso:: :func:`numpy.linalg.det`
    """
    sign, logdet = slogdet(a)
    return sign * cupy.exp(logdet)


def matrix_rank(M, tol=None):
    """Return matrix rank of array using SVD method

    Args:
        M (cupy.ndarray): Input array. Its `ndim` must be less than or equal to
            2.
        tol (None or float): Threshold of singular value of `M`.
            When `tol` is `None`, and `eps` is the epsilon value for datatype
            of `M`, then `tol` is set to `S.max() * max(M.shape) * eps`,
            where `S` is the singular value of `M`.
            It obeys :func:`numpy.linalg.matrix_rank`.

    Returns:
        cupy.ndarray: Rank of `M`.

    .. seealso:: :func:`numpy.linalg.matrix_rank`
    """
    if M.ndim < 2:
        return (M != 0).any().astype(int)
    S = _decomposition.svd(M, compute_uv=False)
    if tol is None:
        tol = (S.max(axis=-1, keepdims=True) * max(M.shape[-2:]) *
               numpy.finfo(S.dtype).eps)
    return (S > tol).sum(axis=-1, dtype=numpy.intp)


def slogdet(a):
    """Returns sign and logarithm of the determinant of an array.

    It calculates the natural logarithm of the determinant of a given value.

    Args:
        a (cupy.ndarray): The input matrix with dimension ``(..., N, N)``.

    Returns:
        tuple of :class:`~cupy.ndarray`:
            It returns a tuple ``(sign, logdet)``. ``sign`` represents each
            sign of the determinant as a real number ``0``, ``1`` or ``-1``.
            'logdet' represents the natural logarithm of the absolute of the
            determinant.
            If the determinant is zero, ``sign`` will be ``0`` and ``logdet``
            will be ``-inf``.
            The shapes of both ``sign`` and ``logdet`` are equal to
            ``a.shape[:-2]``.

    .. warning::
        This function calls one or more cuSOLVER routine(s) which may yield
        invalid results if input conditions are not met.
        To detect these invalid results, you can set the `linalg`
        configuration to a value that is not `ignore` in
        :func:`cupyx.errstate` or :func:`cupyx.seterr`.

    .. warning::
        To produce the same results as :func:`numpy.linalg.slogdet` for
        singular inputs, set the `linalg` configuration to `raise`.

    .. seealso:: :func:`numpy.linalg.slogdet`
    """
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)

    dtype, sign_dtype = _util.linalg_common_type(a)
    logdet_dtype = numpy.dtype(sign_dtype.char.lower())

    a_shape = a.shape
    shape = a_shape[:-2]
    n = a_shape[-2]

    if a.size == 0:
        # empty batch (result is empty, too) or empty matrices det([[]]) == 1
        sign = cupy.ones(shape, sign_dtype)
        logdet = cupy.zeros(shape, logdet_dtype)
        return sign, logdet

    lu, ipiv, dev_info = _decomposition._lu_factor(a, dtype)

    # dev_info < 0 means illegal value (in dimensions, strides, and etc.) that
    # should never happen even if the matrix contains nan or inf.
    # TODO(kataoka): assert dev_info >= 0 if synchronization is allowed for
    # debugging purposes.

    diag = cupy.diagonal(lu, axis1=-2, axis2=-1)

    logdet = cupy.log(cupy.abs(diag)).sum(axis=-1)

    # ipiv is 1-origin
    non_zero = cupy.count_nonzero(ipiv != cupy.arange(1, n + 1), axis=-1)
    if dtype.kind == "f":
        non_zero += cupy.count_nonzero(diag < 0, axis=-1)

    # Note: sign == -1 ** (non_zero % 2)
    sign = (non_zero % 2) * -2 + 1
    if dtype.kind == "c":
        sign = sign * cupy.prod(diag / cupy.abs(diag), axis=-1)

    sign = sign.astype(dtype)
    logdet = logdet.astype(logdet_dtype, copy=False)
    singular = dev_info > 0
    return (
        cupy.where(singular, sign_dtype.type(0), sign).reshape(shape),
        cupy.where(singular, logdet_dtype.type('-inf'), logdet).reshape(shape),
    )


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """Returns the sum along the diagonals of an array.

    It computes the sum along the diagonals at ``axis1`` and ``axis2``.

    Args:
        a (cupy.ndarray): Array to take trace.
        offset (int): Index of diagonals. Zero indicates the main diagonal, a
            positive value an upper diagonal, and a negative value a lower
            diagonal.
        axis1 (int): The first axis along which the trace is taken.
        axis2 (int): The second axis along which the trace is taken.
        dtype: Data type specifier of the output.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The trace of ``a`` along axes ``(axis1, axis2)``.

    .. seealso:: :func:`numpy.trace`

    """
    # TODO(okuta): check type
    return a.trace(offset, axis1, axis2, dtype, out)
