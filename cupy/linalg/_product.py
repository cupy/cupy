from __future__ import annotations

import collections.abc
import numbers
import warnings

import numpy

import cupy
from cupy import _core
from cupy._core import internal
from cupy._core._gufuncs import _GUFunc
from cupy.linalg import _solve
from cupy.linalg import _util


def _try_fastcall_matmul(
        a, b, out=None, axes=None, axis=None, keepdims=False,
        casting=None, dtype=None, signature=None, order=None):
    # Allow matmul to side-step gufunc machinery when no gufunc args
    # are being used.  In theory e.g. dtype may be possible to handle.
    # Matmul supports all inputs, but casting is tricky.
    if (
        axes is not None
        or axis is not None
        or keepdims
        or casting != "same_kind"
        or dtype is not None
        or signature is not None
        or order != 'K'
    ):
        return NotImplemented

    if out is not None:
        # Output casting would require passing on `casting=` so bail
        # if there is any chance that casting may be needed.
        if (
                not isinstance(out, cupy.ndarray)
                or not isinstance(a, cupy.ndarray)
                or not isinstance(b, cupy.ndarray)
        ):
            return NotImplemented

        if out.dtype != a.dtype or out.dtype != b.dtype:
            return NotImplemented

    return _core.matmul(a, b, out=out)


matmul = _GUFunc(
    _core.matmul,
    '(n?,k),(k,m?)->(n?,m?)',
    supports_batched=True,
    supports_out=True,
    try_fastcall_func=_try_fastcall_matmul,
    doc="""matmul(x1, x2, /, out=None, \\*\\*kwargs)

    Matrix product of two arrays.

    Returns the matrix product of two arrays and is the implementation of
    the `@` operator introduced in Python 3.5 following PEP465.

    The main difference against cupy.dot are the handling of arrays with more
    than 2 dimensions. For more information see :func:`numpy.matmul`.

    Args:
        x1 (cupy.ndarray): The left argument.
        x2 (cupy.ndarray): The right argument.
        out (cupy.ndarray, optional): Output array.
        \\*\\*kwargs: ufunc keyword arguments.

    Returns:
        cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.matmul`
    """
)


def dot(a, b, out=None):
    """Returns a dot product of two arrays.

    For arrays with more than one axis, it computes the dot product along the
    last axis of ``a`` and the second-to-last axis of ``b``. This is just a
    matrix product if the both arrays are 2-D. For 1-D arrays, it uses their
    unique axis as an axis to take dot product over.

    Args:
        a (cupy.ndarray): The left argument.
        b (cupy.ndarray): The right argument.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: The dot product of ``a`` and ``b``.

    .. seealso:: :func:`numpy.dot`

    """
    # TODO(okuta): check type
    return a.dot(b, out)


def vdot(a, b):
    """Returns the dot product of two vectors.

    The input arrays are flattened into 1-D vectors and then it performs inner
    product of these vectors.

    Args:
        a (cupy.ndarray): The first argument.
        b (cupy.ndarray): The second argument.

    Returns:
        cupy.ndarray: Zero-dimensional array of the dot product result.

    .. seealso:: :func:`numpy.vdot`

    """
    if a.size != b.size:
        raise ValueError('Axis dimension mismatch')
    if a.dtype.kind == 'c':
        a = a.conj()

    return _core.tensordot_core(a, b, None, 1, 1, a.size, ())


def _vecdot(x1, x2, out=None):
    if x1.dtype.kind == 'c':
        x1 = x1.conj()
    res = cupy.multiply(x1, x2).sum(axis=-1, out=out)
    return res


vecdot = _GUFunc(
    _vecdot,
    '(n),(n)->()',
    supports_batched=True,
    supports_out=True,
    doc=r"""vecdot(x1, x2, /, \*, axis=-1, out=None, \*\*kwargs)

    Vector dot product of two arrays.

    Returns the dot product of two vectors. The dot product is computed over
    the dimension specified by ``axis``. For complex input, it conjugates the
    first argument.

    Args:
        x1 (cupy.ndarray): First input array.
        x2 (cupy.ndarray): Second input array.
        axis (int): Axis over which to compute the dot product.
            The default is -1.
        out (cupy.ndarray, optional): Output array.
        \*\*kwargs: ufunc keyword arguments.

    Returns:
        cupy.ndarray: The vector dot product of ``x1`` and ``x2``.

    .. seealso:: :func:`numpy.vecdot`
    """
)


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Returns the cross product of two vectors.

    The cross product of ``a`` and ``b`` in :math:`R^3` is a vector
    perpendicular to both ``a`` and ``b``.  If ``a`` and ``b`` are arrays
    of vectors, the vectors are defined by the last axis of ``a`` and ``b``
    by default, and these axes can have dimensions 2 or 3.  Where the
    dimension of either ``a`` or ``b`` is 2, the third component of the input
    vector is assumed to be zero and the cross product calculated accordingly.
    In cases where both input vectors have dimension 2, the z-component of
    the cross product is returned.

    Args:
        a (cupy.ndarray): Components of the first vector(s).
        b (cupy.ndarray): Components of the second vector(s).
        axisa (int, optional):
            Axis of ``a`` that defines the vector(s).
            By default, the last axis.
        axisb (int, optional):
            Axis of ``b`` that defines the vector(s).
            By default, the last axis.
        axisc (int, optional):
            Axis of ``c`` containing the cross product vector(s).  Ignored if
            both input vectors have dimension 2, as the return is scalar.
            By default, the last axis.
        axis (int, optional):
            If defined, the axis of ``a``, ``b`` and ``c``
            that defines the vector(s) and cross product(s).
            Overrides ``axisa``, ``axisb`` and ``axisc``.

    Returns:
        cupy.ndarray :
            Vector cross product(s).

    .. seealso:: :func:`numpy.cross`

    """

    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3
    a = cupy.asarray(a)
    b = cupy.asarray(b)
    # Check axisa and axisb are within bounds
    axisa = internal._normalize_axis_index(axisa, a.ndim)
    axisb = internal._normalize_axis_index(axisb, b.ndim)

    # Move working axis to the end of the shape
    a = cupy.moveaxis(a, axisa, -1)
    b = cupy.moveaxis(b, axisb, -1)
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        msg = ('incompatible dimensions for cross product\n'
               '(dimension must be 2 or 3)')
        raise ValueError(msg)
    if a.shape[-1] == 2 or b.shape[-1] == 2:
        # Deprecated in NumPy 2.0, 2023-09-26
        warnings.warn(
            "Arrays of 2-dimensional vectors are deprecated. Use arrays of "
            "3-dimensional vectors instead.",
            DeprecationWarning, stacklevel=2
        )

    # Create the output array
    shape = cupy.broadcast(a[..., 0], b[..., 0]).shape
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
        # Check axisc is within bounds
        axisc = internal._normalize_axis_index(axisc, len(shape))
    dtype = cupy.promote_types(a.dtype, b.dtype)
    cp = cupy.empty(shape, dtype)

    # create local aliases for readability
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]

    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # a0 * b1 - a1 * b0
            cupy.multiply(a0, b1, out=cp)
            cp -= a1 * b0
            return cp
        else:
            assert b.shape[-1] == 3
            # cp0 = a1 * b2 - 0  (a2 = 0)
            # cp1 = 0 - a0 * b2  (a2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            cupy.multiply(a1, b2, out=cp0)
            cupy.multiply(a0, b2, out=cp1)
            cupy.negative(cp1, out=cp1)
            cupy.multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0
    else:
        assert a.shape[-1] == 3
        if b.shape[-1] == 3:
            # cp0 = a1 * b2 - a2 * b1
            # cp1 = a2 * b0 - a0 * b2
            # cp2 = a0 * b1 - a1 * b0
            cupy.multiply(a1, b2, out=cp0)
            tmp = a2 * b1
            cp0 -= tmp
            cupy.multiply(a2, b0, out=cp1)
            cupy.multiply(a0, b2, out=tmp)
            cp1 -= tmp
            cupy.multiply(a0, b1, out=cp2)
            cupy.multiply(a1, b0, out=tmp)
            cp2 -= tmp
        else:
            assert b.shape[-1] == 2
            # cp0 = 0 - a2 * b1  (b2 = 0)
            # cp1 = a2 * b0 - 0  (b2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            cupy.multiply(a2, b1, out=cp0)
            cupy.negative(cp0, out=cp0)
            cupy.multiply(a2, b0, out=cp1)
            cupy.multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0

    return cupy.moveaxis(cp, -1, axisc)


def linalg_cross(x1, x2, /, *, axis=-1):
    """Returns the cross product of 3-element vectors.

    If ``x1`` and/or ``x2`` are multi-dimensional arrays, then
    the cross-product of each pair of corresponding 3-element vectors
    is independently computed.

    This function is Array API compatible, contrary to
    :func:`cupy.cross`.

    Parameters
    ----------
    x1 : ndarray
        The first input array.
    x2 : array_like
        The second input array. Must be compatible with ``x1`` for all
        non-compute axes. The size of the axis over which to compute
        the cross-product must be the same size as the respective axis
        in ``x1``.
    axis : int, optional
        The axis (dimension) of ``x1`` and ``x2`` containing the vectors for
        which to compute the cross-product. Default: ``-1``.

    Returns
    -------
    out : ndarray
        An array containing the cross products.

    See Also
    --------
    cupy.cross
    numpy.linalg.cross
    """
    if x1.shape != x2.shape:
        raise ValueError('x1 and x2 must have the same shape')
    if x1.ndim == 0:
        raise ValueError('cross() requires arrays of dimension at least 1')
    if x1.shape[axis] != 3:
        raise ValueError('cross() dimension must equal 3')
    return cross(x1, x2, axis=axis)


def inner(a, b):
    """Returns the inner product of two arrays.

    It uses the last axis of each argument to take sum product.

    Args:
        a (cupy.ndarray): The first argument.
        b (cupy.ndarray): The second argument.

    Returns:
        cupy.ndarray: The inner product of ``a`` and ``b``.

    .. seealso:: :func:`numpy.inner`

    """
    a_ndim = a.ndim
    b_ndim = b.ndim
    if a_ndim == 0 or b_ndim == 0:
        return cupy.multiply(a, b)

    a_axis = a_ndim - 1
    b_axis = b_ndim - 1

    if a.shape[-1] != b.shape[-1]:
        raise ValueError('Axis dimension mismatch')

    if a_axis:
        a = cupy.rollaxis(a, a_axis, 0)
    if b_axis:
        b = cupy.rollaxis(b, b_axis, 0)

    ret_shape = a.shape[1:] + b.shape[1:]

    k = a.shape[0]
    n = a.size // k
    m = b.size // k

    return _core.tensordot_core(a, b, None, n, m, k, ret_shape)


def outer(a, b, out=None):
    """Returns the outer product of two vectors.

    The input arrays are flattened into 1-D vectors and then it performs outer
    product of these vectors.

    Args:
        a (cupy.ndarray): The first argument.
        b (cupy.ndarray): The second argument.
        out (cupy.ndarray): Output array.

    Returns:
        cupy.ndarray: 2-D array of the outer product of ``a`` and ``b``.

    .. seealso:: :func:`numpy.outer`

    """
    return cupy.multiply(a.ravel()[:, None], b.ravel()[None, :], out=out)


def tensordot(a, b, axes=2):
    """Returns the tensor dot product of two arrays along specified axes.

    This is equivalent to compute dot product along the specified axes which
    are treated as one axis by reshaping.

    Args:
        a (cupy.ndarray): The first argument.
        b (cupy.ndarray): The second argument.
        axes:
            - If it is an integer, then ``axes`` axes at the last of ``a`` and
              the first of ``b`` are used.
            - If it is a pair of sequences of integers, then these two
              sequences specify the list of axes for ``a`` and ``b``. The
              corresponding axes are paired for sum-product.

    Returns:
        cupy.ndarray: The tensor dot product of ``a`` and ``b`` along the
        axes specified by ``axes``.

    .. seealso:: :func:`numpy.tensordot`

    """
    a_ndim = a.ndim
    b_ndim = b.ndim
    if a_ndim == 0 or b_ndim == 0:
        if axes != 0 and axes != ((), ()):
            raise ValueError('An input is zero-dim while axes has dimensions')
        return cupy.multiply(a, b)

    if isinstance(axes, collections.abc.Sequence):
        if len(axes) != 2:
            raise ValueError('Axes must consist of two arrays.')
        a_axes, b_axes = axes
        if numpy.isscalar(a_axes):
            a_axes = a_axes,
        if numpy.isscalar(b_axes):
            b_axes = b_axes,
    else:
        a_axes = tuple(range(a_ndim - axes, a_ndim))
        b_axes = tuple(range(axes))

    sum_ndim = len(a_axes)
    if sum_ndim != len(b_axes):
        raise ValueError('Axes length mismatch')

    for a_axis, b_axis in zip(a_axes, b_axes):
        if a.shape[a_axis] != b.shape[b_axis]:
            raise ValueError('Axis dimension mismatch')

    # Make the axes non-negative
    a = _move_axes_to_head(a, [axis % a_ndim for axis in a_axes])
    b = _move_axes_to_head(b, [axis % b_ndim for axis in b_axes])

    ret_shape = a.shape[sum_ndim:] + b.shape[sum_ndim:]

    k = internal.prod(a.shape[:sum_ndim])
    # Avoid division by zero: _core.tensordot_core returns zeros without
    # checking n, m consistency, thus allowing 0-length dimensions to work
    n = a.size // k if k != 0 else 0
    m = b.size // k if k != 0 else 0

    return _core.tensordot_core(a, b, None, n, m, k, ret_shape)


# TODO: rename `M` to `a`
def matrix_power(M, n):
    """Raise a square matrix to the (integer) power `n`.

    Args:
        M (~cupy.ndarray): Matrix to raise by power n.
        n (~int): Power to raise matrix to.

    Returns:
        ~cupy.ndarray: Output array.

    ..seealso:: :func:`numpy.linalg.matrix_power`
    """
    _util._assert_cupy_array(M)
    _util._assert_stacked_2d(M)
    _util._assert_stacked_square(M)
    if not isinstance(n, int):
        raise TypeError('exponent must be an integer')

    if n == 0:
        return _util.stacked_identity_like(M)
    elif n < 0:
        M = _solve.inv(M)
        n *= -1

    # short-cuts
    if n <= 3:
        if n == 1:
            return M
        elif n == 2:
            return cupy.matmul(M, M)
        else:
            return cupy.matmul(cupy.matmul(M, M), M)

    # binary decomposition to reduce the number of Matrix
    # multiplications for n > 3.
    result, Z = None, None
    for b in cupy.binary_repr(n)[::-1]:
        Z = M if Z is None else cupy.matmul(Z, Z)
        if b == '1':
            result = Z if result is None else cupy.matmul(result, Z)

    return result


def kron(a, b):
    """Returns the kronecker product of two arrays.

    Args:
        a (~cupy.ndarray): The first argument.
        b (~cupy.ndarray): The second argument.

    Returns:
        ~cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.kron`

    """
    a_isnumber = isinstance(a, numbers.Number)
    b_isnumber = isinstance(b, numbers.Number)
    if a_isnumber and b_isnumber:
        return a * b
    if a_isnumber or b_isnumber:
        return cupy.multiply(a, b)

    a_ndim = a.ndim
    b_ndim = b.ndim
    if a_ndim == 0 or b_ndim == 0:
        return cupy.multiply(a, b)

    ndim = b_ndim
    a_shape = a.shape
    b_shape = b.shape
    if a_ndim != b_ndim:
        if b_ndim > a_ndim:
            a_shape = (1,) * (b_ndim - a_ndim) + a_shape
        else:
            b_shape = (1,) * (a_ndim - b_ndim) + b_shape
            ndim = a_ndim

    axis = ndim - 1
    out = _core.tensordot_core(
        a, b, None, a.size, b.size, 1, a_shape + b_shape)
    for _ in range(ndim):
        out = _core.concatenate_method(out, axis=axis)

    return out


def _move_axes_to_head(a, axes):
    # This function moves the axes of ``s`` to the head of the shape.
    for idx, axis in enumerate(axes):
        if idx != axis:
            break
    else:
        return a

    return a.transpose(
        axes + [i for i in range(a.ndim) if i not in axes])


def multi_dot(arrays, *, out=None):
    """Compute the dot product of two or more arrays in a single function call,
    while automatically selecting the fastest evaluation order.

    `multi_dot` chains `cupy.dot` and uses optimal parenthesization
    of the matrices. Depending on the shapes of the matrices,
    this can speed up the multiplication a lot.

    If the first argument is 1-D it is treated as a row vector.
    If the last argument is 1-D it is treated as a column vector.
    The other arguments must be 2-D.

    Args:
        arrays (sequence of cupy.ndarray): Sequence of arrays to multiply.
        out (cupy.ndarray, optional): Output array.

    Returns:
        cupy.ndarray: The dot product of the arrays.

    .. seealso:: :func:`numpy.linalg.multi_dot`
    """
    n = len(arrays)
    if n < 2:
        raise ValueError('multi_dot requires at least two arrays')
    elif n == 2:
        return cupy.dot(arrays[0], arrays[1], out=out)

    arrays = [cupy.asanyarray(a) for a in arrays]

    # save original ndim to reshape the result array into the proper form later
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
    # Explicitly convert vectors to 2D arrays to keep the logic of the internal
    # _multi_dot_* functions as simple as possible.
    if arrays[0].ndim == 1:
        arrays[0] = cupy.atleast_2d(arrays[0])
    if arrays[-1].ndim == 1:
        arrays[-1] = cupy.atleast_2d(arrays[-1]).T
    _util._assert_2d(*arrays)

    # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot_recursive(arrays, order, 0, n - 1, out=out)

    # return proper shape
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]  # scalar
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()  # 1-D
    else:
        return result


def _multi_dot_three(A, B, C, out=None):
    """
    Find the best order for three arrays and do the multiplication.
    """
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return cupy.dot(cupy.dot(A, B), C, out=out)
    else:
        return cupy.dot(A, cupy.dot(B, C), out=out)


def _multi_dot_matrix_chain_order(arrays):
    """
    Return a numpy array that encodes the optimal order of multiplications.
    """
    n = len(arrays)
    # p stores the dimensions of the matrices
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    # m is a matrix of costs of the subproblems
    m = numpy.zeros((n, n), dtype=numpy.double)
    # s is the actual ordering
    s = numpy.empty((n, n), dtype=numpy.intp)

    for length in range(1, n):
        for i in range(n - length):
            j = i + length
            m[i, j] = numpy.inf
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k

    return s


def _multi_dot_recursive(arrays, order, i, j, out=None):
    """Actually do the multiplication with the given order."""
    if i == j:
        return arrays[i]
    else:
        return cupy.dot(_multi_dot_recursive(arrays, order, i, order[i, j]),
                        _multi_dot_recursive(
                            arrays, order, order[i, j] + 1, j),
                        out=out)
