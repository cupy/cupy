import collections.abc

import numpy

import cupy
from cupy import _core
from cupy._core import internal
from cupy._core._gufuncs import _GUFunc
from cupy.linalg import _solve
from cupy.linalg import _util


matmul = _GUFunc(
    _core.matmul,
    '(n?,k),(k,m?)->(n?,m?)',
    supports_batched=True,
    supports_out=True,
    doc="""matmul(x1, x2, /, out=None, \\*\\*kwargs)

    Matrix product of two arrays.

    Returns the matrix product of two arrays and is the implementation of
    the `@` operator introduced in Python 3.5 following PEP465.

    The main difference against cupy.dot are the handling of arrays with more
    than 2 dimensions. For more information see :func:`numpy.matmul`.

    .. note::
        The out array as input is currently not supported.

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


def _multi_dot_three(A, B, C, out=None):
    """Find the best order for three arrays and do the multiplication.
    """
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return dot(dot(A, B), C, out=out)
    else:
        return dot(A, dot(B, C), out=out)


def _multi_dot_matrix_chain_order(arr, return_costs=False):
    """Returns a cupy.ndarray that encodes the optimal order of
    multiplication. The optimal order array is then used by
    `_multi_dot()` to do the multiplication.

    It also return the cost matrix if `return_costs` is `True`

    The cost function is given by:

        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])
    """
    n = len(arr)
    # To store dimensions of the matrices
    p = [a.shape[0] for a in arr] + [arr[-1].shape[1]]
    # Matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = cupy.zeros((n, n), dtype=cupy.double)
    # Actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = cupy.empty((n, n), dtype=cupy.intp)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = cupy.Inf
            for k in range(i, j):
                q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k

    return (s, m) if return_costs else s


def _multi_dot(arr, order, i, j, out=None):
    """Do the multiplication with the given order."""
    i = int(i)
    j = int(j)
    if i == j:
        assert out is None
        return arr[i]
    else:
        return cupy.dot(_multi_dot(arr, order, i, order[i, j]),
                        _multi_dot(arr, order, order[i, j] + 1, j),
                        out=out)


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


def matrix_power(a, n):
    """Raise a square matrix to the (integer) power `n`.

    Args:
        a (~cupy.ndarray): Matrix to raise by power n.
        n (~int): Power to raise matrix to.

    Returns:
        ~cupy.ndarray: Output array.

    ..seealso:: :func:`numpy.linalg.matrix_power`
    """
    _util._assert_cupy_array(a)
    _util._assert_stacked_2d(a)
    _util._assert_stacked_square(a)
    if not isinstance(n, int):
        raise TypeError('exponent must be an integer')

    if n == 0:
        return _util.stacked_identity_like(a)
    elif n < 0:
        a = _solve.inv(a)
        n *= -1

    # short-cuts
    if n <= 3:
        if n == 1:
            return a
        elif n == 2:
            return cupy.matmul(a, a)
        else:
            return cupy.matmul(cupy.matmul(a, a), a)

    # binary decomposition to reduce the number of Matrix
    # multiplications for n > 3.
    result, Z = None, None
    for b in cupy.binary_repr(n)[::-1]:
        Z = a if Z is None else cupy.matmul(Z, Z)
        if b == '1':
            result = Z if result is None else cupy.matmul(result, Z)

    return result


def multi_dot(arr, *, out=None):
    """Compute the dot product of two or more arrays in a single function
    call, and automatically select the fastest evaluation order.

    Note: `multi_dot` chains `cupy.dot` and uses optimal parenthesization
    of the matrices. Depending on the shape of the matrices, this can
    speed up the multiplication a lot.

    Parameters
    ----------
    arr : cupy.ndarray
        If the first argument is 1-D vector of shape (n,) it is
        treated as row vector of shape (1, n). If the last argument is
        1-D vector of shape (n,) it is treated as column vector of shape
        (n, 1). The other arguments must be 2-D.
    out : cupy.ndarray, optional
        The output argument. It must have the exact kind that would be
        returned if it was not used. It must have the right type, must be
        C-contiguous. Its dtype must be similar to the dtype returned for
        `dot(a, b)`.

    Returns
    -------
    output : cupy.ndarray
        Returns the dot product of the given arrays.

    See Also
    --------
    numpy.multi_dot

    """
    n = len(arr)
    # optimization only makes sense for len(arr) > 2
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return cupy.dot(arr[0], arr[1], out=out)

    # save original ndim to reshape the result array into the proper form later
    ndim_first, ndim_last = arr[0].ndim, arr[-1].ndim
    # Convert vectors to 2D arrays
    if arr[0].ndim == 1:
        arr[0] = cupy.atleast_2d(arr[0])
    if arr[-1].ndim == 1:
        arr[-1] = cupy.atleast_2d(arr[-1]).T
    _util._assert_2d(*arr)

    # _multi_dot_three is much faster than _multi_dot_matrix_chain_orde
    if n == 3:
        result = _multi_dot_three(arr[0], arr[1], arr[2], out=out)
    else:
        order = _multi_dot_matrix_chain_order(arr)
        result = _multi_dot(arr, order, 0, n - 1, out=out)

    # return proper shape
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()
    else:
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
