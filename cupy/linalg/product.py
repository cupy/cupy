import numpy
import six

import cupy
from cupy import core
from cupy import internal

from cupy.linalg.solve import inv
from cupy.util import collections_abc


matmul = core.matmul


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

    return core.tensordot_core(a, b, None, 1, 1, a.size, ())


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

    return core.tensordot_core(a, b, None, n, m, k, ret_shape)


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
    n = a.size
    m = b.size
    ret_shape = (n, m)

    if out is None:
        return core.tensordot_core(a, b, None, n, m, 1, ret_shape)

    if out.size != n * m:
        raise ValueError('Output array has an invalid size')
    if out.flags.c_contiguous:
        return core.tensordot_core(a, b, out, n, m, 1, ret_shape)
    else:
        out[:] = core.tensordot_core(a, b, None, n, m, 1, ret_shape)
        return out


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

    if isinstance(axes, collections_abc.Sequence):
        if len(axes) != 2:
            raise ValueError('Axes must consist of two arrays.')
        a_axes, b_axes = axes
        if numpy.isscalar(a_axes):
            a_axes = a_axes,
        if numpy.isscalar(b_axes):
            b_axes = b_axes,
    else:
        a_axes = tuple(six.moves.range(a_ndim - axes, a_ndim))
        b_axes = tuple(six.moves.range(axes))

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
    n = a.size // k
    m = b.size // k

    return core.tensordot_core(a, b, None, n, m, k, ret_shape)


def matrix_power(M, n):
    """Raise a square matrix to the (integer) power `n`.

    Args:
        M (~cupy.ndarray): Matrix to raise by power n.
        n (~int): Power to raise matrix to.

    Returns:
        ~cupy.ndarray: Output array.

    .. note:: M must be of dtype `float32` or `float64`.

    ..seealso:: :func:`numpy.linalg.matrix_power`
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError('input must be a square array')
    if not isinstance(n, six.integer_types):
        raise TypeError('exponent must be an integer')

    if n == 0:
        return cupy.identity(M.shape[0], dtype=M.dtype)
    elif n < 0:
        M = inv(M)
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
    out = core.tensordot_core(a, b, None, a.size, b.size, 1, a_shape + b_shape)
    for _ in six.moves.range(ndim):
        out = core.concatenate_method(out, axis=axis)

    return out


def _move_axes_to_head(a, axes):
    # This function moves the axes of ``s`` to the head of the shape.
    for idx, axis in enumerate(axes):
        if idx != axis:
            break
    else:
        return a

    return a.transpose(
        axes + [i for i in six.moves.range(a.ndim) if i not in axes])
