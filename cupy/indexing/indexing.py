import numpy
import six

import cupy
from cupy import elementwise


def take(a, indices, axis=None, out=None, allocator=None):
    """Takes elements of an array at specified indices along an axis.

    This is an implementation of "fancy indexing" at single axis.

    This function does not support ``mode`` option.

    Args:
        a (cupy.ndarray): Array to extract elements.
        indices (int or array-like): Indices of elements that this function
            takes.
        axis (int): The axis along which to select indices. The flattened input
            is used by default.
        out (cupy.ndarray): Output array. If provided, it should be of
            appropriate shape and dtype.
        allocator (function): CuPy memory allocator. The allocator of ``a`` is
            used by default.

    Returns:
        cupy.ndarray: The result of fancy indexing.

    .. seealso:: :func:`numpy.take`

    """
    if axis is None:
        a = a.ravel()
        aarg = a
        lshape = ()
        rshape = ()
    else:
        if axis >= a.ndim:
            raise ValueError('Axis overrun')
        lshape = a.shape[:axis]
        rshape = a.shape[axis + 1:]
        aarg = a.reduced_view()

    if allocator is None:
        allocator = a.allocator

    if numpy.isscalar(indices):
        a = cupy.rollaxis(a, axis)
        if out is None:
            return a[indices].copy(allocator)
        else:
            out[:] = a[indices]
            return out
    elif not isinstance(indices, cupy.ndarray):
        indices = cupy.array(indices, dtype=int, allocator=allocator)

    out_shape = lshape + indices.shape + rshape
    if out is None:
        out = cupy.empty(out_shape, dtype=a.dtype, allocator=allocator)
        outarg = out.ravel()
    else:
        if out.dtype != a.dtype:
            raise TypeError('Output dtype mismatch')
        if out.shape != out_shape:
            raise ValueError('Output shape mismatch')
        outarg = out.reduced_view()

    cdim = indices.size
    rdim = numpy.prod(rshape, dtype=int)
    _take_kernel(outarg, aarg, indices.reduced_view(), cdim, rdim)
    return out


def choose(a, choices, out=None, mode='raise', allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def compress(condition, a, axis=None, out=None, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


def diagonal(a, offset=0, axis1=0, axis2=1):
    """Returns specified diagonals.

    This function extracts the diagonals along two specified axes. The other
    axes are not changed. This function returns a writable view of this array
    as NumPy 1.10 will do.

    Args:
        a (cupy.ndarray): Array from which the diagonals are taken.
        offset (int): Index of the diagonals. Zero indicates the main
            diagonals, positive value upper diagonals, and negative value lower
            diagonals.
        axis1 (int): The first axis to take diagonals from.
        axis2 (int): The second axis to take diagonals from.

    Returns:
        cupy.ndarray: A view of the diagonals of ``a``.

    .. seealso:: :func:`numpy.diagonal`

    """
    if axis1 < axis2:
        min_axis, max_axis = axis1, axis2
    else:
        min_axis, max_axis = axis2, axis1

    tr = list(six.moves.range(a.ndim))
    del tr[max_axis]
    del tr[min_axis]
    if offset >= 0:
        a = a.transpose(tr + [axis1, axis2])
    else:
        a = a.transpose(tr + [axis2, axis1])
        offset = -offset

    diag_size = max(0, min(a.shape[-2], a.shape[-1] - offset))
    ret_shape = a.shape[:-2] + (diag_size,)
    if diag_size == 0:
        return cupy.empty(ret_shape, dtype=a.dtype, allocator=a.allocator)

    a = a[..., :diag_size, offset:offset + diag_size]

    ret = a.view()
    ret._shape = a.shape[:-2] + (diag_size,)
    ret._strides = a.strides[:-2] + (a.strides[-1] + a.strides[-2],)
    ret._update_contiguity()
    return ret


def select(condlist, choicelist, default=0, allocator=None):
    # TODO(beam2d): Implement it
    raise NotImplementedError


_take_kernel = elementwise.ElementwiseKernel(
    ['out', 'a', 'indices', 'cdim', 'rdim'],
    '''
      long long li = i / (rdim * cdim);
      long long ci = indices[i / rdim % cdim];
      long long ri = i % rdim;
      out[i] = a[(li * cdim + ci) * rdim + ri];
    ''',
    'cupy_take')
