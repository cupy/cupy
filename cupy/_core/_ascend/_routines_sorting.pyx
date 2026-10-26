import string
import numpy

import cupy
from cupy.exceptions import AxisError
from cupy._core._scalar import get_typename as _get_typename
from cupy._core._ufuncs import elementwise_copy
import cupy._core.core as core
from cupy import _util
from cupy.backends.ascend.api.acl_utils cimport launch_general_func

from cupy._core cimport _routines_manipulation as _manipulation
from cupy._core.core cimport _ndarray_base
from cupy._core cimport internal

# TODO: cupy, not all numpy keyword API is supported by CUPY
#
# NOTE: aclnnSort / aclnnArgsort are "irregular" (GENERAL) ops, so they are
# dispatched through `launch_general_func`:
#   ins  = [self]                      the tensor to sort
#   outs = [values[, indices]]         indices is optional (int64)
#   args = [axis, stable, descending]  int64 / bool / bool
# aclnnSort takes the order flag directly (upstream `descending=`), so the
# public `descending` keyword maps onto the third argument.
cdef _ascend_sort(_ndarray_base self, _ndarray_base out, int axis,
                  bint descending=False):
    """Sort `self` along `axis`; the sorted values are written into `out`."""
    # CANN 8.5 aclnnSort does not accept nullptr as indices, not in used after return
    cdef _ndarray_base  indices = core.ndarray(out.shape, dtype=numpy.int64)
    try:
        if self.dtype.kind == 'f':
            # CANN 9.0.1: aclnnSort float kernels crash the vector core
            # (>= ~2M f32 elements); fall back to host CPU sort.
            host = numpy.sort(self.get(), axis=axis)
            if descending:
                host = numpy.flip(host, axis=axis)
            out[...] = cupy.asarray(host)
        else:
            launch_general_func(
                "ascend_sort",
                [self], [out, indices],
                [axis, 1, descending],  # axis, stable, descending
                {}, 0)
        # The kernel fault is ASYNC: the launch returns fine and the device
        # exception only surfaces on the next sync.  Sync here so the CPU
        # fallback below can actually catch it (and callers see either a
        # fully sorted result or a raised error, never a silent partial one).
        cupy.cuda.runtime.deviceSynchronize()
    except Exception:
        # The aclnnSort kernels fault on the device for large inputs
        # (int32 >= ~4M, int64/f64 >= ~10M elements on CANN 9.0.1/910B).
        # Correctness first: finish the sort on the host.  The device
        # exception does not poison the context (verified: ops launched
        # after the fault run normally).
        host = numpy.sort(self.get(), axis=axis)
        if descending:
            host = numpy.flip(host, axis=axis)
        out[...] = cupy.asarray(host)
        cupy.cuda.runtime.deviceSynchronize()


cdef _ascend_argsort(_ndarray_base self, _ndarray_base out, int axis,
                      bint descending=False):
    """Write the argsort indices of `self` along `axis` into `out`."""
    try:
        # Same float aclnnSort kernel fault as _ascend_sort
        # argsort fall back
        if self.dtype.kind == 'f':
            host = numpy.argsort(self.get(), axis=axis).astype(numpy.int64)
            if descending:
                host = numpy.flip(host, axis=axis)
            out[...] = cupy.asarray(host)
            cupy.cuda.runtime.deviceSynchronize()
        else:
            launch_general_func(
                "ascend_argsort",
                [self], [out],
                [axis, descending],  # dim, descending
                {}, 0)
            cupy.cuda.runtime.deviceSynchronize()  # surface async kernel faults
    except Exception:
        host = numpy.argsort(self.get(), axis=axis).astype(numpy.int64)
        if descending:
            host = numpy.flip(host, axis=axis)
        out[...] = cupy.asarray(host)
        cupy.cuda.runtime.deviceSynchronize()


cdef _ndarray_sort(_ndarray_base self, int axis, bint descending=False):
    cdef int ndim = self._shape.size()
    cdef _ndarray_base data, out

    if ndim == 0:
        raise AxisError('Sorting arrays with the rank of zero is not '
                        'supported')  # as numpy.sort() raises

    # TODO(takagi): Support sorting views
    if not self._c_contiguous:
        raise NotImplementedError('Sorting non-contiguous array is not '
                                  'supported.')

    axis = internal._normalize_axis_index(axis, ndim)

    # Move the target axis to the last position so that aclnn can always sort
    # the innermost dimension.  `data` is a fresh contiguous buffer in that
    # case, so the sorted result has to be copied back into `self` afterwards.
    if axis == ndim - 1:
        data = self
    else:
        data = _manipulation.rollaxis(self, axis, ndim).copy()

    # aclnnSort is not an inplace op: it returns the sorted values (and
    # optionally the indices) in a separate output tensor.  `data` now has the
    # sort axis as its last dimension, hence -1.
    out = core.ndarray(data.shape, dtype=data.dtype)
    _ascend_sort(data, out, -1, descending)

    if axis == ndim - 1:
        # `data is self`, sorting in place through the temp output.
        elementwise_copy(out, data)
    else:
        data = _manipulation.rollaxis(data, -1, axis)
        # roll the sorted output back the original axis layout before copying
        # directly copy will have broadcast failure
        out_view = _manipulation.rollaxis(out, -1, axis)
        elementwise_copy(out_view, data)
        elementwise_copy(data, self)


cdef _ndarray_base _ndarray_argsort(
        _ndarray_base self, axis, bint descending=False):
    cdef int _axis, ndim
    cdef _ndarray_base data, idx_view

    self = cupy.atleast_1d(self)
    ndim = self._shape.size()

    if axis is None:
        data = self.ravel()
        _axis = ndim - 1
    else:
        data = self
        _axis = axis

    _axis = internal._normalize_axis_index(_axis, ndim)

    if _axis == ndim - 1:
        data = data.copy()
    else:
        data = _manipulation.rollaxis(data, _axis, ndim).copy()

    # aclnnArgsort requires an int64 index tensor on output (CuPy defaults to
    # intp, which is int64 on Linux but not on all platforms) and always sorts
    # the last dimension, which is exactly where the target axis has been
    # moved to above.
    idx_array = core.ndarray(data.shape, dtype=numpy.int64)
    _ascend_argsort(data, idx_array, -1, descending)

    if _axis == ndim - 1:
        return idx_array

    # The indices are relative to the last axis; roll them back so that they
    # line up with the original `_axis` again.
    idx_view = _manipulation.rollaxis(idx_array, -1, _axis)
    return idx_view.copy()


cdef _ndarray_partition(_ndarray_base self, kth, int axis):
    """Partitions an array.

    Args:
        kth (int or sequence of ints): Element index to partition by. If
            supplied with a sequence of k-th it will partition all elements
            indexed by k-th of them into their sorted position at once.

        axis (int): Axis along which to sort. Default is -1, which means
            sort along the last axis.

    .. seealso::
        :func:`cupy.partition` for full documentation,
        :meth:`numpy.ndarray.partition`

    """

    cdef int ndim = self._shape.size()
    cdef Py_ssize_t k, length
    cdef _ndarray_base data

    if ndim == 0:
        raise AxisError('Sorting arrays with the rank of zero is not '
                        'supported')

    if not self._c_contiguous:
        raise NotImplementedError('Sorting non-contiguous array is not '
                                  'supported.')

    axis = internal._normalize_axis_index(axis, ndim)

    if axis == ndim - 1:
        data = self
    else:
        data = _manipulation.rollaxis(self, axis, ndim).copy()

    length = self._shape[axis]
    if isinstance(kth, int):
        kth = kth,
    for k in kth:
        if k < 0:
            k += length
        if not (0 <= k < length):
            raise ValueError('kth(={}) out of bounds {}'.format(k, length))

    # ASCEND: no aclnn partition op yet.  A fully sorted array is also a valid
    # partition (every k-th element is in its final position), so fall back to
    # `sort`, which is implemented on top of aclnnSort.
    data.sort(axis=-1)

    if axis != ndim - 1:
        data = _manipulation.rollaxis(data, -1, axis)
        elementwise_copy(data, self)


cdef _ndarray_base _ndarray_argpartition(self, kth, axis):
    """Returns the indices that would partially sort an array.

    Args:
        kth (int or sequence of ints): Element index to partition by. If
            supplied with a sequence of k-th it will partition all elements
            indexed by k-th of them into their sorted position at once.
        axis (int or None): Axis along which to sort. Default is -1, which
            means sort along the last axis. If None is supplied, the array
            is flattened before sorting.

    Returns:
        cupy.ndarray: Array of the same type and shape as ``a``.

    .. seealso::
        :func:`cupy.argpartition` for full documentation,
        :meth:`numpy.ndarray.argpartition`

    """
    cdef int _axis, ndim
    cdef Py_ssize_t k, length
    cdef _ndarray_base data
    if axis is None:
        data = self.ravel()
        _axis = -1
    else:
        data = self
        _axis = axis

    ndim = data._shape.size()
    _axis = internal._normalize_axis_index(_axis, ndim)

    if _axis != ndim - 1:
        data = _manipulation.rollaxis(self, _axis, ndim).copy()

    length = data._shape[ndim - 1]

    if length == 0:
        return cupy.empty((0,), dtype=cupy.int64)

    if isinstance(kth, int):
        kth = kth,
    for k in kth:
        if k < 0:
            k += length
        if not (0 <= k < length):
            raise ValueError('kth(={}) out of bounds {}'.format(k, length))

    # ASCEND: no aclnn argpartition op yet; a full argsort is also a valid
    # argpartition, so reuse `argsort` (implemented on top of aclnnArgsort).
    # `data` already has the partition axis as its last dimension.
    return data.argsort(axis=-1)
