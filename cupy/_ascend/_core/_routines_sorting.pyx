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

cdef _ascend_sort(_ndarray_base self, int axis):
    # inplace sort op
    cdef _ndarray_base out
    out = core.ndarray(self.shape, dtype=self.dtype)
    launch_general_func("ascend_sort", [self], [out], [], {"axis": axis, "mode": "stable"}, 0)
    elementwise_copy(out, self)

cdef _ascend_argsort(_ndarray_base self, _ndarray_base out, int axis):
    launch_general_func("ascend_argsort", [self], [out], [], {"axis": axis}, 0)

cdef _ascend_partition(_ndarray_base self, kth, int axis):
    print("Error: Ascend has no such aclop")

cdef _ascend_partitionsort(_ndarray_base self, kth, int axis):
    print("Error: Ascend has no such aclop")


cdef _ndarray_sort(_ndarray_base self, int axis):
    cdef int ndim = self._shape.size()
    cdef _ndarray_base data

    if ndim == 0:
        raise AxisError('Sorting arrays with the rank of zero is not '
                        'supported')  # as numpy.sort() raises

    # TODO(takagi): Support sorting views
    if not self._c_contiguous:
        raise NotImplementedError('Sorting non-contiguous array is not '
                                  'supported.')

    axis = internal._normalize_axis_index(axis, ndim)

    if axis == ndim - 1:
        data = self
    else:
        data = _manipulation.rollaxis(self, axis, ndim).copy()

    if ndim == 1:
        #thrust.sort(self.dtype, data.data.ptr, 0, self.shape)
        _ascend_sort(data, axis)
    else:
        """
        max_size = max(min(1 << 22, data.size) // data.shape[-1], 1)
        keys_array = core.ndarray(
            (max_size * data.shape[-1],), dtype=numpy.intp)
        stop = data.size // data.shape[-1]
        for offset in range(0, stop, max_size):
            width = min(max_size, stop - offset)
            
            thrust.sort(
                self.dtype,
                data.data.ptr + offset * data.shape[-1] * data.itemsize,
                keys_array.data.ptr,
                (width, data.shape[-1]),
            )
        """
        _ascend_sort(data, axis)

    if axis == ndim - 1:
        pass
    else:
        data = _manipulation.rollaxis(data, -1, axis)
        elementwise_copy(data, self)


cdef _ndarray_base _ndarray_argsort(_ndarray_base self, axis):
    cdef int _axis, ndim
    cdef _ndarray_base data

    if not cupy.xpu.thrust.available:
        raise RuntimeError('Thrust is needed to use cupy.argsort. Please '
                           'install CUDA Toolkit with Thrust then '
                           'reinstall CuPy after uninstalling it.')

    self = cupy.atleast_1d(self)
    ndim = self._shape.size()

    if axis is None:
        data = self.ravel()
        _axis = -1
    else:
        data = self
        _axis = axis

    _axis = internal._normalize_axis_index(_axis, ndim)

    if _axis == ndim - 1:
        data = data.copy()
    else:
        data = _manipulation.rollaxis(data, _axis, ndim).copy()
    shape = data.shape

    idx_array = core.ndarray(shape, dtype=numpy.intp)

    """
    if ndim == 1:
        thrust.argsort(self.dtype, idx_array.data.ptr, data.data.ptr, 0,
                       shape)
    else:
        keys_array = core.ndarray(shape, dtype=numpy.intp)
        thrust.argsort(self.dtype, idx_array.data.ptr, data.data.ptr,
                       keys_array.data.ptr, shape)
    """
    _ascend_argsort(data, idx_array, _axis)

    if _axis == ndim - 1:
        return idx_array
    else:
        return _manipulation.rollaxis(idx_array, -1, _axis)


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
    cdef Py_ssize_t k, max_k, length, s, sz, t
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
    max_k = 0
    for k in kth:
        if k < 0:
            k += length
        if not (0 <= k < length):
            raise ValueError('kth(={}) out of bounds {}'.format(k, length))
        if max_k < k:
            max_k = k

    # ASCEND: TODO later
    _ascend_partition(data, kth, axis)

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
    cdef Py_ssize_t k, max_k, length, s, sz, t
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
    max_k = 0
    for k in kth:
        if k < 0:
            k += length
        if not (0 <= k < length):
            raise ValueError('kth(={}) out of bounds {}'.format(k, length))
        if max_k < k:
            max_k = k

    shape = data.shape
    data = data.ravel()
    indices = cupy.arange(0, data.shape[0], dtype=cupy.int64)
    # TODO: create output ndarray
    #_ascend_partitionsort(data, indices, kth, _axis)

    # Rearrange indices w.r.t the original axis
    axis_indices = cupy.unravel_index(indices, shape)
    indices = axis_indices[-1]
    indices = indices.reshape(shape)

    if _axis != ndim - 1:
        indices = _manipulation.rollaxis(indices, -1, _axis)

    return indices
