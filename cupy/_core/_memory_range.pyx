from cupy._core.core cimport _ndarray_base
from cupy.cuda cimport memory

from libcpp.pair cimport pair


cdef void get_range(
        Py_ssize_t itemsize, shape_t& shape, strides_t & strides,
        Py_ssize_t& out_left, Py_ssize_t& out_right) noexcept:
    """Discover the byte range (out_left, out_right] (without ptr offset).
    """
    cdef Py_ssize_t tmp, i
    out_left = 0
    out_right = itemsize

    for i in range(shape.size()):
        if shape[i] == 0:  # empty just return 0 for both
            out_left = 0
            out_right = 0
            return

        tmp = (shape[i] - 1) * strides[i]
        if tmp > 0:
            out_right += tmp
        else:
            out_left += tmp


cdef _get_bound(_ndarray_base array, pair[Py_ssize_t, Py_ssize_t]& bounds):
    # C-version of get_bounds (really to avoid the need for `except *`...)
    cdef Py_ssize_t left, right
    get_range(array.dtype.itemsize, array._shape, array._strides, left, right)

    bounds.first = array.data.ptr + left
    bounds.second = array.data.ptr + right


def get_bound(_ndarray_base array):
    """Discover the pointer byte bounds (left, right] of the array.
    """
    cdef pair[Py_ssize_t, Py_ssize_t] bounds
    _get_bound(array, bounds)
    return bounds


cpdef bint may_share_bounds(_ndarray_base a, _ndarray_base b):
    cdef memory.MemoryPointer a_data = a.data
    cdef memory.MemoryPointer b_data = b.data
    cdef pair[Py_ssize_t, Py_ssize_t] a_range, b_range

    if (a_data.device_id != b_data.device_id
            or a_data.mem.ptr != b_data.mem.ptr
            or a.size == 0 or b.size == 0):
        return False

    _get_bound(a, a_range)
    _get_bound(b, b_range)

    return a_range.first < b_range.second and b_range.first < a_range.second
