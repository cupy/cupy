from cupy._core.core cimport ndarray
from cupy.cuda cimport memory

from libc.stdint cimport intptr_t
from libcpp.pair cimport pair
from libcpp.vector cimport vector


cpdef pair[Py_ssize_t, Py_ssize_t] get_bound(ndarray array):
    cdef Py_ssize_t left = array.data.ptr
    cdef Py_ssize_t right = left
    cdef pair[Py_ssize_t, Py_ssize_t] ret
    cdef size_t i

    for i in range(array._shape.size()):
        right += (array._shape[i] - 1) * array._strides[i]

    if left > right:
        left, right = right, left

    ret.first = left
    ret.second = right + <Py_ssize_t>array.dtype.itemsize
    return ret


cpdef bint may_share_bounds(ndarray a, ndarray b):
    cdef memory.MemoryPointer a_data = a.data
    cdef memory.MemoryPointer b_data = b.data
    cdef pair[Py_ssize_t, Py_ssize_t] a_range, b_range

    if (a_data.device_id != b_data.device_id
            or a_data.mem.ptr != b_data.mem.ptr
            or a.size == 0 or b.size == 0):
        return False

    a_range = get_bound(a)
    b_range = get_bound(b)

    return a_range.first < b_range.second and b_range.first < a_range.second
