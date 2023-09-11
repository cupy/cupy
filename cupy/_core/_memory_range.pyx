from cupy._core.core cimport _ndarray_base
from cupy.cuda cimport memory

from libcpp.pair cimport pair


cpdef pair[Py_ssize_t, Py_ssize_t] get_bound(_ndarray_base array):
    cdef Py_ssize_t left = array.data.ptr
    cdef Py_ssize_t right = left
    cdef Py_ssize_t tmp
    cdef pair[Py_ssize_t, Py_ssize_t] ret
    cdef size_t i

    for i in range(array._shape.size()):
        # shape[i] != 0 is assumed
        tmp = (array._shape[i] - 1) * array._strides[i]
        if tmp > 0:
            right += tmp
        else:
            left += tmp

    ret.first = left
    ret.second = right + <Py_ssize_t>array.dtype.itemsize
    return ret


cpdef bint may_share_bounds(_ndarray_base a, _ndarray_base b):
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
