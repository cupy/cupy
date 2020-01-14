from cupy.core.core cimport ndarray


cpdef ndarray array(
    obj, dtype=*, bint copy=*, order=*, bint subok=*, Py_ssize_t ndmin=*)
cpdef ndarray internal_ascontiguousarray(ndarray a)
cpdef ndarray internal_asfortranarray(ndarray a)
cpdef ndarray ascontiguousarray(ndarray a, dtype=*)
cpdef ndarray asfortranarray(ndarray a, dtype=*)
