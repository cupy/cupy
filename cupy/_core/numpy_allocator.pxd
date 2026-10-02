# distutils: language = c++

cimport numpy as cnp

cdef array_uses_cupy_allocator(cnp.ndarray arr)
