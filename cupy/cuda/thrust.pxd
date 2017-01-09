# distutils: language = c++

cdef extern from "../cuda/cupy_thrust.h" namespace "cupy::thrust":
    void sort[T](void *start, ssize_t num)
