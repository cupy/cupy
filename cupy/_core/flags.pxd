# distutils: language = c++


cdef class Flags:
    cdef readonly bint c_contiguous
    cdef readonly bint f_contiguous
    cdef readonly bint owndata

    @staticmethod
    cdef inline Flags _create(
        bint c_contiguous, bint f_contiguous, bint owndata
    ):
        cdef Flags self = <Flags>Flags.__new__(Flags)
        self.c_contiguous = c_contiguous
        self.f_contiguous = f_contiguous
        self.owndata = owndata
        return self
