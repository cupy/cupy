# distutils: language = c++

from libc.stdint cimport intptr_t

from cupy.cuda.cufft cimport Handle, Result
from cupy.cuda.cufft import check_result


cdef extern from 'cufftXt.h' nogil:
    # cuFFT callback
    ctypedef enum callbackType 'cufftXtCallbackType':
        pass
    Result cufftXtSetCallback(Handle, void**, int, void**)
    ctypedef enum fft_type 'cufftType':
        pass
    Result cufftPlan1d(Handle*, int, fft_type, int)


cpdef setCallback(object plan, intptr_t callback, int cb_type):
    cdef Handle p
    cdef int result
    print(callback)

    with nogil:
        result = cufftPlan1d(&p, 128, <fft_type>0x29, 16)
    check_result(result)
    print("i am done")

    with nogil:
        result = cufftXtSetCallback(p, <void**>(&callback), <callbackType>cb_type, NULL)
    check_result(result)
