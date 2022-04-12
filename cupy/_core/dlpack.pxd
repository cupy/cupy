from cupy._core.core cimport ndarray


cdef extern from './include/cupy/dlpack/dlpack.h' nogil:
    int device_CUDA 'kDLCUDA'
    int managed_CUDA 'kDLCUDAManaged'
    int device_ROCM 'kDLROCM'


cpdef object toDlpack(ndarray array) except +
cpdef ndarray fromDlpack(object dltensor) except +
cpdef from_dlpack(array)
