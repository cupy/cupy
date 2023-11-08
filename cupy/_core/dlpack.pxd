from cupy._core.core cimport _ndarray_base


cdef extern from './include/cupy/_dlpack/dlpack.h' nogil:
    int device_CUDA 'kDLCUDA'
    int managed_CUDA 'kDLCUDAManaged'
    int device_ROCM 'kDLROCM'


cpdef object toDlpack(_ndarray_base array) except +
cpdef _ndarray_base fromDlpack(object dltensor) except +
cpdef from_dlpack(array)
