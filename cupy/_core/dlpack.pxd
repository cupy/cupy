from cupy._core.core cimport ndarray

cpdef object toDlpack(ndarray array) except +
cpdef ndarray fromDlpack(object dltensor) except +
cpdef from_dlpack(array)

cpdef enum:
    device_CUDA = 2  # kDLCUDA
    device_ROCM = 10  # kDLROCM
