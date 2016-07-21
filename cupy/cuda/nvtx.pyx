"""
Wrapper for NVIDIA Tools Extension Library (NVTX)

"""
cimport cython

cdef extern from "cupy_cuda.h":
    int nvtxMarkA(const char *message)
    int nvtxRangePushA(const char *message)
    int nvtxRangePop()

cpdef void Mark(str message) except *:
    cdef bytes b_message = message.encode()
    nvtxMarkA(<const char*>b_message)

cpdef void RangePush(str message) except *:
    cdef bytes b_message = message.encode()
    ret = nvtxRangePushA(<const char*>b_message)
    # assert ret >= 0, 'nvtxRangePushA(): {}'.format(ret)

cpdef void RangePop() except *:
    ret = nvtxRangePop()
    # assert ret >= 0, 'nvtxRangePop(): {}'.format(ret)
