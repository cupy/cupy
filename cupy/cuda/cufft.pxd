from libc.stdint cimport intptr_t

from cupy.cuda cimport memory


cdef extern from *:
    ctypedef float Float 'cufftReal'
    ctypedef double Double 'cufftDoubleReal'
    ctypedef int Result 'cufftResult_t'

    IF not use_hip:
        ctypedef int Handle 'cufftHandle'
    ELSE:
        ctypedef struct hipHandle 'hipfftHandle_t':
            pass
        ctypedef hipHandle* Handle 'cufftHandle'

    ctypedef enum Type 'cufftType_t':
        pass


cpdef enum:
    CUFFT_C2C = 0x29
    CUFFT_R2C = 0x2a
    CUFFT_C2R = 0x2c
    CUFFT_Z2Z = 0x69
    CUFFT_D2Z = 0x6a
    CUFFT_Z2D = 0x6c

    CUFFT_FORWARD = -1
    CUFFT_INVERSE = 1


cpdef get_current_plan()


cdef class Plan1d:
    cdef:
        intptr_t handle
        object work_area  # can be MemoryPointer or a list of it
        readonly int nx
        readonly int batch
        readonly Type fft_type

        list gpus
        readonly bint _use_multi_gpus
        list batch_share
        list gather_streams
        list gather_events
        dict scatter_streams
        dict scatter_events
        intptr_t xtArr
        list xtArr_buffer

        void _single_gpu_get_plan(
            self, Handle plan, int nx, int fft_type, int batch) except*
        void _multi_gpu_get_plan(
            self, Handle plan, int nx, int fft_type, int batch,
            devices, out) except*


cdef class PlanNd:
    cdef:
        intptr_t handle
        memory.MemoryPointer work_area
        readonly tuple shape
        readonly Type fft_type
        readonly str order
        readonly int last_axis
        readonly object last_size
