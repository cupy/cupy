# Note: nothing exposed in this pxd header is considered public API;
# we copy this to sdist only because it's needed at runtime to support
# cuFFT callbacks

from libc.stdint cimport intptr_t

cdef extern from *:
    ctypedef float Float 'cufftReal'
    ctypedef double Double 'cufftDoubleReal'
    ctypedef int Result 'cufftResult_t'

    IF HIP_VERSION > 0:
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

    CUFFT_CB_LD_COMPLEX = 0x0,
    CUFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    CUFFT_CB_LD_REAL = 0x2,
    CUFFT_CB_LD_REAL_DOUBLE = 0x3,
    CUFFT_CB_ST_COMPLEX = 0x4,
    CUFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    CUFFT_CB_ST_REAL = 0x6,
    CUFFT_CB_ST_REAL_DOUBLE = 0x7,


cpdef get_current_plan()
cpdef int getVersion() except? -1


cdef class Plan1d:
    cdef:
        readonly intptr_t handle
        readonly object work_area  # can be MemoryPointer or a list of it
        readonly int nx
        readonly int batch
        readonly Type fft_type

        readonly list gpus
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
        readonly intptr_t handle
        readonly object work_area  # memory.MemoryPointer
        readonly tuple shape
        readonly Type fft_type
        readonly str order
        readonly int last_axis
        readonly object last_size

        # TODO(leofang): support multi-GPU transforms
        readonly list gpus


cdef class XtPlanNd:
    cdef:
        readonly intptr_t handle
        readonly object work_area  # memory.MemoryPointer
        readonly tuple shape
        readonly int itype
        readonly int otype
        readonly int etype
        readonly str order
        readonly int last_axis
        readonly object last_size

        # TODO(leofang): support multi-GPU transforms
        readonly list gpus
