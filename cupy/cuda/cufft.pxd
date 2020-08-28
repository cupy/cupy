from libc.stdint cimport intptr_t

from cupy.cuda cimport memory


cdef extern from *:
    ctypedef float Float 'cufftReal'
    ctypedef double Double 'cufftDoubleReal'
    ctypedef int Result 'cufftResult_t'
    #ctypedef int Handle 'cufftHandle'

    # TODO(leofang): use a macro to split cuda/hip path
    ctypedef struct c_handle 'hipfftHandle_t':
        pass
    ctypedef c_handle* Handle 'cufftHandle'

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
        Handle plan
        object work_area  # can be MemoryPointer or a list of it
        readonly int nx
        readonly int batch
        readonly Type fft_type

        list gpus
        readonly bint _use_multi_gpus
        list batch_share
        list gather_streams
        list gather_events
        list scatter_streams
        list scatter_events
        intptr_t xtArr
        list xtArr_buffer

        void _single_gpu_get_plan(self, Handle plan, int nx, int fft_type, int batch) except*
        void _multi_gpu_get_plan(self, Handle plan, int nx, int fft_type, int batch, devices, out) except*

    #cpdef _single_gpu_fft(self, a, out, direction)


cdef class PlanNd:
    cdef:
        Handle plan
        memory.MemoryPointer work_area
        readonly tuple shape
        readonly Type fft_type
        readonly str order
        int last_axis
        object last_size

    #cpdef fft(self, a, out, direction)
