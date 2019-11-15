cdef extern from *:
    ctypedef float Float 'cufftReal'
    ctypedef double Double 'cufftDoubleReal'
    ctypedef int Result 'cufftResult_t'
    ctypedef int Handle 'cufftHandle'
    ctypedef int Type 'cufftType_t'


cpdef enum:
    CUFFT_C2C = 0x29
    CUFFT_R2C = 0x2a
    CUFFT_C2R = 0x2c
    CUFFT_Z2Z = 0x69
    CUFFT_D2Z = 0x6a
    CUFFT_Z2D = 0x6c

    CUFFT_FORWARD = -1
    CUFFT_INVERSE = 1


cpdef enum:
    CUFFT_XT_FORMAT_INPUT
    CUFFT_XT_FORMAT_OUTPUT
    CUFFT_XT_FORMAT_INPLACE
    CUFFT_XT_FORMAT_INPLACE_SHUFFLED


cpdef enum:
    # Actually, this is 64, but it's undocumented. For the sake
    # of safety, let us use 16, which agrees with the cuFFT doc.
    MAX_CUDA_DESCRIPTOR_GPUS = 16


cpdef get_current_plan()
