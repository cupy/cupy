###############################################################################
# Types
###############################################################################

cdef extern from *:
    ctypedef int Ordering 'curandOrdering_t'
    ctypedef int RngType 'curandRngType_t'

    ctypedef void* Generator 'curandGenerator_t'


###############################################################################
# Enum
###############################################################################
IF CUPY_HIP_VERSION > 0:
    cpdef enum:
        CURAND_RNG_PSEUDO_DEFAULT = 400
        CURAND_RNG_PSEUDO_XORWOW = 401
        CURAND_RNG_PSEUDO_MRG32K3A = 402
        CURAND_RNG_PSEUDO_MTGP32 = 403
        CURAND_RNG_PSEUDO_MT19937 = 404
        CURAND_RNG_PSEUDO_PHILOX4_32_10 = 405
        CURAND_RNG_QUASI_DEFAULT = 500
        CURAND_RNG_QUASI_SOBOL32 = 501
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 502
        CURAND_RNG_QUASI_SOBOL64 = 503
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 504
ELSE:
    cpdef enum:
        CURAND_RNG_PSEUDO_DEFAULT = 100
        CURAND_RNG_PSEUDO_XORWOW = 101
        CURAND_RNG_PSEUDO_MRG32K3A = 121
        CURAND_RNG_PSEUDO_MTGP32 = 141
        CURAND_RNG_PSEUDO_MT19937 = 142
        CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161
        CURAND_RNG_QUASI_DEFAULT = 200
        CURAND_RNG_QUASI_SOBOL32 = 201
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202
        CURAND_RNG_QUASI_SOBOL64 = 203
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
        CURAND_ORDERING_PSEUDO_BEST = 100
        CURAND_ORDERING_PSEUDO_DEFAULT = 101
        CURAND_ORDERING_PSEUDO_SEEDED = 102
        CURAND_ORDERING_QUASI_DEFAULT = 201
