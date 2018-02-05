// This file is a stub header file of cufft for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUFFT_H
#define INCLUDE_GUARD_CUPY_CUFFT_H

#ifndef CUPY_NO_CUDA
#  include <cufft.h>

#else  // CUPY_NO_CUDA
extern "C" {

typedef float cufftReal;
typedef double cufftDoubleReal;

struct cufftComplex{
    float x, y;
};

struct cufftDoubleComplex{
    double x, y;
};

typedef enum {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE = 11,
    CUFFT_PARSE_ERROR = 12,
    CUFFT_NO_WORKSPACE = 13,
    CUFFT_NOT_IMPLEMENTED = 14,
    CUFFT_LICENSE_ERROR = 15,
    CUFFT_NOT_SUPPORTED = 16,
} cufftResult_t;

typedef int cufftHandle;

typedef enum {} cufftType_t;

// cuFFT Helper Function
cufftResult_t cufftCreate(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftDestroy(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftSetAutoAllocation(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftSetWorkArea(...) {
    return CUFFT_SUCCESS;
}

// cuFFT Stream Function
cufftResult_t cufftSetStream(...) {
    return CUFFT_SUCCESS;
}

// cuFFT Plan Function
cufftResult_t cufftMakePlan1d(...) {
    return CUFFT_SUCCESS;
}


// cuFFT Exec Function
cufftResult_t cufftExecC2C(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftExecR2C(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftExecC2R(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftExecZ2Z(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftExecD2Z(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftExecZ2D(...) {
    return CUFFT_SUCCESS;
}

}  // extern "C"

#endif  // CUPY_NO_CUDA

#endif  // INCLUDE_GUARD_CUPY_CUFFT_H
