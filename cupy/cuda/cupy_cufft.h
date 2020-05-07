// This file is a stub header file of cufft for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUFFT_H
#define INCLUDE_GUARD_CUPY_CUFFT_H

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)
#include <cufft.h>
#include <cufftXt.h>

#else  // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#include "cupy_cuda.h"

extern "C" {

typedef float cufftReal;
typedef double cufftDoubleReal;
typedef cuComplex cufftComplex;
typedef cuDoubleComplex cufftDoubleComplex;

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

// cuFFT Plan Functions
cufftResult_t cufftMakePlan1d(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftMakePlanMany(...) {
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

// cuFFT Version
cufftResult_t cufftGetVersion(...) {
    return CUFFT_SUCCESS;
}

// cufftXt relavant data and functions
typedef struct cudaXtDesc_t {
   int version;
   int nGPUs;
   int GPUs[64];
   void* data[64];
   size_t size[64];
   void* cudaXtState;
} cudaXtDesc;

typedef enum cufftXtSubFormat_t {
    CUFFT_XT_FORMAT_INPUT = 0x00,
    CUFFT_XT_FORMAT_OUTPUT = 0x01,
    CUFFT_XT_FORMAT_INPLACE = 0x02,
    CUFFT_XT_FORMAT_INPLACE_SHUFFLED = 0x03,
    CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED = 0x04,
    CUFFT_FORMAT_UNDEFINED = 0x05
} cufftXtSubFormat;

typedef struct cudaLibXtDesc_t{
    int version;
    cudaXtDesc *descriptor;
    int library;  // libFormat is an undoumented type, so use int here
    int subFormat;
    void *libDescriptor;
} cudaLibXtDesc;

typedef enum cufftXtCopyType_t {
    CUFFT_COPY_HOST_TO_DEVICE = 0x00,
    CUFFT_COPY_DEVICE_TO_HOST = 0x01,
    CUFFT_COPY_DEVICE_TO_DEVICE = 0x02,
    CUFFT_COPY_UNDEFINED = 0x03
} cufftXtCopyType;

cufftResult_t cufftXtSetGPUs(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftXtSetWorkArea(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftXtMemcpy(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftXtExecDescriptorC2C(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftXtExecDescriptorZ2Z(...) {
    return CUFFT_SUCCESS;
}

}  // extern "C"

#endif  // #if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)

#endif  // INCLUDE_GUARD_CUPY_CUFFT_H
