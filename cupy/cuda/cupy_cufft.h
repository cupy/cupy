#ifndef INCLUDE_GUARD_CUPY_CUFFT_H
#define INCLUDE_GUARD_CUPY_CUFFT_H

/*
 * Note: this file should *not* be split into 3 and moved under cupy_backends/,
 * because we need to copy this header to sdist and use it at runtime for cuFFT
 * callbacks.
 */

#if !defined(CUPY_NO_CUDA) && !defined(CUPY_USE_HIP)
#include <cufft.h>
#include <cufftXt.h>

#elif defined(CUPY_USE_HIP)
#include <hipfft.h>

extern "C" {

typedef hipfftComplex cufftComplex;
typedef hipfftDoubleComplex cufftDoubleComplex;
typedef hipfftReal cufftReal;
typedef hipfftDoubleReal cufftDoubleReal;

typedef hipfftResult_t cufftResult_t;
typedef hipfftHandle cufftHandle;
typedef hipfftType_t cufftType_t;
typedef hipStream_t cudaStream_t;

// cuFFT Helper Function
cufftResult_t cufftCreate(cufftHandle* plan) {
    return hipfftCreate(plan);
}

cufftResult_t cufftDestroy(cufftHandle plan) {
    return hipfftDestroy(plan);
}

cufftResult_t cufftSetAutoAllocation(cufftHandle plan, int autoAllocate) {
    return hipfftSetAutoAllocation(plan, autoAllocate);
}

cufftResult_t cufftSetWorkArea(cufftHandle plan, void *workArea) {
    return hipfftSetWorkArea(plan, workArea);
}

// cuFFT Stream Function
cufftResult_t cufftSetStream(cufftHandle plan, cudaStream_t stream) {
    return hipfftSetStream(plan, stream);
}

// cuFFT Plan Functions
cufftResult_t cufftMakePlan1d(cufftHandle plan,
                              int nx,
                              cufftType_t type,
                              int batch,
                              size_t *workSize) {
    return hipfftMakePlan1d(plan, nx, type, batch, workSize);
}

cufftResult_t cufftMakePlanMany(cufftHandle plan,
                                int rank,
                                int *n,
                                int *inembed, int istride, int idist,
                                int *onembed, int ostride, int odist,
                                cufftType_t type,
                                int batch,
                                size_t *workSize) {
    return hipfftMakePlanMany(plan, rank, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              type, batch, workSize);
}

// cuFFT Exec Function
cufftResult_t cufftExecC2C(cufftHandle plan,
                           cufftComplex *idata,
                           cufftComplex *odata,
                           int direction) {
    return hipfftExecC2C(plan, idata, odata, direction);
}

cufftResult_t cufftExecR2C(cufftHandle plan,
                           cufftReal *idata,
                           cufftComplex *odata) {
    return hipfftExecR2C(plan, idata, odata);
}

cufftResult_t cufftExecC2R(cufftHandle plan,
                           cufftComplex *idata,
                           cufftReal *odata) {
    return hipfftExecC2R(plan, idata, odata);
}

cufftResult_t cufftExecZ2Z(cufftHandle plan,
                           cufftDoubleComplex *idata,
                           cufftDoubleComplex *odata,
                           int direction) {
    return hipfftExecZ2Z(plan, idata, odata, direction);
}

cufftResult_t cufftExecD2Z(cufftHandle plan,
                           cufftDoubleReal *idata,
                           cufftDoubleComplex *odata) {
    return hipfftExecD2Z(plan, idata, odata);
}

cufftResult_t cufftExecZ2D(cufftHandle plan,
                           cufftDoubleComplex *idata,
                           cufftDoubleReal *odata) {
    return hipfftExecZ2D(plan, idata, odata);
}

// cuFFT Version
cufftResult_t cufftGetVersion(int *version) {
    return hipfftGetVersion(version);
}

// TODO(leofang): move this header to cupy_backends/ and include hip/cupy_hip_common.h
typedef enum {} cudaDataType;

// cufftXt functions
cufftResult_t cufftXtSetGPUs(...) {
    return HIPFFT_NOT_IMPLEMENTED;
}

cufftResult_t cufftXtSetWorkArea(...) {
    return HIPFFT_NOT_IMPLEMENTED;
}

cufftResult_t cufftXtMemcpy(...) {
    return HIPFFT_NOT_IMPLEMENTED;
}

cufftResult_t cufftXtMakePlanMany(...) {
    return HIPFFT_NOT_IMPLEMENTED;
}

cufftResult_t cufftXtExec(...) {
    return HIPFFT_NOT_IMPLEMENTED;
}

cufftResult_t cufftXtExecDescriptorC2C(...) {
    return HIPFFT_NOT_IMPLEMENTED;
}

cufftResult_t cufftXtExecDescriptorZ2Z(...) {
    return HIPFFT_NOT_IMPLEMENTED;
}

} // extern "C"

#else  // defined(CUPY_NO_CUDA)

#include "../../cupy_backends/stub/cupy_cuda_common.h"
#include "../../cupy_backends/stub/cupy_cuComplex.h"

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

// cufftXt functions
cufftResult_t cufftXtSetGPUs(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftXtSetWorkArea(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftXtMemcpy(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftXtMakePlanMany(...) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftXtExec(...) {
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

#if defined(CUPY_NO_CUDA) || defined(CUPY_USE_HIP)
// common stubs for both no-cuda and hip environments

extern "C" {
// cufftXt relavant data structs
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

} // extern "C"

#endif // #if defined(CUPY_NO_CUDA) || defined(CUPY_USE_HIP)

#endif  // INCLUDE_GUARD_CUPY_CUFFT_H
