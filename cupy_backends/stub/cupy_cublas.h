// This file is a stub header file of cuda for Read the Docs.

#ifndef INCLUDE_GUARD_STUB_CUPY_CUBLAS_H
#define INCLUDE_GUARD_STUB_CUPY_CUBLAS_H

#include "cupy_cuda_common.h"

extern "C" {

// Context
cublasStatus_t cublasCreate(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// Stream
cublasStatus_t cublasSetStream(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// Math Mode
cublasStatus_t cublasSetMathMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// BLAS Level 1
cublasStatus_t cublasIsamax(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamax(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIcamax(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIzamax(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamin(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamin(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIcamin(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIzamin(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSasum(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDasum(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScasum(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDzasum(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCaxpy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZaxpy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotu(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotc(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotc(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotu(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSnrm2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDnrm2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScnrm2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDznrm2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDscal(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCscal(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsscal(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZscal(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdscal(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// BLAS Level 2
cublasStatus_t cublasSgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasCgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSger(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger(...) {
    return CUBLAS_STATUS_SUCCESS;
}

int cublasCgeru(...) {
    return 0;
}

int cublasCgerc(...) {
    return 0;
}

int cublasZgeru(...) {
    return 0;
}

int cublasZgerc(...) {
    return 0;
}

// BLAS Level 3
cublasStatus_t cublasSgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasCgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrsm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrsm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

typedef enum{} cublasComputeType_t;
cublasStatus_t cublasGemmEx_v11(...) {
    return CUBLAS_STATUS_SUCCESS;
}

// BLAS extension
cublasStatus_t cublasSgeam(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeam(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgeam(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgeam(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdgmm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdgmm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdgmm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdgmm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetrsBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetrsBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetrsBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetrsBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetriBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgetriBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgetriBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgetriBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrttp(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrttp(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStpttr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtpttr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUBLAS_H
