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

cublasStatus_t cublasSsbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
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

cublasStatus_t cublasStrsmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrsmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsmBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsyrk(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyrk(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsyrk(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZsyrk(...) {
    return CUBLAS_STATUS_SUCCESS;
}

typedef enum{} cublasComputeType_t;
cublasStatus_t cublasGemmExBit_v11(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmStridedBatchedEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmStridedBatchedExBit_v11(...) {
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

typedef int intBit;
#define cublasIsamaxBit cublasIsamax
#define cublasIdamaxBit cublasIdamax
#define cublasIcamaxBit cublasIcamax
#define cublasIzamaxBit cublasIzamax
#define cublasIsaminBit cublasIsamin
#define cublasIdaminBit cublasIdamin
#define cublasIcaminBit cublasIcamin
#define cublasIzaminBit cublasIzamin
#define cublasSasumBit cublasSasum
#define cublasDasumBit cublasDasum
#define cublasScasumBit cublasScasum
#define cublasDzasumBit cublasDzasum
#define cublasSaxpyBit cublasSaxpy
#define cublasDaxpyBit cublasDaxpy
#define cublasCaxpyBit cublasCaxpy
#define cublasZaxpyBit cublasZaxpy
#define cublasSdotBit cublasSdot
#define cublasDdotBit cublasDdot
#define cublasCdotuBit cublasCdotu
#define cublasCdotcBit cublasCdotc
#define cublasZdotuBit cublasZdotu
#define cublasZdotcBit cublasZdotc
#define cublasSnrm2Bit cublasSnrm2
#define cublasDnrm2Bit cublasDnrm2
#define cublasScnrm2Bit cublasScnrm2
#define cublasDznrm2Bit cublasDznrm2
#define cublasSscalBit cublasSscal
#define cublasDscalBit cublasDscal
#define cublasCscalBit cublasCscal
#define cublasCsscalBit cublasCsscal
#define cublasZscalBit cublasZscal
#define cublasZdscalBit cublasZdscal
#define cublasSgemvBit cublasSgemv
#define cublasDgemvBit cublasDgemv
#define cublasCgemvBit cublasCgemv
#define cublasZgemvBit cublasZgemv
#define cublasSgerBit cublasSger
#define cublasDgerBit cublasDger
#define cublasCgeruBit cublasCgeru
#define cublasCgercBit cublasCgerc
#define cublasZgeruBit cublasZgeru
#define cublasZgercBit cublasZgerc
#define cublasSsbmvBit cublasSsbmv
#define cublasDsbmvBit cublasDsbmv
#define cublasSgemmBit cublasSgemm
#define cublasDgemmBit cublasDgemm
#define cublasCgemmBit cublasCgemm
#define cublasZgemmBit cublasZgemm
#define cublasSgemmBatchedBit cublasSgemmBatched
#define cublasDgemmBatchedBit cublasDgemmBatched
#define cublasCgemmBatchedBit cublasCgemmBatched
#define cublasZgemmBatchedBit cublasZgemmBatched
#define cublasSgemmStridedBatchedBit cublasSgemmStridedBatched
#define cublasDgemmStridedBatchedBit cublasDgemmStridedBatched
#define cublasCgemmStridedBatchedBit cublasCgemmStridedBatched
#define cublasZgemmStridedBatchedBit cublasZgemmStridedBatched
#define cublasStrsmBit cublasStrsm
#define cublasDtrsmBit cublasDtrsm
#define cublasCtrsmBit cublasCtrsm
#define cublasZtrsmBit cublasZtrsm
#define cublasStrsmBatchedBit cublasStrsmBatched
#define cublasDtrsmBatchedBit cublasDtrsmBatched
#define cublasCtrsmBatchedBit cublasCtrsmBatched
#define cublasZtrsmBatchedBit cublasZtrsmBatched
#define cublasSsyrkBit cublasSsyrk
#define cublasDsyrkBit cublasDsyrk
#define cublasCsyrkBit cublasCsyrk
#define cublasZsyrkBit cublasZsyrk
#define cublasSgeamBit cublasSgeam
#define cublasDgeamBit cublasDgeam
#define cublasCgeamBit cublasCgeam
#define cublasZgeamBit cublasZgeam
#define cublasSdgmmBit cublasSdgmm
#define cublasDdgmmBit cublasDdgmm
#define cublasCdgmmBit cublasCdgmm
#define cublasZdgmmBit cublasZdgmm
#define cublasSgemmExBit cublasSgemmEx
#define cublasGemmExBit cublasGemmEx
#define cublasGemmStridedBatchedExBit cublasGemmStridedBatchedEx

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUBLAS_H
