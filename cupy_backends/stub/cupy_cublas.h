// This file is a stub header file of cuBLAS for Read the Docs. Its code was
// automatically generated. Do not modify it directly.

#ifndef INCLUDE_GUARD_STUB_CUPY_CUBLAS_H
#define INCLUDE_GUARD_STUB_CUPY_CUBLAS_H

#include "cupy_cuda_common.h"

extern "C" {

typedef void* cublasHandle_t;

typedef enum {
    CUBLAS_STATUS_SUCCESS = 0
} cublasStatus_t;
typedef enum{} cublasFillMode_t;
typedef enum{} cublasDiagType_t;
typedef enum{} cublasSideMode_t;
typedef enum{} cublasOperation_t;
typedef enum{} cublasPointerMode_t;
typedef enum{} cublasAtomicsMode_t;
typedef enum{} cublasGemmAlgo_t;
typedef enum{} cublasMath_t;
typedef enum{} cublasComputeType_t;

cublasStatus_t cublasCreate(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetProperty(...) {
    return CUBLAS_STATUS_SUCCESS;
}

size_t cublasGetCudartVersion(...) {
    return 0;
}

cublasStatus_t cublasSetStream(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetAtomicsMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetAtomicsMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetVector(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVector(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMatrix(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMatrix(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetVectorAsync(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVectorAsync(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMatrixAsync(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMatrixAsync(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasNrm2Ex(...) {
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

cublasStatus_t cublasDotEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDotcEx(...) {
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

cublasStatus_t cublasZdotu(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotc(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScalEx(...) {
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

cublasStatus_t cublasAxpyEx(...) {
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

cublasStatus_t cublasCopyEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScopy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDcopy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCcopy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZcopy(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSswap(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDswap(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCswap(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZswap(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSwapEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

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

cublasStatus_t cublasIamaxEx(...) {
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

cublasStatus_t cublasIaminEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasAsumEx(...) {
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

cublasStatus_t cublasSrot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCrot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsrot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZrot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdrot(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasRotEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSrotg(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrotg(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCrotg(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZrotg(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasRotgEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSrotm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrotm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasRotmEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSrotmg(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrotmg(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasRotmgEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

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

cublasStatus_t cublasSgbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStpmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtpmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtpmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtpmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStpsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtpsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtpsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtpsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStbsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtbsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtbsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtbsv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsymv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsymv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsymv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZsymv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasChemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhemv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasChbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhbmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSspmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDspmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasChpmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhpmv(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSger(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgeru(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgerc(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgeru(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgerc(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsyr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsyr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZsyr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCher(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZher(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSspr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDspr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasChpr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhpr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsyr2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsyr2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZsyr2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCher2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZher2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSspr2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDspr2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasChpr2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhpr2(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemm3m(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm3m(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemmEx(...) {
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

cublasStatus_t cublasCsyrkEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsyrk3mEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCherk(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZherk(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCherkEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCherk3mEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsyr2k(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr2k(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsyr2k(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZsyr2k(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCher2k(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZher2k(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsyrkx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyrkx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsyrkx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZsyrkx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCherkx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZherkx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsymm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsymm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsymm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZsymm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasChemm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhemm(...) {
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

cublasStatus_t cublasStrmm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrmm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrmm(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrmm(...) {
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

cublasStatus_t cublasCgemm3mStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemmStridedBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

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

cublasStatus_t cublasSmatinvBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDmatinvBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCmatinvBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZmatinvBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgeqrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeqrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgeqrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgeqrfBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgelsBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgelsBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgelsBatched(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgelsBatched(...) {
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

cublasStatus_t cublasStpttr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtpttr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtpttr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtpttr(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStrttp(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrttp(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrttp(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrttp(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmBatchedEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmStridedBatchedEx(...) {
    return CUBLAS_STATUS_SUCCESS;
}

#define cublasGemmEx_v11 cublasGemmEx
#define cublasGemmBatchedEx_v11 cublasGemmBatchedEx
#define cublasGemmStridedBatchedEx_v11 cublasGemmStridedBatchedEx

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUBLAS_H
