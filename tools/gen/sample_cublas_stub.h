// This file is a stub header file for Read the Docs. It was automatically
// generated. Do not modify it directly.

#ifndef INCLUDE_GUARD_STUB_CUPY_CUBLAS_H
#define INCLUDE_GUARD_STUB_CUPY_CUBLAS_H

extern "C" {

typedef void* cublasHandle_t;

typedef enum {
  CUBLAS_STATUS_SUCCESS = 0
} cublasStatus_t;
typedef enum {} cublasFillMode_t;
typedef enum {} cublasDiagType_t;
typedef enum {} cublasSideMode_t;
typedef enum {} cublasOperation_t;
typedef enum {} cublasPointerMode_t;
typedef enum {} cublasAtomicsMode_t;
typedef enum {} cublasGemmAlgo_t;
typedef enum {} cublasMath_t;
typedef enum {} cublasComputeType_t;


cublasStatus_t cublasCreate_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamax_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamax_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIcamax_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIzamax_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamin_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamin_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIcamin_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIzamin_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSasum_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDasum_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScasum_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDzasum_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCaxpy_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZaxpy_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotu_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCdotc_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotu_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotc_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSnrm2_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDnrm2_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScnrm2_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDznrm2_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDscal_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCscal_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsscal_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZscal_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdscal_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemv_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemv_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemv_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSger_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgeru_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgerc_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgeru_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgerc_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemm_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemm_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm_v2(...) {
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

cublasStatus_t cublasStrsm_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsm_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCtrsm_v2(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsm_v2(...) {
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

cublasStatus_t cublasSgemmEx(...) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemmEx(...) {
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

}  // extern "C"

#endif  // INCLUDE_GUARD_STUB_CUPY_CUBLAS_H
