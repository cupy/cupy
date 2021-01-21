// This code was automatically generated. Do not modify it directly.

#ifndef INCLUDE_GUARD_CUDA_CUPY_CUSPARSE_H
#define INCLUDE_GUARD_CUDA_CUPY_CUSPARSE_H

#include <cuda.h>
#include <cusparse.h>

extern "C" {

#if !defined(CUSPARSE_VERSION)
#if CUDA_VERSION < 10000
#define CUSPARSE_VERSION CUDA_VERSION // CUDA_VERSION used instead
#else
#define CUSPARSE_VERSION 10000
#endif
#endif // #if !defined(CUSPARSE_VERSION)

#if CUSPARSE_VERSION < 10200
// Added in 10.2

typedef void* cusparseSpVecDescr_t;
typedef void* cusparseDnVecDescr_t;
typedef void* cusparseSpMatDescr_t;
typedef void* cusparseDnMatDescr_t;

typedef enum {} cusparseCsr2CscAlg_t;
typedef enum {} cusparseFormat_t;
typedef enum {} cusparseOrder_t;
typedef enum {} cusparseSpMVAlg_t;
typedef enum {} cusparseSpMMAlg_t;
typedef enum {} cusparseIndexType_t;

cusparseStatus_t cusparseCsr2cscEx2_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsr2cscEx2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateSpVec(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroySpVec(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecGet(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecGetIndexBase(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecGetValues(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecSetValues(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateCoo(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateCooAoS(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateCsr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroySpMat(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCooGet(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCooAoSGet(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsrGet(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetFormat(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetIndexBase(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetValues(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatSetValues(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetStridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatSetStridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateDnVec(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyDnVec(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnVecGet(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnVecGetValues(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnVecSetValues(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateDnMat(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyDnMat(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatGet(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatGetValues(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatSetValues(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatGetStridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatSetStridedBatch(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVV_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVV(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMV_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMV(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMM_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMM(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseConstrainedGeMM_bufferSize(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseConstrainedGeMM(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

#endif  // #if CUSPARSE_VERSION < 10200

#if CUSPARSE_VERSION < 11000
// Added in 11.0

typedef void* cusparseSpGEMMDescr_t;

typedef enum {} cusparseSpGEMMAlg_t;

#endif  // #if CUSPARSE_VERSION < 11000

#if CUSPARSE_VERSION < 11100
// Added in 11.1

typedef enum {} cusparseSparseToDenseAlg_t;
typedef enum {} cusparseDenseToSparseAlg_t;

#endif  // #if CUSPARSE_VERSION < 11100

#if CUSPARSE_VERSION >= 11000
// Removed in 11.0

typedef void* cusparseSolveAnalysisInfo_t;
typedef void* cusparseHybMat_t;

typedef enum {} cusparseHybPartition_t;

cusparseStatus_t cusparseScsrmv(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmv(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrmv(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrmv(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrmm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrmm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrmm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrmm2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmm2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrmm2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrmm2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsrgeamNnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrgeam(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrgeam(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrgeam(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrgeam(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsrgemmNnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrgemm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrgemm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsrgemm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsrgemm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsr2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsr2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCcsr2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseZcsr2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

#endif  // #if CUSPARSE_VERSION >= 11000

}  // extern "C"

#endif  // INCLUDE_GUARD_CUDA_CUPY_CUSPARSE_H
