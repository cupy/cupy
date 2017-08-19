// This file is a stub header file of cusparse for Read the Docs.

#ifndef INCLUDE_GUARD_CUPY_CUSPARSE_H
#define INCLUDE_GUARD_CUPY_CUSPARSE_H

#ifndef CUPY_NO_CUDA
#  include <cusparse.h>

#else  // CUPY_NO_CUDA
extern "C" {

typedef enum {} cusparseIndexBase_t;
typedef enum {
  CUSPARSE_STATUS_SUCCESS=0,
}  cusparseStatus_t;

typedef void* cusparseHandle_t;
typedef void* cusparseMatDescr_t;

typedef enum {} cusparseMatrixType_t;
typedef enum {} cusparseOperation_t;
typedef enum {} cusparsePointerMode_t;
typedef enum {} cusparseAction_t;
typedef enum {} cusparseDirection_t;

// cuSPARSE Helper Function
cusparseStatus_t cusparseCreate(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateMatDescr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroy(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyMatDescr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatIndexBase(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatType(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetPointerMode(...) {
  return CUSPARSE_STATUS_SUCCESS;
}


// cuSPARSE Level1 Function
cusparseStatus_t cusparseSgthr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDgthr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

// cuSPARSE Level2 Function
cusparseStatus_t cusparseScsrmv(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmv(...) {
  return CUSPARSE_STATUS_SUCCESS;
}


// cuSPARSE Level3 Function
cusparseStatus_t cusparseScsrmm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmm(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrmm2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmm2(...) {
  return CUSPARSE_STATUS_SUCCESS;
}


// cuSPARSE Extra Function
cusparseStatus_t cusparseXcsrgeamNnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsrgeam(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrgeam(...) {
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


// cuSPARSE Format Convrsion
cusparseStatus_t cusparseXcoo2csr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsc2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsc2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsr2coo(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsr2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsr2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseScsr2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsr2dense(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSdense2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDdense2csc(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSdense2csr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDdense2csr(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSnnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnnz(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateIdentityPermutation(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcoosort_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcoosortByRow(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsrsort_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsrsort(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcscsort_bufferSizeExt(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcscsort(...) {
  return CUSPARSE_STATUS_SUCCESS;
}

}  // extern "C"

#endif  // CUPY_NO_CUDA

#endif  // INCLUDE_GUARD_CUPY_CUSPARSE_H
