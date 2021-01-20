// This code was automatically generated. Do not modify it directly.

#ifndef INCLUDE_GUARD_CUDA_CUPY_CUSOLVER_H
#define INCLUDE_GUARD_CUDA_CUPY_CUSOLVER_H

#include <cuda.h>
#include <cusolverDn.h>
#include <cusolverSp.h>

extern "C" {

#if CUDA_VERSION < 10020
// Added in 10.2

typedef void* cusolverDnIRSParams_t;
typedef void* cusolverDnIRSInfos_t;

typedef enum {} cusolverEigRange_t;
typedef enum {} cusolverNorm_t;
typedef enum {} cusolverIRSRefinement_t;

cusolverStatus_t cusolverDnZZgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZCgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZKgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCCgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCKgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDDgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDSgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDHgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSSgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSHgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZZgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZCgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZKgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCCgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCKgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDDgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDSgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDHgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSSgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSHgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvdaStridedBatched_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvdaStridedBatched_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvdaStridedBatched_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvdaStridedBatched_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvdaStridedBatched(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvdaStridedBatched(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvdaStridedBatched(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZgesvdaStridedBatched(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

#endif  // #if CUDA_VERSION < 10020

#if CUDA_VERSION < 11000
// Added in 11.0

typedef void* cusolverDnParams_t;

typedef enum {} cusolverDnFunction_t;
typedef enum {} cusolverPrecType_t;
typedef enum {} cusolverAlgMode_t;
typedef enum {} cusolverStorevMode_t;
typedef enum {} cusolverDirectMode_t;

cusolverStatus_t cusolverDnZEgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZYgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCEgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCYgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDBgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDXgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSBgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSXgesv_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZEgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZYgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCEgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCYgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDBgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDXgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSBgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSXgesv(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZZgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZCgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZKgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZEgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZYgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCCgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCKgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCEgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCYgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDDgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDSgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDHgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDBgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDXgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSSgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSHgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSBgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSXgels_bufferSize(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZZgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZCgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZKgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZEgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnZYgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCCgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCKgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCEgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCYgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDDgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDSgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDHgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDBgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDXgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSSgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSHgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSBgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSXgels(...) {
  return CUSOLVER_STATUS_SUCCESS;
}

#endif  // #if CUDA_VERSION < 11000

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_CUDA_CUPY_CUSOLVER_H
