// This file is a stub header file for Read the Docs. It was automatically
// generated. Do not modify it directly.

#ifndef INCLUDE_GUARD_STUB_CUPY_CUTENSOR_H
#define INCLUDE_GUARD_STUB_CUPY_CUTENSOR_H

extern "C" {

typedef enum {} cutensorOperator_t;
typedef enum {
  CUTENSOR_STATUS_SUCCESS = 0
} cutensorStatus_t;
typedef enum {} cutensorAlgo_t;
typedef enum {} cutensorWorksizePreference_t;
typedef enum {} cutensorComputeType_t;
typedef enum {} cutensorContractionDescriptorAttributes_t;
typedef enum {} cutensorContractionFindAttributes_t;
typedef enum {} cutensorAutotuneMode_t;
typedef enum {} cutensorCacheMode_t;

cutensorStatus_t cutensorInit(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorInitTensorDescriptor(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorGetAlignmentRequirement(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorElementwiseTrinary(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorElementwiseBinary(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorInitContractionDescriptor(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorInitContractionFind(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorInitContractionPlan(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorContraction(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorContractionGetWorkspace(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorContractionMaxAlgos(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorReduction(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

cutensorStatus_t cutensorReductionGetWorkspace(...) {
  return CUTENSOR_STATUS_SUCCESS;
}

}  // extern "C"

#endif  // INCLUDE_GUARD_STUB_CUPY_CUTENSOR_H
