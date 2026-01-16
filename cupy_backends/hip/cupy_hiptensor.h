#ifndef INCLUDE_GUARD_HIP_CUPY_HIPTENSOR_H
#define INCLUDE_GUARD_HIP_CUPY_HIPTENSOR_H

#include "cupy_hip.h"

#include <cupy/hiptensor.h>

extern "C" {

// Core types
typedef hiptensorHandle_t cutensorHandle_t;
typedef hiptensorTensorDescriptor_t cutensorTensorDescriptor_t;
typedef hiptensorOperationDescriptor_t cutensorOperationDescriptor_t;
typedef hiptensorPlanPreference_t cutensorPlanPreference_t;
typedef hiptensorPlan_t cutensorPlan_t;

typedef hiptensorStatus_t cutensorStatus_t;
typedef hiptensorAlgo_t cutensorAlgo_t;
typedef hiptensorJitMode_t cutensorJitMode_t;
typedef hiptensorOperator_t cutensorOperator_t;
typedef hiptensorWorksizePreference_t cutensorWorksizePreference_t;
typedef hiptensorPlanAttribute_t cutensorPlanAttribute_t;
typedef hiptensorPlanPreferenceAttribute_t cutensorPlanPreferenceAttribute_t;
typedef hiptensorCacheMode_t cutensorCacheMode_t;
typedef hiptensorDataType_t cutensorDataType_t;
typedef hiptensorComputeType_t cutensorComputeType_t;

typedef hiptensorComputeDescriptor_t cutensorComputeDescriptor_t;

// Version mapping
#ifndef CUTENSOR_VERSION
#ifdef HIPTENSOR_VERSION
#define CUTENSOR_VERSION HIPTENSOR_VERSION
#else
#define CUTENSOR_VERSION 0
#endif
#endif

// Status mapping (for stubs below)
#ifndef CUTENSOR_STATUS_SUCCESS
#ifdef HIPTENSOR_STATUS_SUCCESS
#define CUTENSOR_STATUS_SUCCESS HIPTENSOR_STATUS_SUCCESS
#else
#define CUTENSOR_STATUS_SUCCESS 0
#endif
#endif

#ifndef CUTENSOR_STATUS_NOT_SUPPORTED
#ifdef HIPTENSOR_STATUS_NOT_SUPPORTED
#define CUTENSOR_STATUS_NOT_SUPPORTED HIPTENSOR_STATUS_NOT_SUPPORTED
#else
#define CUTENSOR_STATUS_NOT_SUPPORTED 15
#endif
#endif

// Function mapping (core APIs)
#define cutensorGetErrorString hiptensorGetErrorString
#define cutensorGetVersion hiptensorGetVersion

#define cutensorCreate hiptensorCreate
#define cutensorDestroy hiptensorDestroy

#define cutensorCreateTensorDescriptor hiptensorCreateTensorDescriptor
#define cutensorDestroyTensorDescriptor hiptensorDestroyTensorDescriptor

#define cutensorCreatePlanPreference hiptensorCreatePlanPreference
#define cutensorDestroyPlanPreference hiptensorDestroyPlanPreference

#define cutensorEstimateWorkspaceSize hiptensorEstimateWorkspaceSize

#define cutensorCreatePlan hiptensorCreatePlan
#define cutensorPlanGetAttribute hiptensorPlanGetAttribute
#define cutensorDestroyPlan hiptensorDestroyPlan

#define cutensorCreateElementwiseTrinary hiptensorCreateElementwiseTrinary
#define cutensorElementwiseTrinaryExecute hiptensorElementwiseTrinaryExecute

#define cutensorCreateElementwiseBinary hiptensorCreateElementwiseBinary
#define cutensorElementwiseBinaryExecute hiptensorElementwiseBinaryExecute

#define cutensorCreatePermutation hiptensorCreatePermutation
#define cutensorPermute hiptensorPermute

#define cutensorCreateContraction hiptensorCreateContraction
#define cutensorContract hiptensorContract

#define cutensorCreateReduction hiptensorCreateReduction
#define cutensorReduce hiptensorReduce

#define cutensorDestroyOperationDescriptor hiptensorDestroyOperationDescriptor

#define cutensorGetAlignmentRequirement hiptensorGetAlignmentRequirement

// cuTENSOR compute descriptor constants
#ifdef HIPTENSOR_COMPUTE_DESC_16F
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16F =
    HIPTENSOR_COMPUTE_DESC_16F;
#else
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16F = NULL;
#endif
#ifdef HIPTENSOR_COMPUTE_DESC_16BF
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16BF =
    HIPTENSOR_COMPUTE_DESC_16BF;
#else
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16BF = NULL;
#endif
#ifdef HIPTENSOR_COMPUTE_DESC_TF32
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_TF32 =
    HIPTENSOR_COMPUTE_DESC_TF32;
#else
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_TF32 = NULL;
#endif
#ifdef HIPTENSOR_COMPUTE_DESC_3XTF32
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_3XTF32 =
    HIPTENSOR_COMPUTE_DESC_3XTF32;
#else
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_3XTF32 = NULL;
#endif
#ifdef HIPTENSOR_COMPUTE_DESC_32F
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_32F =
    HIPTENSOR_COMPUTE_DESC_32F;
#else
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_32F = NULL;
#endif
#ifdef HIPTENSOR_COMPUTE_DESC_64F
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_64F =
    HIPTENSOR_COMPUTE_DESC_64F;
#else
static const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_64F = NULL;
#endif

// hipTensor does not expose a CUDA runtime version query; fall back to HIP.
static inline size_t cutensorGetCudartVersion() {
    int version = 0;
    hipRuntimeGetVersion(&version);
    return static_cast<size_t>(version);
}

// cuTENSORMg is not available in hipTensor. Provide stubs.
typedef void* cutensorMgHandle_t;
typedef void* cutensorMgTensorDescriptor_t;
typedef void* cutensorMgCopyDescriptor_t;
typedef void* cutensorMgCopyPlan_t;
typedef void* cutensorMgContractionDescriptor_t;
typedef void* cutensorMgContractionFind_t;
typedef void* cutensorMgContractionPlan_t;
typedef int cutensorMgAlgo_t;

static inline cutensorStatus_t cutensorMgCreate(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgDestroy(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgCreateTensorDescriptor(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgDestroyTensorDescriptor(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgCreateCopyDescriptor(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgDestroyCopyDescriptor(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgCopyGetWorkspace(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgCreateCopyPlan(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgDestroyCopyPlan(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgCopy(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgCreateContractionDescriptor(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgDestroyContractionDescriptor(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgCreateContractionFind(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgDestroyContractionFind(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgContractionGetWorkspace(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgCreateContractionPlan(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgDestroyContractionPlan(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}
static inline cutensorStatus_t cutensorMgContraction(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}

}  // extern "C"

#endif  // INCLUDE_GUARD_HIP_CUPY_HIPTENSOR_H
