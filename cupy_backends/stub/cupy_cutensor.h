// Stub header file of cuTENSOR

#ifndef INCLUDE_GUARD_STUB_CUPY_CUTENSOR_H
#define INCLUDE_GUARD_STUB_CUPY_CUTENSOR_H

#define CUTENSOR_VERSION 0

#include "cupy_cuda_common.h"

extern "C" {

    typedef enum {} cutensorDataType_t;
    typedef enum {} cutensorComputeType_t;

    typedef enum {
	CUTENSOR_STATUS_SUCCESS = 0,
    } cutensorStatus_t;

    typedef enum {} cutensorAlgo_t;
    typedef enum {} cutensorOperator_t;
    typedef enum {} cutensorPlan_t;
    typedef enum {} cutensorPlanPreference_t;
    typedef enum {} cutensorPlanPreferenceAttribute_t;
    typedef enum {} cutensorJitMode_t;
    typedef enum {} cutensorCacheMode_t;
    typedef enum {} cutensorWorksizePreference_t;

    typedef void* cutensorHandle_t;
    typedef void* cutensorTensorDescriptor_t;
    typedef void* cutensorOperationDescriptor_t;
    typedef void* cutensorComputeDescriptor_t;
   
    const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16F = NULL;
    const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_16BF = NULL;
    const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_TF32 = NULL;
    const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_3XTF32 = NULL;
    const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_32F = NULL;
    const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_64F = NULL;

    cutensorStatus_t cutensorInit(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorCreate(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorDestroy(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorCreateTensorDescriptor(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorDestroyTensorDescriptor(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorDestroyOperationDescriptor(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorCreateElementwiseTrinary(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorElementwiseTrinaryExecute(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorCreateElementwiseBinary(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorElementwiseBinaryExecute(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorCreatePermutation(...) {
        return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorPermute(...) {
        return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorCreateContraction(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorDestroyContraction(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorCreatePlanPreference(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorDestroyPlanPreference(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorCreatePlan(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorDestroyPlan(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorContract(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorEstimateWorkspaceSize(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorCreateReduction(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorReduce(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorGetAlignmentRequirement(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    size_t cutensorGetVersion(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    size_t cutensorGetCudartVersion(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    const char* cutensorGetErrorString(...) {
	return NULL;
    }

    typedef enum {} cudaDataType_t;
    typedef enum {} cutensorMgCopyPlan_t;
    typedef enum {} cutensorMgCopyDescriptor_t;
    typedef enum {} cutensorMgContractionDescriptor_t;
    typedef enum {} cutensorMgContractionFind_t;
    typedef enum {} cutensorMgContractionPlan_t;
    typedef enum {} cutensorMgAlgo_t;
    typedef void* cutensorMgHandle_t;
    typedef void* cutensorMgTensorDescriptor_t;

    cutensorStatus_t cutensorMgCreate(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgDestroy(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgCreateTensorDescriptor(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgDestroyTensorDescriptor(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgCreateCopyDescriptor(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgDestroyCopyDescriptor(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgCopyGetWorkspace(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgCreateCopyPlan(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgDestroyCopyPlan(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgCopy(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgCreateContractionDescriptor(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgDestroyContractionDescriptor(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgCreateContractionFind(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgDestroyContractionFind(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgContractionGetWorkspace(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgCreateContractionPlan(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgDestroyContractionPlan(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorMgContraction(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }
} // extern "C"

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUTENSOR_H
