// Stub header file of cuTENSOR

#ifndef INCLUDE_GUARD_STUB_CUPY_CUTENSOR_H
#define INCLUDE_GUARD_STUB_CUPY_CUTENSOR_H

#include "cupy_cuda_common.h"

extern "C" {

    typedef enum {} cudaDataType_t;

    typedef enum {
	CUTENSOR_STATUS_SUCCESS = 0,
    } cutensorStatus_t;

    typedef enum {} cutensorAlgo_t;
    typedef enum {} cutensorOperator_t;
    typedef enum {} cutensorWorksizePreference_t;
    typedef enum {} cutensorComputeType_t;

    typedef void* cutensorHandle_t;
    typedef void* cutensorTensorDescriptor_t;
    typedef void* cutensorContractionDescriptor_t;
    typedef void* cutensorContractionFind_t;
    typedef void* cutensorContractionPlan_t;

    cutensorStatus_t cutensorInit(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorInitTensorDescriptor(...) {
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

    cutensorStatus_t cutensorGetAlignmentRequirement(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    size_t cutensorGetVersion(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    const char* cutensorGetErrorString(...) {
	return NULL;
    }

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_STUB_CUPY_CUTENSOR_H
