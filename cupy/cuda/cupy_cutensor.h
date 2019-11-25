// Stub header file of cuTENSOR

#ifndef INCLUDE_GUARD_CUPY_CUTENSOR_H
#define INCLUDE_GUARD_CUPY_CUTENSOR_H

#ifndef CUPY_NO_CUDA

#include <library_types.h>
#include <cutensor.h>

void _cutensor_alloc_handle(cutensorHandle_t **handle);
void _cutensor_free_handle(cutensorHandle_t *handle);

void _cutensor_alloc_tensor_descriptor(cutensorTensorDescriptor_t **desc);
void _cutensor_free_tensor_descriptor(cutensorTensorDescriptor_t *desc);

void _cutensor_alloc_contraction_descriptor(cutensorContractionDescriptor_t **desc);
void _cutensor_free_contraction_descriptor(cutensorContractionDescriptor_t *desc);

void _cutensor_alloc_contraction_plan(cutensorContractionPlan_t **plan);
void _cutensor_free_contraction_plan(cutensorContractionPlan_t *plan);

void _cutensor_alloc_contraction_find(cutensorContractionFind_t **find);
void _cutensor_free_contraction_find(cutensorContractionFind_t *find);

#else // #ifndef CUPY_NO_CUDA

#include "cupy_cuda_common.h"

extern "C" {

    typedef enum {} cudaDataType_t;

    typedef enum {
	CUTENSOR_STATUS_SUCCESS = 0,
    } cutensorStatus_t;

    typedef enum {} cutensorAlgo_t;
    typedef enum {} cutensorOperator_t;
    typedef enum {} cutensorWorksizePreference_t;

    typedef void* cutensorHandle_t;
    typedef void* cutensorTensorDescriptor_t;

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

    cutensorStatus_t cutensorElementwiseTrinary(...) {
	return CUTENSOR_STATUS_SUCCESS;
    }

    cutensorStatus_t cutensorElementwiseBinary(...) {
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

    const char* cutensorGetErrorString(...) {
	return NULL;
    }

} // extern "C"

void _cutensor_alloc_handle(...) {}
void _cutensor_free_handle(...) {}

void _cutensor_alloc_tensor_descriptor(...) {}
void _cutensor_free_tensor_descriptor(...) {}

void _cutensor_alloc_contraction_descriptor(...) {}
void _cutensor_free_contraction_descriptor(...) {}

void _cutensor_alloc_contraction_plan(...) {}
void _cutensor_free_contraction_plan(...) {}

void _cutensor_alloc_contraction_find(...) {}
void _cutensor_free_contraction_find(...) {}

#endif // #ifndef CUPY_NO_CUDA

#endif // #ifndef INCLUDE_GUARD_CUPY_CUTENSOR_H
