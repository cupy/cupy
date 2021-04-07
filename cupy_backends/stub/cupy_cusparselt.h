// Stub header file for cuSPARSELt

#ifndef INCLUDE_GUARD_STUB_CUPY_CUSPARSELT_H
#define INCLUDE_GUARD_STUB_CUPY_CUSPARSELT_H

#define CUSPARSELT_VERSION -1

extern "C" {

    typedef enum {
	CUSPARSE_STATUS_SUCCESS=0,
    }  cusparseStatus_t;
    typedef enum {} cudaDataType;
    typedef enum {} cusparseOrder_t;
    typedef enum {} cusparseOperation_t;
    typedef enum {} cusparseLtSparsity_t;
    typedef enum {} cusparseComputeType;
    typedef enum {} cusparseLtMatmulAlg_t;
    typedef enum {} cusparseLtMatmulAlgAttribute_t;
    typedef enum {} cusparseLtPruneAlg_t;

    typedef void* cudaStream_t;
    typedef void* cusparseLtHandle_t;
    typedef void* cusparseLtMatDescriptor_t;
    typedef void* cusparseLtMatmulDescriptor_t;
    typedef void* cusparseLtMatmulAlgSelection_t;
    typedef void* cusparseLtMatmulPlan_t;

    cusparseStatus_t cusparseLtInit(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtDestroy(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtDenseDescriptorInit(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtStructuredDescriptorInit(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtMatmulDescriptorInit(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtMatmulAlgSelectionInit(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtMatmulAlgSetAttribute(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtMatmulAlgGetAttribute(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtMatmulGetWorkspace(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtMatmulPlanInit(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtMatmulPlanDestroy(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtMatmul(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtMatmulSearch(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtSpMMAPrune(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtSpMMAPruneCheck(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtSpMMACompressedSize(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

    cusparseStatus_t cusparseLtSpMMACompress(...) {
	return CUSPARSE_STATUS_SUCCESS;
    }

} // extern "C"

#endif  // INCLUDE_GUARD_STUB_CUPY_CUSPARSE_H
