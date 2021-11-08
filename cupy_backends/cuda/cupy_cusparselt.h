#ifndef INCLUDE_GUARD_CUDA_CUPY_CUSPARSELT_H
#define INCLUDE_GUARD_CUDA_CUPY_CUSPARSELT_H

#include <library_types.h>
#include <cusparseLt.h>

#if CUSPARSELT_VERSION < 100
// Added in cuSPARSELt 0.1.0

cusparseStatus_t cusparseLtMatDescriptorDestroy(...) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseLtSpMMAPrune2(...) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseLtSpMMAPruneCheck2(...) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseLtSpMMACompressedSize2(...) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseLtSpMMACompress2(...) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

#endif  // CUSPARSELT_VERSION < 100

#if CUSPARSELT_VERSION < 200
// Added in cuSPARSELt 0.2.0

typedef enum {} cusparseLtMatDescAttribute_t;

cusparseStatus_t cusparseLtMatDescSetAttribute(...) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseLtMatDescGetAttribute(...) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

typedef enum {} cusparseLtMatmulDescAttribute_t;

cusparseStatus_t cusparseLtMatmulDescSetAttribute(...) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseLtMatmulDescGetAttribute(...) {
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

#endif  // CUSPARSELT_VERSION < 200

#endif  // INCLUDE_GUARD_CUDA_CUPY_CUSPARSELT_H
