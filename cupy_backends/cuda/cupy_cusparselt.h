#ifndef INCLUDE_GUARD_CUDA_CUPY_CUSPARSELT_H
#define INCLUDE_GUARD_CUDA_CUPY_CUSPARSELT_H

#include <library_types.h>
#include <cusparseLt.h>

#if CUSPARSELT_VERSION < 100
// Added in cuSPARSELt 0.1.0

cusparseStatus_t cusparseLtMatDescriptorDestroy(...) {
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseLtSpMMAPrune2(...) {
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseLtSpMMAPruneCheck2(...) {
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseLtSpMMACompressedSize2(...) {
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseLtSpMMACompress2(...) {
    return CUSPARSE_STATUS_SUCCESS;
}

#endif  // CUSPARSELT_VERSION < 100

#endif  // INCLUDE_GUARD_CUDA_CUPY_CUSPARSELT_H
