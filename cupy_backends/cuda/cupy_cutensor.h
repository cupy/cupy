#ifndef INCLUDE_GUARD_CUDA_CUPY_CUTENSOR_H
#define INCLUDE_GUARD_CUDA_CUPY_CUTENSOR_H

#include <library_types.h>
#include <cutensor.h>

#if CUTENSOR_VERSION < 10500

cutensorStatus_t cutensorContractionGetWorkspaceSize(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}

cutensorStatus_t cutensorReductionGetWorkspaceSize(...) {
    return CUTENSOR_STATUS_NOT_SUPPORTED;
}

#endif  // CUTENSOR_VERSION < 10500

#endif  // INCLUDE_GUARD_CUDA_CUPY_CUTENSOR_H
