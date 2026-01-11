#ifndef INCLUDE_GUARD_ASCEND_CUPY_PROFILER_H
#define INCLUDE_GUARD_ASCEND_CUPY_PROFILER_H

#include "cupy_ascend_common.h"
#include "acl/acl_prof.h"

extern "C" {

// profiler init
cudaError_t cudaProfilerStart() {
    //return aclprofStart();  // TODO: this lead to runtime error?
    return CUDA_SUCCESS;
}

cudaError_t cudaProfilerStop() {
    // return aclprofStop();
    return CUDA_SUCCESS;
}

} // extern "C"

#endif // #ifndef INCLUDE_GUARD_ASCEND_CUPY_PROFILER_H
