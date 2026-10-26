// asdsip BLAS
// cublas: https://docs.nvidia.com/cuda/cublas/index.html
// If the cuBLAS library workspace is not set, all kernels will use the default workspace pool allocated during the cuBLAS context creation.
// AtomicsMode
// cublas accept `float*` while, ascend accepts `aclTensor*`, pure C API, while SIP has c++ Tensor
// enum value compatile?  ask for a compact doc from SIP team

#ifndef CUPY_ASCEND_BLAS_H
#define CUPY_ASCEND_BLAS_H
#include "blas_api.h"
#include "acl_utils.h"
#include <stdexcept>  // for gcc 10

// ACL_MAJOR_VERSION  "acl.h"  v 1.14, not CANN version?

extern "C" {

}

#endif // end of header