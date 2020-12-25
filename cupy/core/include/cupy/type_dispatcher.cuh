#ifndef INCLUDE_GUARD_CUPY_TYPE_DISPATCHER_H
#define INCLUDE_GUARD_CUPY_TYPE_DISPATCHER_H

#include <cupy/complex.cuh>
#include <stdexcept>
#include <cstdint>
#if (__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
    && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))
#include <cuda_fp16.h>
#elif (defined(__HIPCC__) || defined(CUPY_USE_HIP))
#include <hip/hip_fp16.h>
#endif


#define CUPY_TYPE_INT8        0
#define CUPY_TYPE_UINT8       1
#define CUPY_TYPE_INT16       2
#define CUPY_TYPE_UINT16      3
#define CUPY_TYPE_INT32       4
#define CUPY_TYPE_UINT32      5
#define CUPY_TYPE_INT64       6
#define CUPY_TYPE_UINT64      7
#define CUPY_TYPE_FLOAT16     8
#define CUPY_TYPE_FLOAT32     9
#define CUPY_TYPE_FLOAT64    10
#define CUPY_TYPE_COMPLEX64  11
#define CUPY_TYPE_COMPLEX128 12
#define CUPY_TYPE_BOOL       13


//
// **** dtype_dispatcher ****
//
// This is implemented with reference to the following implementation.
// https://github.com/rapidsai/cudf/blob/branch-0.6/cpp/src/utilities/type_dispatcher.hpp
//
template <class functor_t, typename... Ts>
void dtype_dispatcher(int dtype_id, functor_t f, Ts&&... args)
{
    switch (dtype_id) {
    case CUPY_TYPE_INT8:       return f.template operator()<char>(std::forward<Ts>(args)...);
    case CUPY_TYPE_INT16:      return f.template operator()<short>(std::forward<Ts>(args)...);
    case CUPY_TYPE_INT32:      return f.template operator()<int>(std::forward<Ts>(args)...);
    case CUPY_TYPE_INT64:      return f.template operator()<int64_t>(std::forward<Ts>(args)...);
    case CUPY_TYPE_UINT8:      return f.template operator()<unsigned char>(std::forward<Ts>(args)...);
    case CUPY_TYPE_UINT16:     return f.template operator()<unsigned short>(std::forward<Ts>(args)...);
    case CUPY_TYPE_UINT32:     return f.template operator()<unsigned int>(std::forward<Ts>(args)...);
    case CUPY_TYPE_UINT64:     return f.template operator()<uint64_t>(std::forward<Ts>(args)...);
#if ((__CUDACC_VER_MAJOR__ > 9 || (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ == 2)) \
     && (__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))) || (defined(__HIPCC__) || defined(CUPY_USE_HIP))
    case CUPY_TYPE_FLOAT16:    return f.template operator()<__half>(std::forward<Ts>(args)...);
#endif
    case CUPY_TYPE_FLOAT32:    return f.template operator()<float>(std::forward<Ts>(args)...);
    case CUPY_TYPE_FLOAT64:    return f.template operator()<double>(std::forward<Ts>(args)...);
    case CUPY_TYPE_COMPLEX64:  return f.template operator()<complex<float>>(std::forward<Ts>(args)...);
    case CUPY_TYPE_COMPLEX128: return f.template operator()<complex<double>>(std::forward<Ts>(args)...);
    case CUPY_TYPE_BOOL:       return f.template operator()<bool>(std::forward<Ts>(args)...);
    default:
	    throw std::runtime_error("Unsupported dtype ID");
    }
}

#endif  // #ifndef INCLUDE_GUARD_CUPY_TYPE_DISPATCHER_H
