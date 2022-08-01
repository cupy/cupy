#include <functional>

#include <cupy/type_dispatcher.cuh>
#include "cupy_thrust.h"
#include "cupy_thrust.inl"


namespace cupy {
  std::function<cupy::SortTargetT> sort_targets[CUPY_NUM_TYPES] = {
    thrust_sort_CUPY_TYPE_INT8      ,
    thrust_sort_CUPY_TYPE_UINT8     ,
    thrust_sort_CUPY_TYPE_INT16     ,
    thrust_sort_CUPY_TYPE_UINT16    ,
    thrust_sort_CUPY_TYPE_INT32     ,
    thrust_sort_CUPY_TYPE_UINT32    ,
    thrust_sort_CUPY_TYPE_INT64     ,
    thrust_sort_CUPY_TYPE_UINT64    ,
    thrust_sort_CUPY_TYPE_FLOAT16   ,
    thrust_sort_CUPY_TYPE_FLOAT32   ,
    thrust_sort_CUPY_TYPE_FLOAT64   ,
    thrust_sort_CUPY_TYPE_COMPLEX64 ,
    thrust_sort_CUPY_TYPE_COMPLEX128,
    thrust_sort_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::LexSortTargetT> lexsort_targets[CUPY_NUM_TYPES] = {
    thrust_lexsort_CUPY_TYPE_INT8      ,
    thrust_lexsort_CUPY_TYPE_UINT8     ,
    thrust_lexsort_CUPY_TYPE_INT16     ,
    thrust_lexsort_CUPY_TYPE_UINT16    ,
    thrust_lexsort_CUPY_TYPE_INT32     ,
    thrust_lexsort_CUPY_TYPE_UINT32    ,
    thrust_lexsort_CUPY_TYPE_INT64     ,
    thrust_lexsort_CUPY_TYPE_UINT64    ,
    thrust_lexsort_CUPY_TYPE_FLOAT16   ,
    thrust_lexsort_CUPY_TYPE_FLOAT32   ,
    thrust_lexsort_CUPY_TYPE_FLOAT64   ,
    thrust_lexsort_CUPY_TYPE_COMPLEX64 ,
    thrust_lexsort_CUPY_TYPE_COMPLEX128,
    thrust_lexsort_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::ArgSortTargetT> argsort_targets[CUPY_NUM_TYPES] = {
    thrust_argsort_CUPY_TYPE_INT8      ,
    thrust_argsort_CUPY_TYPE_UINT8     ,
    thrust_argsort_CUPY_TYPE_INT16     ,
    thrust_argsort_CUPY_TYPE_UINT16    ,
    thrust_argsort_CUPY_TYPE_INT32     ,
    thrust_argsort_CUPY_TYPE_UINT32    ,
    thrust_argsort_CUPY_TYPE_INT64     ,
    thrust_argsort_CUPY_TYPE_UINT64    ,
    thrust_argsort_CUPY_TYPE_FLOAT16   ,
    thrust_argsort_CUPY_TYPE_FLOAT32   ,
    thrust_argsort_CUPY_TYPE_FLOAT64   ,
    thrust_argsort_CUPY_TYPE_COMPLEX64 ,
    thrust_argsort_CUPY_TYPE_COMPLEX128,
    thrust_argsort_CUPY_TYPE_BOOL      ,
  };
}  // namespace cupy


//
// APIs exposed to CuPy
//

/* -------- sort -------- */

void thrust_sort(int dtype_id,
                 void* data_start,
                 size_t* keys_start,
                 const std::vector<ptrdiff_t>& shape,
                 intptr_t stream, 
                 void* memory) {
    return cupy::sort_targets[dtype_id](data_start,
                                        keys_start,
                                        shape,
                                        stream,
                                        memory);
}

/* -------- lexsort -------- */
void thrust_lexsort(int dtype_id,
                    size_t* idx_start,
                    void* keys_start,
                    size_t k,
                    size_t n,
                    intptr_t stream,
                    void *memory) {
    return cupy::lexsort_targets[dtype_id](idx_start,
                                           keys_start,
                                           k,
                                           n,
                                           stream,
                                           memory);
}

/* -------- argsort -------- */
void thrust_argsort(int dtype_id,
                    size_t* idx_start,
                    void* data_start,
                    void* keys_start,
                    const std::vector<ptrdiff_t>& shape,
                    intptr_t stream,
                    void *memory) {
    return cupy::argsort_targets[dtype_id](idx_start,
                                           data_start,
                                           keys_start,
                                           shape,
                                           stream,
                                           memory);
}
