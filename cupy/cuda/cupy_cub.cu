#include <functional>

#include <cupy/type_dispatcher.cuh>
#include "cupy_cub.h"  // need to make atomicAdd visible to CUB templates early
#include "cupy_cub.inl"


namespace cupy {

  std::function<cupy::DeviceReduceT> device_reduce_sum_targets[CUPY_NUM_TYPES] = {
    cub_device_reduce_sum_CUPY_TYPE_INT8      ,
    cub_device_reduce_sum_CUPY_TYPE_UINT8     ,
    cub_device_reduce_sum_CUPY_TYPE_INT16     ,
    cub_device_reduce_sum_CUPY_TYPE_UINT16    ,
    cub_device_reduce_sum_CUPY_TYPE_INT32     ,
    cub_device_reduce_sum_CUPY_TYPE_UINT32    ,
    cub_device_reduce_sum_CUPY_TYPE_INT64     ,
    cub_device_reduce_sum_CUPY_TYPE_UINT64    ,
    cub_device_reduce_sum_CUPY_TYPE_FLOAT16   ,
    cub_device_reduce_sum_CUPY_TYPE_FLOAT32   ,
    cub_device_reduce_sum_CUPY_TYPE_FLOAT64   ,
    cub_device_reduce_sum_CUPY_TYPE_COMPLEX64 ,
    cub_device_reduce_sum_CUPY_TYPE_COMPLEX128,
    cub_device_reduce_sum_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceReduceT> device_reduce_prod_targets[CUPY_NUM_TYPES] = {
    cub_device_reduce_prod_CUPY_TYPE_INT8      ,
    cub_device_reduce_prod_CUPY_TYPE_UINT8     ,
    cub_device_reduce_prod_CUPY_TYPE_INT16     ,
    cub_device_reduce_prod_CUPY_TYPE_UINT16    ,
    cub_device_reduce_prod_CUPY_TYPE_INT32     ,
    cub_device_reduce_prod_CUPY_TYPE_UINT32    ,
    cub_device_reduce_prod_CUPY_TYPE_INT64     ,
    cub_device_reduce_prod_CUPY_TYPE_UINT64    ,
    cub_device_reduce_prod_CUPY_TYPE_FLOAT16   ,
    cub_device_reduce_prod_CUPY_TYPE_FLOAT32   ,
    cub_device_reduce_prod_CUPY_TYPE_FLOAT64   ,
    cub_device_reduce_prod_CUPY_TYPE_COMPLEX64 ,
    cub_device_reduce_prod_CUPY_TYPE_COMPLEX128,
    cub_device_reduce_prod_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceReduceT> device_reduce_min_targets[CUPY_NUM_TYPES] = {
    cub_device_reduce_min_CUPY_TYPE_INT8      ,
    cub_device_reduce_min_CUPY_TYPE_UINT8     ,
    cub_device_reduce_min_CUPY_TYPE_INT16     ,
    cub_device_reduce_min_CUPY_TYPE_UINT16    ,
    cub_device_reduce_min_CUPY_TYPE_INT32     ,
    cub_device_reduce_min_CUPY_TYPE_UINT32    ,
    cub_device_reduce_min_CUPY_TYPE_INT64     ,
    cub_device_reduce_min_CUPY_TYPE_UINT64    ,
    cub_device_reduce_min_CUPY_TYPE_FLOAT16   ,
    cub_device_reduce_min_CUPY_TYPE_FLOAT32   ,
    cub_device_reduce_min_CUPY_TYPE_FLOAT64   ,
    cub_device_reduce_min_CUPY_TYPE_COMPLEX64 ,
    cub_device_reduce_min_CUPY_TYPE_COMPLEX128,
    cub_device_reduce_min_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceReduceT> device_reduce_max_targets[CUPY_NUM_TYPES] = {
    cub_device_reduce_max_CUPY_TYPE_INT8      ,
    cub_device_reduce_max_CUPY_TYPE_UINT8     ,
    cub_device_reduce_max_CUPY_TYPE_INT16     ,
    cub_device_reduce_max_CUPY_TYPE_UINT16    ,
    cub_device_reduce_max_CUPY_TYPE_INT32     ,
    cub_device_reduce_max_CUPY_TYPE_UINT32    ,
    cub_device_reduce_max_CUPY_TYPE_INT64     ,
    cub_device_reduce_max_CUPY_TYPE_UINT64    ,
    cub_device_reduce_max_CUPY_TYPE_FLOAT16   ,
    cub_device_reduce_max_CUPY_TYPE_FLOAT32   ,
    cub_device_reduce_max_CUPY_TYPE_FLOAT64   ,
    cub_device_reduce_max_CUPY_TYPE_COMPLEX64 ,
    cub_device_reduce_max_CUPY_TYPE_COMPLEX128,
    cub_device_reduce_max_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceReduceT> device_reduce_argmin_targets[CUPY_NUM_TYPES] = {
    cub_device_reduce_argmin_CUPY_TYPE_INT8      ,
    cub_device_reduce_argmin_CUPY_TYPE_UINT8     ,
    cub_device_reduce_argmin_CUPY_TYPE_INT16     ,
    cub_device_reduce_argmin_CUPY_TYPE_UINT16    ,
    cub_device_reduce_argmin_CUPY_TYPE_INT32     ,
    cub_device_reduce_argmin_CUPY_TYPE_UINT32    ,
    cub_device_reduce_argmin_CUPY_TYPE_INT64     ,
    cub_device_reduce_argmin_CUPY_TYPE_UINT64    ,
    cub_device_reduce_argmin_CUPY_TYPE_FLOAT16   ,
    cub_device_reduce_argmin_CUPY_TYPE_FLOAT32   ,
    cub_device_reduce_argmin_CUPY_TYPE_FLOAT64   ,
    cub_device_reduce_argmin_CUPY_TYPE_COMPLEX64 ,
    cub_device_reduce_argmin_CUPY_TYPE_COMPLEX128,
    cub_device_reduce_argmin_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceReduceT> device_reduce_argmax_targets[CUPY_NUM_TYPES] = {
    cub_device_reduce_argmax_CUPY_TYPE_INT8      ,
    cub_device_reduce_argmax_CUPY_TYPE_UINT8     ,
    cub_device_reduce_argmax_CUPY_TYPE_INT16     ,
    cub_device_reduce_argmax_CUPY_TYPE_UINT16    ,
    cub_device_reduce_argmax_CUPY_TYPE_INT32     ,
    cub_device_reduce_argmax_CUPY_TYPE_UINT32    ,
    cub_device_reduce_argmax_CUPY_TYPE_INT64     ,
    cub_device_reduce_argmax_CUPY_TYPE_UINT64    ,
    cub_device_reduce_argmax_CUPY_TYPE_FLOAT16   ,
    cub_device_reduce_argmax_CUPY_TYPE_FLOAT32   ,
    cub_device_reduce_argmax_CUPY_TYPE_FLOAT64   ,
    cub_device_reduce_argmax_CUPY_TYPE_COMPLEX64 ,
    cub_device_reduce_argmax_CUPY_TYPE_COMPLEX128,
    cub_device_reduce_argmax_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceSegmentedReduceT> device_segmented_reduce_sum_targets[CUPY_NUM_TYPES] = {
    cub_device_segmented_reduce_sum_CUPY_TYPE_INT8      ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_UINT8     ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_INT16     ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_UINT16    ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_INT32     ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_UINT32    ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_INT64     ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_UINT64    ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_FLOAT16   ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_FLOAT32   ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_FLOAT64   ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_COMPLEX64 ,
    cub_device_segmented_reduce_sum_CUPY_TYPE_COMPLEX128,
    cub_device_segmented_reduce_sum_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceSegmentedReduceT> device_segmented_reduce_prod_targets[CUPY_NUM_TYPES] = {
    cub_device_segmented_reduce_prod_CUPY_TYPE_INT8      ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_UINT8     ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_INT16     ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_UINT16    ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_INT32     ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_UINT32    ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_INT64     ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_UINT64    ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_FLOAT16   ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_FLOAT32   ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_FLOAT64   ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_COMPLEX64 ,
    cub_device_segmented_reduce_prod_CUPY_TYPE_COMPLEX128,
    cub_device_segmented_reduce_prod_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceSegmentedReduceT> device_segmented_reduce_min_targets[CUPY_NUM_TYPES] = {
    cub_device_segmented_reduce_min_CUPY_TYPE_INT8      ,
    cub_device_segmented_reduce_min_CUPY_TYPE_UINT8     ,
    cub_device_segmented_reduce_min_CUPY_TYPE_INT16     ,
    cub_device_segmented_reduce_min_CUPY_TYPE_UINT16    ,
    cub_device_segmented_reduce_min_CUPY_TYPE_INT32     ,
    cub_device_segmented_reduce_min_CUPY_TYPE_UINT32    ,
    cub_device_segmented_reduce_min_CUPY_TYPE_INT64     ,
    cub_device_segmented_reduce_min_CUPY_TYPE_UINT64    ,
    cub_device_segmented_reduce_min_CUPY_TYPE_FLOAT16   ,
    cub_device_segmented_reduce_min_CUPY_TYPE_FLOAT32   ,
    cub_device_segmented_reduce_min_CUPY_TYPE_FLOAT64   ,
    cub_device_segmented_reduce_min_CUPY_TYPE_COMPLEX64 ,
    cub_device_segmented_reduce_min_CUPY_TYPE_COMPLEX128,
    cub_device_segmented_reduce_min_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceSegmentedReduceT> device_segmented_reduce_max_targets[CUPY_NUM_TYPES] = {
    cub_device_segmented_reduce_max_CUPY_TYPE_INT8      ,
    cub_device_segmented_reduce_max_CUPY_TYPE_UINT8     ,
    cub_device_segmented_reduce_max_CUPY_TYPE_INT16     ,
    cub_device_segmented_reduce_max_CUPY_TYPE_UINT16    ,
    cub_device_segmented_reduce_max_CUPY_TYPE_INT32     ,
    cub_device_segmented_reduce_max_CUPY_TYPE_UINT32    ,
    cub_device_segmented_reduce_max_CUPY_TYPE_INT64     ,
    cub_device_segmented_reduce_max_CUPY_TYPE_UINT64    ,
    cub_device_segmented_reduce_max_CUPY_TYPE_FLOAT16   ,
    cub_device_segmented_reduce_max_CUPY_TYPE_FLOAT32   ,
    cub_device_segmented_reduce_max_CUPY_TYPE_FLOAT64   ,
    cub_device_segmented_reduce_max_CUPY_TYPE_COMPLEX64 ,
    cub_device_segmented_reduce_max_CUPY_TYPE_COMPLEX128,
    cub_device_segmented_reduce_max_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceScanT> device_scan_cumsum_targets[CUPY_NUM_TYPES] = {
    cub_device_scan_cumsum_CUPY_TYPE_INT8      ,
    cub_device_scan_cumsum_CUPY_TYPE_UINT8     ,
    cub_device_scan_cumsum_CUPY_TYPE_INT16     ,
    cub_device_scan_cumsum_CUPY_TYPE_UINT16    ,
    cub_device_scan_cumsum_CUPY_TYPE_INT32     ,
    cub_device_scan_cumsum_CUPY_TYPE_UINT32    ,
    cub_device_scan_cumsum_CUPY_TYPE_INT64     ,
    cub_device_scan_cumsum_CUPY_TYPE_UINT64    ,
    cub_device_scan_cumsum_CUPY_TYPE_FLOAT16   ,
    cub_device_scan_cumsum_CUPY_TYPE_FLOAT32   ,
    cub_device_scan_cumsum_CUPY_TYPE_FLOAT64   ,
    cub_device_scan_cumsum_CUPY_TYPE_COMPLEX64 ,
    cub_device_scan_cumsum_CUPY_TYPE_COMPLEX128,
    cub_device_scan_cumsum_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceScanT> device_scan_cumprod_targets[CUPY_NUM_TYPES] = {
    cub_device_scan_cumprod_CUPY_TYPE_INT8      ,
    cub_device_scan_cumprod_CUPY_TYPE_UINT8     ,
    cub_device_scan_cumprod_CUPY_TYPE_INT16     ,
    cub_device_scan_cumprod_CUPY_TYPE_UINT16    ,
    cub_device_scan_cumprod_CUPY_TYPE_INT32     ,
    cub_device_scan_cumprod_CUPY_TYPE_UINT32    ,
    cub_device_scan_cumprod_CUPY_TYPE_INT64     ,
    cub_device_scan_cumprod_CUPY_TYPE_UINT64    ,
    cub_device_scan_cumprod_CUPY_TYPE_FLOAT16   ,
    cub_device_scan_cumprod_CUPY_TYPE_FLOAT32   ,
    cub_device_scan_cumprod_CUPY_TYPE_FLOAT64   ,
    cub_device_scan_cumprod_CUPY_TYPE_COMPLEX64 ,
    cub_device_scan_cumprod_CUPY_TYPE_COMPLEX128,
    cub_device_scan_cumprod_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceSpmvT> device_spmv_targets[CUPY_NUM_TYPES] = {
    cub_device_spmv_CUPY_TYPE_INT8      ,
    cub_device_spmv_CUPY_TYPE_UINT8     ,
    cub_device_spmv_CUPY_TYPE_INT16     ,
    cub_device_spmv_CUPY_TYPE_UINT16    ,
    cub_device_spmv_CUPY_TYPE_INT32     ,
    cub_device_spmv_CUPY_TYPE_UINT32    ,
    cub_device_spmv_CUPY_TYPE_INT64     ,
    cub_device_spmv_CUPY_TYPE_UINT64    ,
    cub_device_spmv_CUPY_TYPE_FLOAT16   ,
    cub_device_spmv_CUPY_TYPE_FLOAT32   ,
    cub_device_spmv_CUPY_TYPE_FLOAT64   ,
    cub_device_spmv_CUPY_TYPE_COMPLEX128,
    cub_device_spmv_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceHistRangeT> device_histogram_range_targets[CUPY_NUM_TYPES] = {
    cub_device_histogram_range_CUPY_TYPE_INT8      ,
    cub_device_histogram_range_CUPY_TYPE_UINT8     ,
    cub_device_histogram_range_CUPY_TYPE_INT16     ,
    cub_device_histogram_range_CUPY_TYPE_UINT16    ,
    cub_device_histogram_range_CUPY_TYPE_INT32     ,
    cub_device_histogram_range_CUPY_TYPE_UINT32    ,
    cub_device_histogram_range_CUPY_TYPE_INT64     ,
    cub_device_histogram_range_CUPY_TYPE_UINT64    ,
    cub_device_histogram_range_CUPY_TYPE_FLOAT16   ,
    cub_device_histogram_range_CUPY_TYPE_FLOAT32   ,
    cub_device_histogram_range_CUPY_TYPE_FLOAT64   ,
    nullptr,
    nullptr,
    cub_device_histogram_range_CUPY_TYPE_BOOL      ,
  };

  std::function<cupy::DeviceHistEvenT> device_histogram_even_targets[CUPY_NUM_TYPES] = {
    cub_device_histogram_even_CUPY_TYPE_INT8      ,
    cub_device_histogram_even_CUPY_TYPE_UINT8     ,
    cub_device_histogram_even_CUPY_TYPE_INT16     ,
    cub_device_histogram_even_CUPY_TYPE_UINT16    ,
    cub_device_histogram_even_CUPY_TYPE_INT32     ,
    cub_device_histogram_even_CUPY_TYPE_UINT32    ,
    cub_device_histogram_even_CUPY_TYPE_INT64     ,
    cub_device_histogram_even_CUPY_TYPE_UINT64    ,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    cub_device_histogram_even_CUPY_TYPE_BOOL      ,
  };

}  // namespace cupy


//
// APIs exposed to CuPy
//

/* -------- device reduce -------- */

void cub_device_reduce(void* workspace, size_t& workspace_size, void* x, void* y,
    int num_items, cudaStream_t stream, int op, int dtype_id)
{
    switch(op) {
    case CUPY_CUB_SUM:      return cupy::device_reduce_sum_targets[dtype_id](
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_PROD:      return cupy::device_reduce_prod_targets[dtype_id](
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_MIN:      return cupy::device_reduce_min_targets[dtype_id](
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_MAX:      return cupy::device_reduce_max_targets[dtype_id](
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_ARGMIN:      return cupy::device_reduce_argmin_targets[dtype_id](
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_ARGMAX:      return cupy::device_reduce_argmax_targets[dtype_id](
                                workspace, workspace_size, x, y, num_items, stream);
    default:            throw std::runtime_error("Unsupported operation");
    }
}

size_t cub_device_reduce_get_workspace_size(void* x, void* y, int num_items,
    cudaStream_t stream, int op, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_reduce(NULL, workspace_size, x, y, num_items, stream,
                      op, dtype_id);
    return workspace_size;
}

/* -------- device segmented reduce -------- */

void cub_device_segmented_reduce(void* workspace, size_t& workspace_size,
    void* x, void* y, int num_segments, int segment_size,
    cudaStream_t stream, int op, int dtype_id)
{
    switch(op) {
      case CUPY_CUB_SUM:
          return cupy::device_segmented_reduce_sum_targets[dtype_id](
                     workspace, workspace_size, x, y, num_segments, segment_size, stream);
      case CUPY_CUB_PROD:
          return cupy::device_segmented_reduce_prod_targets[dtype_id](
                     workspace, workspace_size, x, y, num_segments, segment_size, stream);
      case CUPY_CUB_MIN:
          return cupy::device_segmented_reduce_min_targets[dtype_id](
                     workspace, workspace_size, x, y, num_segments, segment_size, stream);
      case CUPY_CUB_MAX:
          return cupy::device_segmented_reduce_max_targets[dtype_id](
                     workspace, workspace_size, x, y, num_segments, segment_size, stream);
    default:
        throw std::runtime_error("Unsupported operation");
    }
}

size_t cub_device_segmented_reduce_get_workspace_size(void* x, void* y,
    int num_segments, int segment_size,
    cudaStream_t stream, int op, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_segmented_reduce(NULL, workspace_size, x, y,
                                num_segments, segment_size, stream,
                                op, dtype_id);
    return workspace_size;
}

/*--------- device spmv (sparse-matrix dense-vector multiply) ---------*/

void cub_device_spmv(void* workspace, size_t& workspace_size, void* values,
    void* row_offsets, void* column_indices, void* x, void* y, int num_rows,
    int num_cols, int num_nonzeros, cudaStream_t stream,
    int dtype_id)
{
    return cupy::device_spmv_targets[dtype_id](
        workspace, workspace_size, values, row_offsets, column_indices,
        x, y, num_rows, num_cols, num_nonzeros, stream);
}

size_t cub_device_spmv_get_workspace_size(void* values, void* row_offsets,
    void* column_indices, void* x, void* y, int num_rows, int num_cols,
    int num_nonzeros, cudaStream_t stream, int dtype_id)
{
    size_t workspace_size = 0;
    #ifndef CUPY_USE_HIP
    cub_device_spmv(NULL, workspace_size, values, row_offsets, column_indices,
                    x, y, num_rows, num_cols, num_nonzeros, stream, dtype_id);
    #endif
    return workspace_size;
}

/* -------- device scan -------- */

void cub_device_scan(void* workspace, size_t& workspace_size, void* x, void* y,
    int num_items, cudaStream_t stream, int op, int dtype_id)
{
    switch(op) {
    case CUPY_CUB_CUMSUM:
        return cupy::device_scan_cumsum_targets[dtype_id](
                                workspace, workspace_size, x, y, num_items, stream);
    case CUPY_CUB_CUMPROD:
        return cupy::device_scan_cumprod_targets[dtype_id](
                                workspace, workspace_size, x, y, num_items, stream);
    default:
        throw std::runtime_error("Unsupported operation");
    }
}

size_t cub_device_scan_get_workspace_size(void* x, void* y, int num_items,
    cudaStream_t stream, int op, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_scan(NULL, workspace_size, x, y, num_items, stream,
                    op, dtype_id);
    return workspace_size;
}

/* -------- device histogram -------- */

void cub_device_histogram_range(void* workspace, size_t& workspace_size, void* x, void* y,
    int n_bins, void* bins, size_t n_samples, cudaStream_t stream, int dtype_id)
{
    // TODO(leofang): support complex
    if (dtype_id == CUPY_TYPE_COMPLEX64 || dtype_id == CUPY_TYPE_COMPLEX128) {
	    throw std::runtime_error("complex dtype is not yet supported");
    }

    // TODO(leofang): n_samples is of type size_t, but if it's < 2^31 we cast it to int later
    return cupy::device_histogram_range_targets[dtype_id](
                            workspace, workspace_size, x, y, n_bins, bins, n_samples, stream);
}

size_t cub_device_histogram_range_get_workspace_size(void* x, void* y, int n_bins,
    void* bins, size_t n_samples, cudaStream_t stream, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_histogram_range(NULL, workspace_size, x, y, n_bins, bins, n_samples,
                               stream, dtype_id);
    return workspace_size;
}

void cub_device_histogram_even(void* workspace, size_t& workspace_size, void* x, void* y,
    int n_bins, int lower, int upper, size_t n_samples, cudaStream_t stream, int dtype_id)
{
    return cupy::device_histogram_even_targets[dtype_id](
                            workspace, workspace_size, x, y, n_bins, lower, upper, n_samples, stream);
}

size_t cub_device_histogram_even_get_workspace_size(void* x, void* y, int n_bins,
    int lower, int upper, size_t n_samples, cudaStream_t stream, int dtype_id)
{
    size_t workspace_size = 0;
    cub_device_histogram_even(NULL, workspace_size, x, y, n_bins, lower, upper, n_samples,
                              stream, dtype_id);
    return workspace_size;
}
