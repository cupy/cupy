#include "../cupy_cub.inl"


namespace cupy {

void cub_device_spmv_CUPY_TYPE_INT64(void* workspace,
                                size_t& workspace_size,
                                void* values,
                                void* row_offsets,
                                void* column_indices,
                                void* x,
                                void* y,
                                int num_rows,
                                int num_cols,
                                int num_nonzeros,
                                cudaStream_t stream) {
#ifndef CUPY_USE_HIP
#if ( CUPY_TYPE_INT64 != CUPY_TYPE_FLOAT16 )                        \
    || (( CUPY_TYPE_INT64 == CUPY_TYPE_FLOAT16 )                    \
        && ((__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))  \
            || (defined(__HIPCC__) || defined(CUPY_USE_HIP))))

    _cub_device_spmv op;
    return dtype_forwarder< int64_t >(op,
                                         workspace,
                                         workspace_size,
                                         values,
                                         row_offsets,
                                         column_indices,
                                         x,
                                         y,
                                         num_rows,
                                         num_cols,
                                         num_nonzeros,
                                         stream);

#endif
#endif  // CUPY_USE_HIP
}

}  // namespace cupy
