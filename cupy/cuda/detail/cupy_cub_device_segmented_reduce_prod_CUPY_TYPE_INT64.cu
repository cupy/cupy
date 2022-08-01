#include "../cupy_cub.inl"


namespace cupy {

void cub_device_segmented_reduce_prod_CUPY_TYPE_INT64(void* workspace,
                                                size_t& workspace_size,
                                                void* x,
                                                void* y,
                                                int num_segments,
                                                int segment_size,
                                                cudaStream_t stream) {
#if ( CUPY_TYPE_INT64 != CUPY_TYPE_FLOAT16 )                        \
    || (( CUPY_TYPE_INT64 == CUPY_TYPE_FLOAT16 )                    \
        && ((__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))  \
            || (defined(__HIPCC__) || defined(CUPY_USE_HIP))))

    // CUB internally use int for offset...
    // This iterates over [0, segment_size, 2*segment_size, 3*segment_size, ...]
    #ifndef CUPY_USE_HIP
    CountingInputIterator<int> count_itr(0);
    #else
    rocprim::counting_iterator<int> count_itr(0);
    #endif
    _arange scaling(segment_size);
    seg_offset_itr itr(count_itr, scaling);

    _cub_segmented_reduce_prod op;
    return dtype_forwarder< int64_t >(op,
                                         workspace,
                                         workspace_size,
                                         x,
                                         y,
                                         num_segments,
                                         itr,
                                         stream);
#endif
}

}  // namespace cupy
