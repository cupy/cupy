#include "../cupy_cub.inl"


namespace cupy {

void cub_device_histogram_even_CUPY_TYPE_INT64(void* workspace,
                                          size_t& workspace_size,
                                          void* x,
                                          void* y,
                                          int n_bins,
                                          int lower,
                                          int upper,
                                          size_t n_samples,
                                          cudaStream_t stream) {
#ifndef CUPY_USE_HIP
#if ( CUPY_TYPE_INT64 != CUPY_TYPE_FLOAT16 )                        \
    || (( CUPY_TYPE_INT64 == CUPY_TYPE_FLOAT16 )                    \
        && ((__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))  \
            || (defined(__HIPCC__) || defined(CUPY_USE_HIP))))

    _cub_histogram_even op;
    return dtype_forwarder< int64_t >(op,
                                         workspace,
                                         workspace_size,
                                         x,
                                         y,
                                         n_bins,
                                         lower,
                                         upper,
                                         n_samples,
                                         stream);

#endif
#endif  // CUPY_USE_HIP
}

}  // namespace cupy
