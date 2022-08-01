#include "../cupy_cub.inl"


namespace cupy {

void cub_device_histogram_range_CUPY_TYPE_INT32(void* workspace,
                                           size_t& workspace_size,
                                           void* x,
                                           void* y,
                                           int n_bins,
                                           void* bins,
                                           size_t n_samples,
                                           cudaStream_t stream) {
#if ( CUPY_TYPE_INT32 != CUPY_TYPE_FLOAT16 )                        \
    || (( CUPY_TYPE_INT32 == CUPY_TYPE_FLOAT16 )                    \
        && ((__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))  \
            || (defined(__HIPCC__) || defined(CUPY_USE_HIP))))

    // TODO(leofang): n_samples is of type size_t, but if it's < 2^31 we cast it to int later
    _cub_histogram_range op;
    return dtype_forwarder< int >(op,
                                         workspace,
                                         workspace_size,
                                         x,
                                         y,
                                         n_bins,
                                         bins,
                                         n_samples,
                                         stream);

#endif
}

}  // namespace cupy
