#include "../cupy_cub.inl"


namespace cupy {

void cub_device_reduce_min_CUPY_TYPE_BOOL(void* workspace,
                                      size_t& workspace_size,
                                      void* x,
                                      void* y,
                                      int num_items,
                                      cudaStream_t stream) {
#if ( CUPY_TYPE_BOOL != CUPY_TYPE_FLOAT16 )                        \
    || (( CUPY_TYPE_BOOL == CUPY_TYPE_FLOAT16 )                    \
        && ((__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))  \
            || (defined(__HIPCC__) || defined(CUPY_USE_HIP))))

    _cub_reduce_min op;
    return dtype_forwarder< bool >(op,
                                         workspace,
                                         workspace_size,
                                         x,
                                         y,
                                         num_items,
                                         stream);

#endif
}

}  // namespace cupy
