#include "../cupy_thrust.inl"


namespace cupy{

void thrust_argsort_CUPY_TYPE_INT32(size_t *idx_start,
                               void *data_start,
                               void *keys_start,
                               const std::vector<ptrdiff_t>& shape, 
                               intptr_t stream,
                               void *memory) {
#if ( CUPY_TYPE_INT32 != CUPY_TYPE_FLOAT16 )                        \
    || (( CUPY_TYPE_INT32 == CUPY_TYPE_FLOAT16 )                    \
        && ((__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))  \
            || (defined(__HIPCC__) || defined(CUPY_USE_HIP))))

    _argsort op;
    return dtype_forwarder< int >(op, 
                                         idx_start,
                                         data_start,
                                         keys_start,
                                         shape,
                                         stream,
                                         memory);

#endif
}

}  // namespace cupy
