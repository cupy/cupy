#include "../cupy_thrust.inl"


namespace cupy {

void thrust_lexsort_CUPY_TYPE_COMPLEX64(size_t *idx_start,
                               void *keys_start,
                               size_t k,
                               size_t n,
                               intptr_t stream,
                               void *memory) {
#if ( CUPY_TYPE_COMPLEX64 != CUPY_TYPE_FLOAT16 )                        \
    || (( CUPY_TYPE_COMPLEX64 == CUPY_TYPE_FLOAT16 )                    \
        && ((__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))  \
            || (defined(__HIPCC__) || defined(CUPY_USE_HIP))))

    _lexsort op;
    return dtype_forwarder< complex<float> >(op,
                                         idx_start,
                                         keys_start,
                                         k,
                                         n,
                                         stream,
                                         memory);

#endif
}

}  // namespace cupy
