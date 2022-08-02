#include "../cupy_thrust.inl"


namespace cupy {

void thrust_sort_CUPY_TYPE_UINT32(void *data_start,
                            size_t *keys_start,
                            const std::vector<ptrdiff_t>& shape,
                            intptr_t stream,
                            void* memory) {
#if ( CUPY_TYPE_UINT32 != CUPY_TYPE_FLOAT16 )                        \
    || (( CUPY_TYPE_UINT32 == CUPY_TYPE_FLOAT16 )                    \
        && ((__CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__))  \
            || (defined(__HIPCC__) || defined(CUPY_USE_HIP))))

    _sort op;
    return dtype_forwarder< unsigned int >(op,
                                         data_start,
                                         keys_start,
                                         shape,
                                         stream,
                                         memory);

#endif
}

}  // namespace cupy
