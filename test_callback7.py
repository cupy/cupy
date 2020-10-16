import cupy as cp


code = r'''
__device__ cufftComplex CB_ConvertInputC(
    void *dataIn, 
    size_t offset, 
    void *callerInfo, 
    void *sharedPtr) 
{
    cufftComplex x;
    x.x = 1.;
    x.y = 0.;
    return x;
}

__device__ cufftCallbackLoadC d_loadCallbackPtr = CB_ConvertInputC;
'''
cp.fft.config.set_callbacks(cb_load=code)

# 1D
#a = cp.random.random((64, 128)).astype(cp.complex64)
#b = cp.fft.fft(a)
#c = cp.fft.fft(cp.ones(shape=a.shape, dtype=cp.complex64))

# ND
a = cp.random.random((64, 128, 128)).astype(cp.complex64)
b = cp.fft.fftn(a, axes=(1,2))
c = cp.fft.fftn(cp.ones(shape=a.shape, dtype=cp.complex64), axes=(1,2))

cp.fft.config.show_plan_cache_info()
assert cp.allclose(b, c)
