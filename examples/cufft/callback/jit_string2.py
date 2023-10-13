import os

import cupy as cp

# a load callback that overwrites the input array to 1
code = r'''
#include <cufftXt.h>

__device__ cufftComplex cufftJITCallbackLoadComplex(
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
'''

a = cp.random.random((128,)).astype(cp.complex64)

# this fft call uses callback
with cp.fft.config.set_cufft_callbacks(cb_load=code, cb_ver='jit'):
    b = cp.fft.fft(a)

# this does not use
c = cp.fft.fft(cp.ones_like(a))

# result agrees
assert cp.allclose(b, c)

# "static" plans are also cached, but are distinct from their no-callback counterparts
cp.fft.config.get_plan_cache().show_info()
