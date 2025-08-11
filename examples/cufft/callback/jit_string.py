import string

import cupy as cp


# a load callback that overwrites the input array to 1
callback_name = 'cufftJITCallbackLoadComplex'

code = string.Template(r'''
#include <cufftXt.h>

__device__ cufftComplex ${callback_name}(
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
''').substitute(callback_name=callback_name)

a = cp.random.random((64, 128, 128)).astype(cp.complex64)

# this fftn call uses callback
with cp.fft.config.set_cufft_callbacks(
        cb_load=code, cb_load_name=callback_name, cb_ver='jit'):
    b = cp.fft.fftn(a, axes=(1, 2))

# this does not use
c = cp.fft.fftn(cp.ones(shape=a.shape, dtype=cp.complex64), axes=(1, 2))

# result agrees
assert cp.allclose(b, c)

# "static" plans are also cached, but are distinct from their no-callback
# counterparts
cp.fft.config.get_plan_cache().show_info()
