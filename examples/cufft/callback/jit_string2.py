import string

import cupy as cp


# a load callback that overwrites the input array to 1
callback_name = 'my_complex_load_cb'

code = string.Template(r'''
#include <cufftXt.h>

__device__ cufftComplex ${callback_name}(
    void *dataIn,
    unsigned long long offset,
    void *callerInfo,
    void *sharedPtr)
{
    cufftComplex x;
    x.x = 1.;
    x.y = 0.;
    return x;
}
''').substitute(callback_name=callback_name)

a = cp.random.random((128,)).astype(cp.complex64)

# this fft call uses callback
with cp.fft.config.set_cufft_callbacks(
        cb_load=code, cb_load_name=callback_name, cb_ver='jit'):
    b = cp.fft.fft(a)

# this does not use
c = cp.fft.fft(cp.ones_like(a))

# result agrees
assert cp.allclose(b, c)

# "static" plans are also cached, but are distinct from their no-callback
# counterparts
cp.fft.config.get_plan_cache().show_info()
