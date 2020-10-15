import os
import string
import subprocess

import cupy as cp
from cupyx.scipy.fftpack import get_fft_plan



code = r'''
#include <cufft.h>
#include <cufftXt.h>


__device__ cufftComplex CB_ConvertInputC(
    void *dataIn, 
    size_t offset, 
    void *callerInfo, 
    void *sharedPtr) 
{
    return ((cufftComplex*)dataIn)[offset];
}

__device__ 
cufftCallbackLoadC d_loadCallbackPtr = CB_ConvertInputC;
'''

mod = cp.RawModule(code=code,
                   backend='nvcc',
                   options=('-lcufft_static', '-lculibos', '-dc',))
#sym = mod.get_global('CB_ConvertInputC')
sym = mod.get_global('d_loadCallbackPtr')
print(sym.ptr)
a = cp.random.random((64, 64, 64)).astype(cp.complex64)
plan = get_fft_plan(a, axes=(1,2))
mgr = cp.cuda.cufft.CallbackManager()
mgr.set_callback(plan, sym.ptr, 0)
