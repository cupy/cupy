import os

import cupy as cp
import numpy as np

# TODO(leofang): update these imports once they're upstreamed
from numba import cuda, types
from numba.core.extending import (make_attribute_wrapper, models,
                                  register_model)
from numba_ltoir import compile_ltoir


# User code

class cb_params:
    def __init__(self, window_N, signal_size):
        self.window_N = window_N
        self.signal_size = signal_size


# def cufftJITCallbackLoadComplex(cb_input, index, info, sharedmem):
def _Z27cufftJITCallbackLoadComplexPvmS_S_(cb_input, index, info, sharedmem):
    params = info[0]
    sample = index % params.signal_size

    if sample < params.window_N:
        return cb_input[index]
    else:
        return np.complex64(0 + 0j)


# Numba extensions

class CbParamsType(types.Type):
    def __init__(self):
        super().__init__(name='cb_params')


cb_params_type = CbParamsType()


@register_model(CbParamsType)
class CbParamsModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('window_N', types.uint32),
            ('signal_size', types.uint32),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(CbParamsType, 'window_N', 'window_N')
make_attribute_wrapper(CbParamsType, 'signal_size', 'signal_size')


# Code to compile the function using the extensions
cufftComplexPointer = types.CPointer(types.complex64)
signature = (
    cufftComplexPointer,
    types.uint64,
    types.CPointer(cb_params_type),
    types.voidptr
)

# Generate LTO IR for the user callback
ltoir = compile_ltoir(
        _Z27cufftJITCallbackLoadComplexPvmS_S_, signature, device=True)

# Problem input parameters
batches             = 830
signal_size         = 328
window_size         =  32
complex_signal_size = signal_size // 2 + 1

# Wave parameters
waves               = 12
signal_max_A        = 20.
signal_max_f        = 500.
sampling_dt         = 1e-3

# Precision threshold
threshold = 1e-6;

# Initialize the input signal as a composite of sine waves
# with random amplitudes and frequencies
wave_amplitudes = signal_max_A * cp.random.random((batches, waves), dtype=cp.float32)
wave_frequencies = signal_max_f * cp.random.random((batches, waves), dtype=cp.float32)

# Compose the signal
input_signals = cp.empty((batches, signal_size), dtype=cp.float32)
time = 0.
for s in range(signal_size):
    input_signals[..., s] = cp.sum(wave_amplitudes[...] * cp.sin(2 * cp.pi * wave_frequencies[...] * time), axis=-1)
    time += sampling_dt

# Define a structure used to pass in the window size
# Note: we are cheating here for itemsize! For a rigorous treatment, please
# refer to examples/custom_struct/complex_struct.py.
cb_params_dtype = np.dtype(
    {'names': ('window_N', 'signal_size'),
     'formats': (np.uint32, np.uint32),
     'itemsize': 8,
    }, align=True
)
cb_params_h = np.empty(1, dtype=cb_params_dtype)
cb_params_h['window_N'] = window_size
cb_params_h['signal_size'] = complex_signal_size

# Allocate and copy callback parameters from host to GPU
# Note: we don't just do cp.asarray(cb_params_h) because CuPy does not yet
# support structured dtypes.
memptr_d = cp.cuda.alloc(cb_params_h.nbytes)
memptr_d.copy_from_host(cb_params_h.ctypes.data, cb_params_h.nbytes)

# Transform signal forward
print("Transforming signal with rfft (cufftExecR2C)");
out = cp.fft.rfft(input_signals)

# Apply window via load callback and inverse-transform the signal
print("Transforming signal with irfft (cufftExecC2R)");
with cp.fft.config.set_cufft_callbacks(cb_load=ltoir,
                                       cb_load_data=memptr_d,
                                       cb_ver='jit'):
    out = cp.fft.irfft(out)


# Compare against reference implementation
out_ref = cp.fft.rfft(input_signals)
out_ref[:, window_size:] = 0
out_ref = cp.fft.irfft(out_ref)
assert cp.allclose(out, out_ref)
