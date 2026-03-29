from __future__ import annotations

import string

import cupy as cp
import numpy as np


callback_name = 'my_complex_load_cb'

code = string.Template(r"""
#include <cufftXt.h>

struct cb_params {
    unsigned window_N;
    unsigned signal_size;
};

__device__ cufftComplex ${callback_name}(void *input,
                                         unsigned long long index,
                                         void *info,
                                         void *sharedmem) {
    const cb_params* params = static_cast<const cb_params*>(info);
    cufftComplex* cb_output = static_cast<cufftComplex*>(input);
    const unsigned sample   = index % params->signal_size;

    return (sample < params->window_N) ? cb_output[index] \
                                       : cufftComplex{0.f, 0.f};
}
""").substitute(callback_name=callback_name)


# Problem input parameters
batches = 830
signal_size = 328
window_size = 32
complex_signal_size = signal_size // 2 + 1

# Wave parameters
waves = 12
signal_max_A = 20.
signal_max_f = 500.
sampling_dt = 1e-3

# Precision threshold
threshold = 1e-6

# Initialize the input signal as a composite of sine waves
# with random amplitudes and frequencies
wave_amplitudes = signal_max_A * \
    cp.random.random((batches, waves), dtype=cp.float32)
wave_frequencies = signal_max_f * \
    cp.random.random((batches, waves), dtype=cp.float32)

# Compose the signal
input_signals = cp.empty((batches, signal_size), dtype=cp.float32)
time = 0.
for s in range(signal_size):
    input_signals[..., s] = cp.sum(
        wave_amplitudes[...] *
        cp.sin(2 * cp.pi * wave_frequencies[...] * time), axis=-1)
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
print("Transforming signal with rfft (cufftExecR2C)")
out = cp.fft.rfft(input_signals)

# Apply window via load callback and inverse-transform the signal
print("Transforming signal with irfft (cufftExecC2R)")
with cp.fft.config.set_cufft_callbacks(cb_load=code,
                                       cb_load_name=callback_name,
                                       cb_load_data=memptr_d,
                                       cb_ver='jit'):
    out = cp.fft.irfft(out)


# Compare against reference implementation
out_ref = cp.fft.rfft(input_signals)
out_ref[:, window_size:] = 0
out_ref = cp.fft.irfft(out_ref)
assert cp.allclose(out, out_ref)
