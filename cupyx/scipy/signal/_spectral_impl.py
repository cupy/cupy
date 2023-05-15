"""
Spectral analysis functions and utilities.

Some of the functions defined here were ported directly from CuSignal under
terms of the Apache license, under the following notice

Copyright (c) 2019-2020, NVIDIA CORPORATION.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cupy

from cupy_backends.cuda.api import runtime


def _get_raw_typename(dtype):
    return cupy.dtype(dtype).name


def _get_module_func_raw(module, func_name, *template_args):
    args_dtypes = [_get_raw_typename(arg.dtype) for arg in template_args]
    template = '_'.join(args_dtypes)
    kernel_name = f'{func_name}_{template}' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


if runtime.is_hip:
    KERNEL_BASE = r"""
    #include <hip/hip_runtime.h>
"""
else:
    KERNEL_BASE = r"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
"""

LOMBSCARGLE_KERNEL = KERNEL_BASE + r"""

///////////////////////////////////////////////////////////////////////////////
//                            LOMBSCARGLE                                    //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__device__ void _cupy_lombscargle_float( const int x_shape,
                                         const int freqs_shape,
                                         const T *__restrict__ x,
                                         const T *__restrict__ y,
                                         const T *__restrict__ freqs,
                                         T *__restrict__ pgram,
                                         const T *__restrict__ y_dot ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    T yD {};
    if ( y_dot[0] == 0 ) {
        yD = 1.0f;
    } else {
        yD = 2.0f / y_dot[0];
    }

    for ( int tid = tx; tid < freqs_shape; tid += stride ) {

        T freq { freqs[tid] };

        T xc {};
        T xs {};
        T cc {};
        T ss {};
        T cs {};
        T c {};
        T s {};

        for ( int j = 0; j < x_shape; j++ ) {
            sincosf( freq * x[j], &s, &c );
            xc += y[j] * c;
            xs += y[j] * s;
            cc += c * c;
            ss += s * s;
            cs += c * s;
        }

        T c_tau {};
        T s_tau {};
        T tau { atan2f( 2.0f * cs, cc - ss ) / ( 2.0f * freq ) };
        sincosf( freq * tau, &s_tau, &c_tau );
        T c_tau2 { c_tau * c_tau };
        T s_tau2 { s_tau * s_tau };
        T cs_tau { 2.0f * c_tau * s_tau };

        pgram[tid] = ( 0.5f * ( ( ( c_tau * xc + s_tau * xs ) *
                                  ( c_tau * xc + s_tau * xs ) /
                                  ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
                                ( ( c_tau * xs - s_tau * xc ) *
                                  ( c_tau * xs - s_tau * xc ) /
                                  ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) *
                     yD;
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_lombscargle_float32(
        const int x_shape, const int freqs_shape, const float *__restrict__ x,
        const float *__restrict__ y, const float *__restrict__ freqs,
        float *__restrict__ pgram, const float *__restrict__ y_dot ) {
    _cupy_lombscargle_float<float>( x_shape, freqs_shape, x, y,
                                    freqs, pgram, y_dot );
}

template<typename T>
__device__ void _cupy_lombscargle_double( const int x_shape,
                                          const int freqs_shape,
                                          const T *__restrict__ x,
                                          const T *__restrict__ y,
                                          const T *__restrict__ freqs,
                                          T *__restrict__ pgram,
                                          const T *__restrict__ y_dot ) {

    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };
    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };

    T yD {};
    if ( y_dot[0] == 0 ) {
        yD = 1.0;
    } else {
        yD = 2.0 / y_dot[0];
    }

    for ( int tid = tx; tid < freqs_shape; tid += stride ) {

        T freq { freqs[tid] };

        T xc {};
        T xs {};
        T cc {};
        T ss {};
        T cs {};
        T c {};
        T s {};

        for ( int j = 0; j < x_shape; j++ ) {

            sincos( freq * x[j], &s, &c );
            xc += y[j] * c;
            xs += y[j] * s;
            cc += c * c;
            ss += s * s;
            cs += c * s;
        }

        T c_tau {};
        T s_tau {};
        T tau { atan2( 2.0 * cs, cc - ss ) / ( 2.0 * freq ) };
        sincos( freq * tau, &s_tau, &c_tau );
        T c_tau2 { c_tau * c_tau };
        T s_tau2 { s_tau * s_tau };
        T cs_tau { 2.0 * c_tau * s_tau };

        pgram[tid] = ( 0.5 * ( ( ( c_tau * xc + s_tau * xs ) *
                                 ( c_tau * xc + s_tau * xs ) /
                                 ( c_tau2 * cc + cs_tau * cs + s_tau2 * ss ) ) +
                               ( ( c_tau * xs - s_tau * xc ) *
                                 ( c_tau * xs - s_tau * xc ) /
                                 ( c_tau2 * ss - cs_tau * cs + s_tau2 * cc ) ) ) ) *
                     yD;
    }
}

extern "C" __global__ void __launch_bounds__( 512 ) _cupy_lombscargle_float64(
        const int x_shape, const int freqs_shape, const double *__restrict__ x,
        const double *__restrict__ y, const double *__restrict__ freqs,
        double *__restrict__ pgram, const double *__restrict__ y_dot ) {

    _cupy_lombscargle_double<double>( x_shape, freqs_shape, x, y, freqs,
                                      pgram, y_dot );
}
"""  # NOQA


LOMBSCARGLE_MODULE = cupy.RawModule(
    code=LOMBSCARGLE_KERNEL, options=('-std=c++11',),
    name_expressions=['_cupy_lombscargle_float32',
                      '_cupy_lombscargle_float64'])


def _lombscargle(x, y, freqs, pgram, y_dot):
    device_id = cupy.cuda.Device()

    num_blocks = device_id.attributes["MultiProcessorCount"] * 20
    block_sz = 512
    lombscargle_kernel = _get_module_func_raw(
        LOMBSCARGLE_MODULE, '_cupy_lombscargle', x)

    args = (x.shape[0], freqs.shape[0], x, y, freqs, pgram, y_dot)
    lombscargle_kernel((num_blocks,), (block_sz,), args)
