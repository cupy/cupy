
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime

from cupyx.scipy.signal._signaltools import lfilter


if runtime.is_hip:
    SYMIIR2_KERNEL = r"""#include <hip/hip_runtime.h>
"""
else:
    SYMIIR2_KERNEL = r"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
"""

SYMIIR2_KERNEL = SYMIIR2_KERNEL + r"""
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>

template<typename T>
__device__ T _compute_symiirorder2_fwd_hc(
        const int k, const T cs, const T r, const T omega) {
    T base;

    if(k < 0) {
        return 0;
    }

    if(omega == 0.0) {
        base = cs * pow(r, ((T) k)) * (k + 1);
    } else if(omega == M_PI) {
        base = cs * pow(r, ((T) k)) * (k + 1) * (1 - 2 * (k % 2));
    } else {
        base = (cs * pow(r, ((T) k)) * sin(omega * (k + 1)) /
                sin(omega));
    }
    return base;
}

template<typename T>
__global__ void compute_symiirorder2_fwd_sc(
        const int n, const int off, const T* cs_ptr, const T* r_ptr,
        const T* omega_ptr, const double precision, bool* valid, T* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx + off >= n) {
        return;
    }

    const T cs = cs_ptr[0];
    const T r = r_ptr[0];
    const T omega = omega_ptr[0];

    T val = _compute_symiirorder2_fwd_hc<T>(idx + off + 1, cs, r, omega);
    T err = val * val;

    out[idx] = val;
    valid[idx] = err <= precision;
}

template<typename T>
__device__ T _compute_symiirorder2_bwd_hs(
        const int ki, const T cs, const T rsq, const T omega) {
    T c0;
    T gamma;

    T cssq = cs * cs;
    int k = abs(ki);
    T rsupk = pow(rsq, ((T) k) / ((T) 2.0));


    if(omega == 0.0) {
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq;
        gamma = (1 - rsq) / (1 + rsq);
        return c0 * rsupk * (1 + gamma * k);
    }

    if(omega == M_PI) {
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq;
        gamma = (1 - rsq) / (1 + rsq) * (1 - 2 * (k % 2));
        return c0 * rsupk * (1 + gamma * k);
    }

    c0 = (cssq * (1.0 + rsq) / (1.0 - rsq) /
                (1 - 2 * rsq * cos(2 * omega) + rsq * rsq));
    gamma = (1.0 - rsq) / (1.0 + rsq) / tan(omega);
    return c0 * rsupk * (cos(omega * k) + gamma * sin(omega * k));
}

template<typename T>
__global__ void compute_symiirorder2_bwd_sc(
        const int n, const int off, const int l_off, const int r_off,
        const T* cs_ptr, const T* rsq_ptr, const T* omega_ptr,
        const double precision, bool* valid, T* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx + off >= n) {
        return;
    }

    const T cs = cs_ptr[0];
    const T rsq = rsq_ptr[0];
    const T omega = omega_ptr[0];

    T v1 = _compute_symiirorder2_bwd_hs<T>(idx + l_off + off, cs, rsq, omega);
    T v2 = _compute_symiirorder2_bwd_hs<T>(idx + r_off + off, cs, rsq, omega);

    T diff = v1 + v2;
    T err = diff * diff;
    out[idx] = diff;
    valid[idx] = err <= precision;
}
"""

SYMIIR2_MODULE = cupy.RawModule(
    code=SYMIIR2_KERNEL, options=('-std=c++11',),
    name_expressions=[f'compute_symiirorder2_bwd_sc<{t}>'
                      for t in ['float', 'double']] +
    [f'compute_symiirorder2_fwd_sc<{t}>'
     for t in ['float', 'double']])


def _get_module_func(module, func_name, *template_args):
    args_dtypes = [get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


def _find_initial_cond(all_valid, cum_poly, n, off=0):
    indices = cupy.where(all_valid)[0] + 1 + off
    zi = cupy.nan
    if indices.size > 0:
        zi = cupy.where(
            indices[0] >= n, cupy.nan, cum_poly[indices[0] - 1 - off])
    return zi


def symiirorder1(input, c0, z1, precision=-1.0):
    """
    Implement a smoothing IIR filter with mirror-symmetric boundary conditions
    using a cascade of first-order sections.  The second section uses a
    reversed sequence.  This implements a system with the following
    transfer function and mirror-symmetric boundary conditions::

                           c0
           H(z) = ---------------------
                   (1-z1/z) (1 - z1 z)

    The resulting signal will have mirror symmetric boundary conditions
    as well.

    Parameters
    ----------
    input : ndarray
        The input signal.
    c0, z1 : scalar
        Parameters in the transfer function.
    precision :
        Specifies the precision for calculating initial conditions
        of the recursive filter based on mirror-symmetric input.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """

    if cupy.abs(z1) >= 1:
        raise ValueError('|z1| must be less than 1.0')

    if precision <= 0.0 or precision > 1.0:
        precision = cupy.finfo(input.dtype).resolution

    precision *= precision
    pos = cupy.arange(1, input.size + 1, dtype=input.dtype)
    pow_z1 = z1 ** pos

    diff = pow_z1 * cupy.conjugate(pow_z1)
    cum_poly = cupy.cumsum(pow_z1 * input) + input[0]
    all_valid = diff <= precision

    zi = _find_initial_cond(all_valid, cum_poly, input.size)

    if cupy.isnan(zi):
        raise ValueError(
            'Sum to find symmetric boundary conditions did not converge.')

    # Apply first the system 1 / (1 - z1 * z^-1)
    y1, _ = lfilter(
        cupy.ones(1, dtype=input.dtype), cupy.r_[1, -z1], input[1:], zi=zi)
    y1 = cupy.r_[zi, y1]

    # Compute backward symmetric condition and apply the system
    # c0 / (1 - z1 * z)
    zi = -c0 / (z1 - 1.0) * y1[-1]
    out, _ = lfilter(c0, cupy.r_[1, -z1], y1[:-1][::-1], zi=zi)
    return cupy.r_[out[::-1], zi]


def _compute_symiirorder2_fwd_hc(k, cs, r, omega):
    base = None
    if omega == 0.0:
        base = cs * cupy.power(r, k) * (k+1)
    elif omega == cupy.pi:
        base = cs * cupy.power(r, k) * (k + 1) * (1 - 2 * (k % 2))
    else:
        base = (cs * cupy.power(r, k) * cupy.sin(omega * (k + 1)) /
                cupy.sin(omega))
    return cupy.where(k < 0, 0.0, base)


def _compute_symiirorder2_bwd_hs(k, cs, rsq, omega):
    cssq = cs * cs
    k = cupy.abs(k)
    rsupk = cupy.power(rsq, k / 2.0)

    if omega == 0.0:
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq
        gamma = (1 - rsq) / (1 + rsq)
        return c0 * rsupk * (1 + gamma * k)

    if omega == cupy.pi:
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq
        gamma = (1 - rsq) / (1 + rsq) * (1 - 2 * (k % 2))
        return c0 * rsupk * (1 + gamma * k)

    c0 = (cssq * (1.0 + rsq) / (1.0 - rsq) /
          (1 - 2 * rsq * cupy.cos(2 * omega) + rsq * rsq))
    gamma = (1.0 - rsq) / (1.0 + rsq) / cupy.tan(omega)
    return c0 * rsupk * (cupy.cos(omega * k) + gamma * cupy.sin(omega * k))


def symiirorder2(input, r, omega, precision=-1.0):
    """
    Implement a smoothing IIR filter with mirror-symmetric boundary conditions
    using a cascade of second-order sections.  The second section uses a
    reversed sequence.  This implements the following transfer function::

                                  cs^2
         H(z) = ---------------------------------------
                (1 - a2/z - a3/z^2) (1 - a2 z - a3 z^2 )

    where::

          a2 = 2 * r * cos(omega)
          a3 = - r ** 2
          cs = 1 - 2 * r * cos(omega) + r ** 2

    Parameters
    ----------
    input : ndarray
        The input signal.
    r, omega : float
        Parameters in the transfer function.
    precision : float
        Specifies the precision for calculating initial conditions
        of the recursive filter based on mirror-symmetric input.

    Returns
    -------
    output : ndarray
        The filtered signal.
    """
    if r >= 1.0:
        raise ValueError('r must be less than 1.0')

    if precision <= 0.0 or precision > 1.0:
        if input.dtype is cupy.dtype(cupy.float64):
            precision = 1e-11
        elif input.dtype is cupy.dtype(cupy.float32):
            precision = 1e-6
        else:
            precision = 10 ** -cupy.finfo(input.dtype).iexp

    block_sz = 128
    rsq = r * r
    a2 = 2 * r * cupy.cos(omega)
    a3 = -rsq
    cs = cupy.atleast_1d(1 - 2 * r * cupy.cos(omega) + rsq)
    omega = cupy.asarray(omega, cs.dtype)
    r = cupy.asarray(r, cs.dtype)
    rsq = cupy.asarray(rsq, cs.dtype)

    precision *= precision

    # First compute the symmetric forward starting conditions
    compute_symiirorder2_fwd_sc = _get_module_func(
        SYMIIR2_MODULE, 'compute_symiirorder2_fwd_sc', cs)

    diff = cupy.empty((block_sz + 1,), dtype=cs.dtype)
    all_valid = cupy.empty((block_sz + 1,), dtype=cupy.bool_)

    starting_diff = cupy.arange(2, dtype=input.dtype)
    starting_diff = _compute_symiirorder2_fwd_hc(starting_diff, cs, r, omega)

    y0 = cupy.nan
    y1 = cupy.nan

    for i in range(0, input.size + 2, block_sz):
        compute_symiirorder2_fwd_sc(
            (1,), (block_sz + 1,), (
                input.size + 2, i, cs, r, omega, precision, all_valid, diff))

        input_slice = input[i:i + block_sz]
        diff_y0 = diff[:-1][:input_slice.size]
        diff_y1 = diff[1:][:input_slice.size]

        if cupy.isnan(y0):
            cum_poly_y0 = cupy.cumsum(
                diff_y0 * input_slice) + starting_diff[0] * input[0]
            y0 = _find_initial_cond(
                all_valid[:-1][:input_slice.size], cum_poly_y0, input.size, i)

        if cupy.isnan(y1):
            cum_poly_y1 = (cupy.cumsum(diff_y1 * input_slice) +
                           starting_diff[0] * input[1] +
                           starting_diff[1] * input[0])
            y1 = _find_initial_cond(
                all_valid[1:][:input_slice.size], cum_poly_y1, input.size, i)

        if not cupy.any(cupy.isnan(cupy.r_[y0, y1])):
            break

    if cupy.any(cupy.isnan(cupy.r_[y0, y1])):
        raise ValueError(
            'Sum to find symmetric boundary conditions did not converge.')

    # Apply the system cs / (1 - a2 * z^-1 - a3 * z^-2)
    zi = cupy.r_[y0, y1]
    y_fwd, _ = lfilter(cs, cupy.r_[1, -a2, -a3], input[2:], zi=zi)
    y_fwd = cupy.r_[zi, y_fwd]

    # Then compute the symmetric backward starting conditions
    compute_symiirorder2_bwd_sc = _get_module_func(
        SYMIIR2_MODULE, 'compute_symiirorder2_bwd_sc', cs)

    diff = cupy.empty((block_sz,), dtype=cs.dtype)
    all_valid = cupy.empty((block_sz,), dtype=cupy.bool_)
    rev_input = input[::-1]
    y0 = cupy.nan

    for i in range(0, input.size + 1, block_sz):
        compute_symiirorder2_bwd_sc(
            (1,), (block_sz,), (
                input.size + 1, i, 0, 1, cs, cupy.asarray(rsq, cs.dtype),
                cupy.asarray(omega, cs.dtype), precision, all_valid, diff))

        input_slice = rev_input[i:i + block_sz]
        cum_poly_y0 = cupy.cumsum(diff[:input_slice.size] * input_slice)
        y0 = _find_initial_cond(
            all_valid[:input_slice.size], cum_poly_y0, input.size, i)
        if not cupy.isnan(y0):
            break

    if cupy.isnan(y0):
        raise ValueError(
            'Sum to find symmetric boundary conditions did not converge.')

    y1 = cupy.nan
    for i in range(0, input.size + 1, block_sz):
        compute_symiirorder2_bwd_sc(
            (1,), (block_sz,), (
                input.size + 1, i, -1, 2, cs, cupy.asarray(rsq, cs.dtype),
                cupy.asarray(omega, cs.dtype), precision, all_valid, diff))

        input_slice = rev_input[i:i + block_sz]
        cum_poly_y1 = cupy.cumsum(diff[:input_slice.size] * input_slice)
        y1 = _find_initial_cond(
            all_valid[:input_slice.size], cum_poly_y1, input.size, i)
        if not cupy.isnan(y1):
            break

    if cupy.isnan(y1):
        raise ValueError(
            'Sum to find symmetric boundary conditions did not converge.')

    # Apply the system cs / (1 - a2 * z^1 - a3 * z^2)
    zi = cupy.r_[y0, y1]
    out, _ = lfilter(cs, cupy.r_[1, -a2, -a3], y_fwd[:-2][::-1], zi=zi)
    return cupy.r_[out[::-1], zi[::-1]]
