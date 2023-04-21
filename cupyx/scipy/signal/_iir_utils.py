
from itertools import product

import cupy
from cupy._core.internal import _normalize_axis_index
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime


def _get_typename(dtype):
    typename = get_typename(dtype)
    if cupy.dtype(dtype).kind == 'c':
        typename = 'thrust::' + typename
    elif typename == 'float16':
        if runtime.is_hip:
            # 'half' in name_expressions weirdly raises
            # HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID in getLoweredName() on
            # ROCm
            typename = '__half'
        else:
            typename = 'half'
    return typename


FLOAT_TYPES = [cupy.float16, cupy.float32, cupy.float64]
INT_TYPES = [cupy.int8, cupy.int16, cupy.int32, cupy.int64]
COMPLEX_TYPES = [cupy.complex64, cupy.complex128]
UNSIGNED_TYPES = [cupy.uint8, cupy.uint16, cupy.uint32, cupy.uint64]
TYPES = FLOAT_TYPES + INT_TYPES + UNSIGNED_TYPES + COMPLEX_TYPES  # type: ignore  # NOQA
TYPE_PAIRS = [(x, y) for x, y in product(TYPES, TYPES)
              if cupy.promote_types(x, y) is cupy.dtype(x)]

TYPE_NAMES = [_get_typename(t) for t in TYPES]
TYPE_PAIR_NAMES = [(_get_typename(x), _get_typename(y)) for x, y in TYPE_PAIRS]


if runtime.is_hip:
    IIR_KERNEL = r"""#include <hip/hip_runtime.h>
"""
else:
    IIR_KERNEL = r"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
"""

IIR_KERNEL = IIR_KERNEL + r"""
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

template<typename U, typename T>
__global__ void compute_correction_factors(
        const int m, const int k, const T* b, U* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= k) {
        return;
    }

    U* out_start = out + idx * (k + m);
    U* out_off = out_start + k;

    for(int i = 0; i < m; i++) {
        U acc = 0.0;
        for(int j = 0; j < k; j++) {
            acc += ((U) b[j]) * out_off[i - j - 1];

        }
        out_off[i] = acc;
    }
}

template<typename T>
__global__ void first_pass_iir(
        const int m, const int k, const int n, const int n_blocks,
        const int carries_stride, const T* factors, T* out,
        T* carries) {
    int orig_idx = blockDim.x * (blockIdx.x % n_blocks) + threadIdx.x;

    int num_row = blockIdx.x / n_blocks;
    int idx = 2 * orig_idx + 1;

    if(idx >= n) {
        return;
    }

    int group_num = idx / m;
    int group_pos = idx % m;

    T* out_off = out + num_row * n;
    T* carries_off = carries + num_row * carries_stride;

    T* group_start = out_off + m * group_num;
    T* group_carries = carries_off + k * group_num;

    int pos = group_pos;
    int up_bound = pos;
    int low_bound = pos;
    int rel_pos;

    for(int level = 1, iter = 1; level < m; level *=2, iter++) {
        int sz = min(pow(2.0f, ((float) iter)), ((float) m));

        if(level > 1) {
            int factor = ceil(pos / ((float) sz));
            up_bound = sz * factor - 1;
            low_bound = up_bound - level + 1;
        }

        if(level == 1) {
            pos = low_bound;
        }

        if(pos < low_bound) {
            pos += level / 2;
        }

        if(pos + m * group_num >= n) {
            break;
        }

        rel_pos = pos % level;
        T carry = 0.0;
        for(int i = 1; i <= min(k, level); i++) {
            T k_value = group_start[low_bound - i];
            const T* k_factors = factors + (m + k) * (i - 1) + k;
            T factor = k_factors[rel_pos];
            carry += k_value * factor;
        }

        group_start[pos] += carry;
        __syncthreads();
    }

    if(pos >= m - k) {
        if(carries != NULL) {
            group_carries[pos - (m - k)] = group_start[pos];
        }
    }

}

template<typename T>
__global__ void correct_carries(
    const int m, const int k, const int n_blocks, const int carries_stride,
    const int offset, const T* factors, T* carries) {

    int idx = threadIdx.x;
    int pos = idx + (m - k);
    T* row_carries = carries + carries_stride * blockIdx.x;

    for(int i = offset; i < n_blocks; i++) {
        T* this_carries = row_carries + k * (i + (1 - offset));
        T* prev_carries = row_carries + k * (i - offset);

        T carry = 0.0;
        for(int j = 1; j <= k; j++) {
            const T* k_factors = factors + (m + k) * (j - 1) + k;
            T factor = k_factors[pos];
            T k_value = prev_carries[k - j];
            carry += factor * k_value;
        }

        this_carries[idx] += carry;
        __syncthreads();
    }
}

template<typename T>
__global__ void second_pass_iir(
        const int m, const int k, const int n, const int carries_stride,
        const int n_blocks, const int offset, const T* factors,
        T* carries, T* out) {

    int idx = blockDim.x * (blockIdx.x % n_blocks) + threadIdx.x;
    idx += offset * m;

    int row_num = blockIdx.x / n_blocks;
    int n_group = idx / m;
    int pos = idx % m;

    if(idx >= n) {
        return;
    }

    T* out_off = out + row_num * n;
    T* carries_off = carries + row_num * carries_stride;
    const T* prev_carries = carries_off + (n_group - offset) * k;

    T carry = 0.0;
    for(int i = 1; i <= k; i++) {
        const T* k_factors = factors + (m + k) * (i - 1) + k;
        T factor = k_factors[pos];
        T k_value = prev_carries[k - i];
        carry += factor * k_value;
    }

    out_off[idx] += carry;
}

template<typename T>
__global__ void first_pass_iir_sos(
        const int m, const int k, const int n, const int n_blocks,
        const int carries_stride, const T* factors, T* out,
        T* carries) {
    int orig_idx = blockDim.x * (blockIdx.x % n_blocks) + threadIdx.x;

    int num_row = blockIdx.x / n_blocks;
    int idx = 2 * orig_idx + 1;

    if(idx >= n) {
        return;
    }

    int group_num = idx / m;
    int group_pos = idx % m;

    T* out_off = out + num_row * n;
    T* carries_off = carries + num_row * carries_stride;

    T* group_start = out_off + m * group_num;
    T* group_carries = carries_off + k * group_num;

    int pos = group_pos;
    int up_bound = pos;
    int low_bound = pos;
    int rel_pos;

    for(int level = 1, iter = 1; level < m; level *=2, iter++) {
        int sz = min(pow(2.0f, ((float) iter)), ((float) m));

        if(level > 1) {
            int factor = ceil(pos / ((float) sz));
            up_bound = sz * factor - 1;
            low_bound = up_bound - level + 1;
        }

        if(level == 1) {
            pos = low_bound;
        }

        if(pos < low_bound) {
            pos += level / 2;
        }

        if(pos + m * group_num >= n) {
            break;
        }

        rel_pos = pos % level;
        T carry = 0.0;
        for(int i = 1; i <= min(k, level); i++) {
            T k_value = group_start[low_bound - i];
            const T* k_factors = factors + (m + k) * (i - 1) + k;
            T factor = k_factors[rel_pos];
            carry += k_value * factor;
        }

        group_start[pos] += carry;
        __syncthreads();
    }

    if(pos >= m - k) {
        if(carries != NULL) {
            group_carries[pos - (m - k)] = group_start[pos];
        }
    }

}
"""

IIR_MODULE = cupy.RawModule(
    code=IIR_KERNEL, options=('-std=c++11',),
    name_expressions=[f'compute_correction_factors<{x}, {y}>'
                      for x, y in TYPE_PAIR_NAMES] +
                     [f'correct_carries<{x}>' for x in TYPE_NAMES] +
                     [f'first_pass_iir<{x}>' for x in TYPE_NAMES] +
                     [f'second_pass_iir<{x}>' for x in TYPE_NAMES])


def _get_module_func(module, func_name, *template_args):
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


def collapse_2d(x, axis):
    x = cupy.moveaxis(x, axis, -1)
    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    if not x.flags.c_contiguous:
        x = x.copy()
    return x, x_shape


def compute_correction_factors(a, block_sz, dtype):
    k = a.size
    correction = cupy.eye(k, dtype=dtype)
    correction = cupy.c_[
        correction[::-1], cupy.empty((k, block_sz), dtype=dtype)]
    corr_kernel = _get_module_func(
        IIR_MODULE, 'compute_correction_factors', correction, a)
    corr_kernel((k,), (1,), (block_sz, k, a, correction))
    return correction


def apply_iir(x, a, axis=-1, zi=None, dtype=None, block_sz=1024):
    # GPU throughput is faster when using single precision floating point
    # numbers
    # x = x.astype(cupy.float32)
    if dtype is None:
        dtype = cupy.result_type(x.dtype, a.dtype)

    a = a.astype(dtype)

    if zi is not None:
        zi = zi.astype(dtype)

    x_shape = x.shape
    x_ndim = x.ndim
    axis = _normalize_axis_index(axis, x_ndim)
    k = a.size
    n = x_shape[axis]

    if x_ndim > 1:
        x, x_shape = collapse_2d(x, axis)
        if zi is not None:
            zi, _ = collapse_2d(zi, axis)

    out = cupy.array(x, dtype=dtype, copy=True)

    num_rows = 1 if x.ndim == 1 else x.shape[0]
    n_blocks = (n + block_sz - 1) // block_sz
    total_blocks = num_rows * n_blocks

    correction = cupy.eye(k, dtype=dtype)
    correction = cupy.c_[
        correction[::-1], cupy.empty((k, block_sz), dtype=dtype)]
    carries = cupy.empty(
        (num_rows, n_blocks, k), dtype=dtype)

    corr_kernel = _get_module_func(
        IIR_MODULE, 'compute_correction_factors', correction, a)
    first_pass_kernel = _get_module_func(IIR_MODULE, 'first_pass_iir', out)
    second_pass_kernel = _get_module_func(IIR_MODULE, 'second_pass_iir', out)
    carry_correction_kernel = _get_module_func(
        IIR_MODULE, 'correct_carries', out)

    corr_kernel((k,), (1,), (block_sz, k, a, correction))
    first_pass_kernel((total_blocks,), (block_sz // 2,),
                      (block_sz, k, n, n_blocks, (n_blocks) * k,
                       correction, out, carries))

    if zi is not None:
        if zi.ndim == 1:
            zi = cupy.broadcast_to(zi, (num_rows, 1, zi.size))
        elif zi.ndim == 2:
            zi = zi.reshape(num_rows, 1, zi.shape[-1])

        if carries.size == 0:
            carries = zi
        else:
            carries = cupy.concatenate((zi, carries), axis=1)

        if not carries.flags.c_contiguous:
            carries = carries.copy()

    if n_blocks > 1 or zi is not None:
        starting_group = int(zi is None)
        blocks_to_merge = n_blocks - starting_group
        carries_stride = (n_blocks + (1 - starting_group)) * k
        carry_correction_kernel(
            (num_rows,), (k,),
            (block_sz, k, n_blocks, carries_stride, starting_group,
             correction, carries))
        second_pass_kernel(
            (num_rows * blocks_to_merge,), (block_sz,),
            (block_sz, k, n, carries_stride, blocks_to_merge,
             starting_group, correction, carries, out))

    if x_ndim > 1:
        out = out.reshape(x_shape)
        out = cupy.moveaxis(out, -1, axis)
        if not out.flags.c_contiguous:
            out = out.copy()

    return out


def apply_iir_sos(x, a, axis=-1, zi=None, dtype=None, block_sz=1024):
    if dtype is None:
        dtype = cupy.result_type(x.dtype, a.dtype)

    a = a.astype(dtype)

    if zi is not None:
        zi = zi.astype(dtype)

    x_shape = x.shape
    x_ndim = x.ndim
    axis = _normalize_axis_index(axis, x_ndim)
    k = a.size
    n = x_shape[axis]

    if x_ndim > 1:
        x, x_shape = collapse_2d(x, axis)
        if zi is not None:
            zi, _ = collapse_2d(zi, axis)

    out = cupy.array(x, dtype=dtype, copy=True)  # NOQA

    num_rows = 1 if x.ndim == 1 else x.shape[0]
    n_blocks = (n + block_sz - 1) // block_sz
    total_blocks = num_rows * n_blocks  # NOQA

    correction = cupy.eye(k, dtype=dtype)
    correction = cupy.c_[
        correction[::-1], cupy.empty((k, block_sz), dtype=dtype)]
    carries = cupy.empty(  # NOQA
        (num_rows, n_blocks, k), dtype=dtype)  # NOQA
