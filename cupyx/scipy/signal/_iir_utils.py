
from itertools import product

import cupy
from cupy._core.internal import _normalize_axis_index
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx.scipy.signal._arraytools import axis_slice


def _get_typename(dtype):
    typename = get_typename(dtype)
    if typename == 'float16':
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


IIR_KERNEL = r"""
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
"""

IIR_SOS_KERNEL = r"""
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

template<typename T>
__global__ void pick_carries(
        const int m, const int n, const int carries_stride, const int n_blocks,
        const int offset, T* x, T* carries) {

    int idx = m * (blockIdx.x % n_blocks) + threadIdx.x + m - 2;
    int pos = threadIdx.x;
    int row_num = blockIdx.x / n_blocks;
    int n_group = idx / m;

    T* x_off = x + row_num * n;
    T* carries_off = carries + row_num * carries_stride;
    T* group_carries = carries_off + (n_group + (1 - offset)) * 2;

    if(idx >= n) {
        return;
    }

    group_carries[pos] = x_off[idx];
}

template<typename U, typename T>
__global__ void compute_correction_factors_sos(
        const int m, const T* f_const, U* all_out) {

    extern __shared__ __align__(sizeof(T)) thrust::complex<double> bc_d[2];
    T* b_c = reinterpret_cast<T*>(bc_d);

    extern __shared__ __align__(sizeof(T)) thrust::complex<double> off_d[4];
    U* off_cache = reinterpret_cast<U*>(off_d);

    int idx = threadIdx.x;
    int num_section = blockIdx.x;

    const int n_const = 6;
    const int a_off = 3;
    const int k = 2;
    const int off_idx = 1;

    U* out = all_out + num_section * k * m;
    U* out_start = out + idx * m;
    const T* b = f_const + num_section * n_const + a_off + 1;

    b_c[idx] = b[idx];
    __syncthreads();

    U* this_cache = off_cache + k * idx;
    this_cache[off_idx - idx] = 1;
    this_cache[idx] = 0;

    for(int i = 0; i < m; i++) {
        U acc = 0.0;
        for(int j = 0; j < k; j++) {
            acc += -((U) b_c[j]) * this_cache[off_idx - j];

        }
        this_cache[0] = this_cache[1];
        this_cache[1] = acc;
        out_start[i] = acc;
    }
}


template<typename T>
__global__ void first_pass_iir_sos(
        const int m, const int n, const int n_blocks,
        const T* factors, T* out, T* carries) {

    extern __shared__ unsigned int thread_status[2];
    extern __shared__ __align__(sizeof(T)) thrust::complex<double> fc_d[2 * 1024];
    T* factor_cache = reinterpret_cast<T*>(fc_d);

    int orig_idx = blockDim.x * (blockIdx.x % n_blocks) + threadIdx.x;

    int num_row = blockIdx.x / n_blocks;
    int idx = 2 * orig_idx + 1;
    const int k = 2;

    if(idx >= n) {
        return;
    }

    int group_num = idx / m;
    int group_pos = idx % m;
    T* out_off = out + num_row * n;
    T* carries_off = carries + num_row * n_blocks * k;

    T* group_start = out_off + m * group_num;
    T* group_carries = carries_off + group_num * k;

    const T* section_factors = factors;
    T* section_carries = group_carries;

    factor_cache[group_pos] = section_factors[group_pos];
    factor_cache[group_pos - 1] = section_factors[group_pos - 1];
    factor_cache[m + group_pos] = section_factors[m + group_pos];
    factor_cache[m + group_pos - 1] = section_factors[m + group_pos - 1];
    __syncthreads();

    int pos = group_pos;
    int up_bound = pos;
    int low_bound = pos;
    int rel_pos;

    for(int level = 1, iter = 1; level < m; level *= 2, iter++) {
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
            const T* k_factors = factor_cache + m  * (i - 1);
            T factor = k_factors[rel_pos];
            carry += k_value * factor;
        }

        group_start[pos] += carry;
        __syncthreads();
    }

    if(pos >= m - k) {
        if(carries != NULL) {
            section_carries[pos - (m - k)] = group_start[pos];
        }
    }
}

template<typename T>
__global__ void correct_carries_sos(
    const int m, const int n_blocks, const int carries_stride,
    const int offset, const T* factors, T* carries) {

    extern __shared__ __align__(sizeof(T)) thrust::complex<double> fcd3[4];
    T* factor_cache = reinterpret_cast<T*>(fcd3);

    int idx = threadIdx.x;
    const int k = 2;
    int pos = idx + (m - k);
    T* row_carries = carries + carries_stride * blockIdx.x;

    factor_cache[2 * idx] = factors[pos];
    factor_cache[2 * idx + 1] = factors[m + pos];
    __syncthreads();

    for(int i = offset; i < n_blocks; i++) {
        T* this_carries = row_carries + k * (i + (1 - offset));
        T* prev_carries = row_carries + k * (i - offset);

        T carry = 0.0;
        for(int j = 1; j <= k; j++) {
            // const T* k_factors = factors + m * (j - 1);
            // T factor = k_factors[pos];
            T factor = factor_cache[2 * idx + (j - 1)];
            T k_value = prev_carries[k - j];
            carry += factor * k_value;
        }

        this_carries[idx] += carry;
        __syncthreads();
    }
}

template<typename T>
__global__ void second_pass_iir_sos(
        const int m, const int n, const int carries_stride,
        const int n_blocks, const int offset, const T* factors,
        T* carries, T* out) {

    extern __shared__ __align__(sizeof(T)) thrust::complex<double> fcd2[2 * 1024];
    T* factor_cache = reinterpret_cast<T*>(fcd2);

    extern __shared__ __align__(sizeof(T)) thrust::complex<double> c_d[2];
    T* carries_cache = reinterpret_cast<T*>(c_d);

    int idx = blockDim.x * (blockIdx.x % n_blocks) + threadIdx.x;
    idx += offset * m;

    int row_num = blockIdx.x / n_blocks;
    int n_group = idx / m;
    int pos = idx % m;
    const int k = 2;

    T* out_off = out + row_num * n;
    T* carries_off = carries + row_num * carries_stride;
    const T* prev_carries = carries_off + (n_group - offset) * k;

    if(pos < k) {
        carries_cache[pos] = prev_carries[pos];
    }

    if(idx >= n) {
        return;
    }

    factor_cache[pos] = factors[pos];
    factor_cache[pos + m] = factors[pos + m];
    __syncthreads();

    T carry = 0.0;
    for(int i = 1; i <= k; i++) {
        const T* k_factors = factor_cache + m * (i - 1);
        T factor = k_factors[pos];
        T k_value = carries_cache[k - i];
        carry += factor * k_value;
    }

    out_off[idx] += carry;
}

template<typename T>
__global__ void fir_sos(
        const int m, const int n, const int carries_stride, const int n_blocks,
        const int offset, const T* sos, T* carries, T* out) {

    extern __shared__ __align__(sizeof(T)) thrust::complex<double> fir_cc[1024 + 2];
    T* fir_cache = reinterpret_cast<T*>(fir_cc);

    extern __shared__ __align__(sizeof(T)) thrust::complex<double> fir_b[3];
    T* b = reinterpret_cast<T*>(fir_b);

    int idx = blockDim.x * (blockIdx.x % n_blocks) + threadIdx.x;
    int row_num = blockIdx.x / n_blocks;
    int n_group = idx / m;
    int pos = idx % m;
    const int k = 2;

    T* out_row = out + row_num * n;
    T* out_off = out_row + n_group * m;
    T* carries_off = carries + row_num * carries_stride;
    T* this_carries = carries_off + k * (n_group + (1 - offset));
    T* group_carries = carries_off + (n_group - offset) * k;

    if(pos <= k) {
        b[pos] = sos[pos];
    }

    if(pos < k) {
        if(offset && n_group == 0) {
            fir_cache[pos] = 0;
        } else {
            fir_cache[pos] = group_carries[pos];
        }
    }

    if(idx >= n) {
        return;
    }

    fir_cache[pos + k] = out_off[pos];
    __syncthreads();

    T acc = 0.0;
    for(int i = k; i >= 0; i--) {
        acc += fir_cache[pos + i] * b[k - i];
    }

    out_off[pos] = acc;
}
"""  # NOQA

IIR_MODULE = cupy.RawModule(
    code=IIR_KERNEL, options=('-std=c++11',),
    name_expressions=[f'compute_correction_factors<{x}, {y}>'
                      for x, y in TYPE_PAIR_NAMES] +
                     [f'correct_carries<{x}>' for x in TYPE_NAMES] +
                     [f'first_pass_iir<{x}>' for x in TYPE_NAMES] +
                     [f'second_pass_iir<{x}>' for x in TYPE_NAMES])

IIR_SOS_MODULE = cupy.RawModule(
    code=IIR_SOS_KERNEL, options=('-std=c++11',),
    name_expressions=[f'compute_correction_factors_sos<{x}, {y}>'
                      for x, y in TYPE_PAIR_NAMES] +
    [f'pick_carries<{x}>' for x in TYPE_NAMES] +
    [f'correct_carries_sos<{x}>' for x in TYPE_NAMES] +
    [f'first_pass_iir_sos<{x}>' for x in TYPE_NAMES] +
    [f'second_pass_iir_sos<{x}>' for x in TYPE_NAMES] +
    [f'fir_sos<{x}>' for x in TYPE_NAMES])


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


def collapse_2d_rest(x, axis):
    x = cupy.moveaxis(x, axis + 1, -1)
    x_shape = x.shape
    x = x.reshape(x.shape[0], -1, x.shape[-1])
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


def compute_correction_factors_sos(sos, block_sz, dtype):
    n_sections = sos.shape[0]
    correction = cupy.empty((n_sections, 2, block_sz), dtype=dtype)
    corr_kernel = _get_module_func(
        IIR_SOS_MODULE, 'compute_correction_factors_sos', correction, sos)
    corr_kernel((n_sections,), (2,), (block_sz, sos, correction))
    return correction


def apply_iir_sos(x, sos, axis=-1, zi=None, dtype=None, block_sz=1024,
                  apply_fir=True, out=None):
    if dtype is None:
        dtype = cupy.result_type(x.dtype, sos.dtype)

    sos = sos.astype(dtype)

    if zi is not None:
        zi = zi.astype(dtype)

    x_shape = x.shape
    x_ndim = x.ndim
    n_sections = sos.shape[0]
    axis = _normalize_axis_index(axis, x_ndim)
    k = 2
    n = x_shape[axis]
    zi_shape = None

    if x_ndim > 1:
        x, x_shape = collapse_2d(x, axis)

    if zi is not None:
        zi, zi_shape = collapse_2d_rest(zi, axis)

    if out is None:
        out = cupy.array(x, dtype=dtype, copy=True)

    num_rows = 1 if x.ndim == 1 else x.shape[0]
    n_blocks = (n + block_sz - 1) // block_sz
    total_blocks = num_rows * n_blocks

    correction = compute_correction_factors_sos(sos, block_sz, dtype)
    carries = cupy.empty(
        (num_rows, n_blocks, k), dtype=dtype)
    all_carries = carries
    zi_out = None
    if zi is not None:
        zi_out = cupy.empty_like(zi)
        all_carries = cupy.empty(
            (num_rows, n_blocks + 1, k), dtype=dtype)

    first_pass_kernel = _get_module_func(
        IIR_SOS_MODULE, 'first_pass_iir_sos', out)
    second_pass_kernel = _get_module_func(
        IIR_SOS_MODULE, 'second_pass_iir_sos', out)
    carry_correction_kernel = _get_module_func(
        IIR_SOS_MODULE, 'correct_carries_sos', out)
    fir_kernel = _get_module_func(IIR_SOS_MODULE, 'fir_sos', out)
    carries_kernel = _get_module_func(IIR_SOS_MODULE, 'pick_carries', out)

    starting_group = int(zi is None)
    blocks_to_merge = n_blocks - starting_group
    carries_stride = (n_blocks + (1 - starting_group)) * k

    carries_kernel((num_rows * n_blocks,), (k,),
                   (block_sz, n, carries_stride, n_blocks, starting_group,
                    out, all_carries))

    for s in range(n_sections):
        b = sos[s]
        if zi is not None:
            section_zi = zi[s, :, :2]
            all_carries[:, 0, :] = section_zi
            zi_out[s, :, :2] = axis_slice(out, n - 2, n)

        if apply_fir:
            fir_kernel((num_rows * n_blocks,), (block_sz,),
                       (block_sz, n, carries_stride, n_blocks, starting_group,
                        b, all_carries, out))

        first_pass_kernel(
            (total_blocks,), (block_sz // 2,),
            (block_sz, n, n_blocks, correction[s], out, carries))

        if n_blocks > 1 or zi is not None:
            if zi is not None:
                section_zi = zi[s, :, 2:]
                all_carries[:, 0, :] = section_zi
                all_carries[:, 1:, :] = carries

            carry_correction_kernel(
                (num_rows,), (k,),
                (block_sz, n_blocks, carries_stride, starting_group,
                    correction[s], all_carries))
            second_pass_kernel(
                (num_rows * blocks_to_merge,), (block_sz,),
                (block_sz, n, carries_stride, blocks_to_merge,
                    starting_group, correction[s], all_carries, out))

        if apply_fir:
            carries_kernel(
                (num_rows * n_blocks,), (k,),
                (block_sz, n, carries_stride, n_blocks, starting_group,
                 out, all_carries))

        if zi is not None:
            zi_out[s, :, 2:] = axis_slice(out, n - 2, n)

    if x_ndim > 1:
        out = out.reshape(x_shape)
        out = cupy.moveaxis(out, -1, axis)
        if not out.flags.c_contiguous:
            out = out.copy()

    if zi is not None:
        zi_out = zi_out.reshape(zi_shape)
        if len(zi_shape) > 2:
            zi_out = cupy.moveaxis(zi_out, -1, axis)
        if not zi_out.flags.c_contiguous:
            zi_out = zi_out.copy()

    if zi is not None:
        return out, zi_out
    return out
