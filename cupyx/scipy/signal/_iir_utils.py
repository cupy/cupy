
import cupy
from cupy._core.internal import _normalize_axis_index


IIR_KERNEL = r"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cupy/math_constants.h>

__global__ void compute_correction_factors(
        const int m, const int k, const double* b, double* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= k) {
        return;
    }

    double* out_start = out + idx * (k + m);
    double* out_off = out_start + k;

    for(int i = 0; i < m; i++) {
        double acc = 0.0;
        for(int j = 0; j < k; j++) {
            acc += b[j] * out_off[i - j - 1];

        }
        out_off[i] = acc;
    }
}

__global__ void first_pass_iir(
        const int m, const int k, const int n, const int n_blocks,
        const int carries_stride, const double* factors, double* out,
        double* carries) {
    int orig_idx = blockDim.x * (blockIdx.x % n_blocks) + threadIdx.x;

    int num_row = blockIdx.x / n_blocks;
    int idx = 2 * orig_idx + 1;

    if(idx >= n) {
        return;
    }

    int group_num = idx / m;
    int group_pos = idx % m;

    double* out_off = out + num_row * n;
    double* carries_off = carries + num_row * carries_stride;

    double* group_start = out_off + m * group_num;
    double* group_carries = carries_off + k * group_num;

    int pos = group_pos;
    int up_bound = pos;
    int low_bound = pos;
    int rel_pos;

    for(int level = 1, iter = 1; level < m; level *=2, iter++) {
        int sz = min(pow(2.0, ((double) iter)), ((double) m));

        if(level > 1) {
            int factor = ceil(pos / ((double) sz));
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
        double carry = 0.0;
        for(int i = 1; i <= min(k, level); i++) {
            double k_value = group_start[low_bound - i];
            const double* k_factors = factors + (m + k) * (i - 1) + k;
            double factor = k_factors[rel_pos];
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

__global__ void second_pass_iir(
        const int m, const int k, const int n, const int carries_stride,
        const int block_num, const int n_group, const int offset,
        const double* factors, double* carries, double* out) {

    int idx = threadIdx.x;

    const int row_num = block_num < 0 ? blockIdx.x : block_num;
    int group_start = n_group * m;

    if(group_start + idx >= n) {
        return;
    }

    double* out_off = out + row_num * n;
    double* carries_off = carries + row_num * carries_stride;

    double* this_group = out_off + group_start;
    double* this_carries = carries_off + k * (n_group + offset);
    const double* prev_carries = carries_off + (n_group + offset - 1) * k;

    double carry = 0.0;
    for(int i = 1; i <= k; i++) {
        const double* k_factors = factors + (m + k) * (i - 1) + k;
        double factor = k_factors[idx];
        double k_value = prev_carries[k - i];
        carry += factor * k_value;
    }

    this_group[idx] += carry;

    if(idx >= m - k) {
        int k_idx = idx - (m - k);
        this_carries[k_idx] += carry;
        __syncthreads();

        if(k_idx == 0) {
            second_pass_iir<<<1, m>>>(m, k, n, carries_stride, row_num,
                                      n_group + 1, offset, factors, carries,
                                      out);
        }
    }
}

"""

IIR_MODULE = cupy.RawModule(
    code=IIR_KERNEL, options=('-std=c++11', '-dc'),
    name_expressions=['compute_correction_factors',
                      'first_pass_iir',
                      'second_pass_iir'])


def collapse_2d(x, axis):
    x = cupy.moveaxis(x, axis, -1)
    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    if not x.flags.c_contiguous:
        x = x.copy()
    return x, x_shape


def apply_iir(x, a, axis=-1, zi=None):
    x = x.astype(cupy.float64)
    a = a.astype(cupy.float64)

    if zi is not None:
        zi = zi.astype(cupy.float64)

    x_shape = x.shape
    x_ndim = x.ndim
    axis = _normalize_axis_index(axis, x_ndim)
    k = a.size
    n = x_shape[axis]

    if x_ndim > 1:
        x, x_shape = collapse_2d(x, axis)
        if zi is not None:
            zi, _ = collapse_2d(zi, axis)

    out = x.copy()

    num_rows = 1 if x.ndim == 1 else x.shape[0]
    block_sz = 32
    n_blocks = (n + block_sz - 1) // block_sz
    total_blocks = num_rows * n_blocks

    correction = cupy.eye(k)
    correction = cupy.c_[correction[::-1], cupy.empty((k, block_sz))]
    carries = cupy.empty(
        (num_rows, n_blocks, k), dtype=cupy.float64)

    corr_kernel = IIR_MODULE.get_function('compute_correction_factors')
    first_pass_kernel = IIR_MODULE.get_function('first_pass_iir')
    second_pass_kernel = IIR_MODULE.get_function('second_pass_iir')

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
        starting_group = 1 if zi is None else 0
        second_pass_kernel((num_rows,), (block_sz,),
                           (block_sz, k, n, (n_blocks) * k, -1,
                            starting_group, int(zi is not None), correction,
                            carries, out))
    if x_ndim > 1:
        out = out.reshape(x_shape)
        out = cupy.moveaxis(out, -1, axis)
        if not out.flags.c_contiguous:
            out = out.copy()

    return out
