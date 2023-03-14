
import cupy
from cupy._core.internal import _normalize_axis_index


IIR_KERNEL = r"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cupy/math_constants.h>

__global__ void compute_correction_factors(
        const int m, const int k, const float* b, float* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= k) {
        return;
    }

    float* out_start = out + idx * (k + m);
    float* out_off = out_start + k;

    for(int i = 0; i < m; i++) {
        float acc = 0.0;
        for(int j = 0; j < k; j++) {
            acc += b[j] * out_off[i - j - 1];

        }
        out_off[i] = acc;
    }
}

__global__ void first_pass_iir(
        const int m, const int k, const int n, const int n_blocks,
        const int carries_stride, const float* factors, float* out,
        float* carries) {
    int orig_idx = blockDim.x * (blockIdx.x % n_blocks) + threadIdx.x;

    int num_row = blockIdx.x / n_blocks;
    int idx = 2 * orig_idx + 1;

    if(idx >= n) {
        return;
    }

    int group_num = idx / m;
    int group_pos = idx % m;

    float* out_off = out + num_row * n;
    float* carries_off = carries + num_row * carries_stride;

    float* group_start = out_off + m * group_num;
    float* group_carries = carries_off + k * group_num;

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
        float carry = 0.0;
        for(int i = 1; i <= min(k, level); i++) {
            float k_value = group_start[low_bound - i];
            const float* k_factors = factors + (m + k) * (i - 1) + k;
            float factor = k_factors[rel_pos];
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

__global__ void correct_carries(
    const int m, const int k, const int n_blocks, const int carries_stride,
    const int offset, const float* factors, float* carries) {

    int idx = threadIdx.x;
    int pos = idx + (m - k);
    float* row_carries = carries + carries_stride * blockIdx.x;

    for(int i = offset; i < n_blocks; i++) {
        float* this_carries = row_carries + k * (i + (1 - offset));
        float* prev_carries = row_carries + k * (i - offset);

        float carry = 0.0;
        for(int j = 1; j <= k; j++) {
            const float* k_factors = factors + (m + k) * (j - 1) + k;
            float factor = k_factors[pos];
            float k_value = prev_carries[k - j];
            carry += factor * k_value;
        }

        this_carries[idx] += carry;
        __syncthreads();
    }
}

__global__ void second_pass_iir(
        const int m, const int k, const int n, const int carries_stride,
        const int n_blocks, const int offset, const float* factors,
        float* carries, float* out) {

    int idx = blockDim.x * (blockIdx.x % n_blocks) + threadIdx.x;
    idx += offset * m;

    int row_num = blockIdx.x / n_blocks;
    int n_group = idx / m;
    int pos = idx % m;

    if(idx >= n) {
        return;
    }

    float* out_off = out + row_num * n;
    float* carries_off = carries + row_num * carries_stride;
    const float* prev_carries = carries_off + (n_group - offset) * k;

    float carry = 0.0;
    for(int i = 1; i <= k; i++) {
        const float* k_factors = factors + (m + k) * (i - 1) + k;
        float factor = k_factors[pos];
        float k_value = prev_carries[k - i];
        carry += factor * k_value;
    }

    out_off[idx] += carry;
}

"""

IIR_MODULE = cupy.RawModule(
    code=IIR_KERNEL, options=('-std=c++11',),
    name_expressions=['compute_correction_factors',
                      'correct_carries',
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
    # GPU throughput is faster when using single precision floating point
    # numbers

    x = x.astype(cupy.float32)
    a = a.astype(cupy.float32)

    if zi is not None:
        zi = zi.astype(cupy.float32)

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
    block_sz = 1024
    n_blocks = (n + block_sz - 1) // block_sz
    total_blocks = num_rows * n_blocks

    correction = cupy.eye(k, dtype=cupy.float32)
    correction = cupy.c_[
        correction[::-1], cupy.empty((k, block_sz), dtype=cupy.float32)]
    carries = cupy.empty(
        (num_rows, n_blocks, k), dtype=cupy.float32)

    corr_kernel = IIR_MODULE.get_function('compute_correction_factors')
    first_pass_kernel = IIR_MODULE.get_function('first_pass_iir')
    second_pass_kernel = IIR_MODULE.get_function('second_pass_iir')
    carry_correction_kernel = IIR_MODULE.get_function('correct_carries')

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
