
import cupy

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
        const int m, const int k, const int n, const double* factors,
        double* out, double* carries) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = 2 * idx + 1;

    if(idx >= n) {
        return;
    }

    int group_num = idx / m;
    int group_pos = idx % m;

    double* group_start = out + m * group_num;
    double* group_carries = carries + k * group_num;

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
        group_carries[pos - (m - k)] = group_start[pos];
    }

}

__global__ void second_pass_iir(
        const int m, const int k, const int n, const int n_group,
        const int offset, const double* factors, double* carries,
        double* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int group_start = n_group * m;

    if(group_start + idx >= n) {
        return;
    }

    double* this_group = out + group_start;
    double* this_carries = carries + k * (n_group + offset);
    const double* prev_carries = carries + (n_group + offset - 1) * k;

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
            second_pass_iir<<<1, m>>>(m, k, n, n_group + 1, offset, factors,
                                      carries, out);
        }
    }
}

"""

IIR_MODULE = cupy.RawModule(
    code=IIR_KERNEL, options=('-std=c++11', '-dc'),
    name_expressions=['compute_correction_factors',
                      'first_pass_iir',
                      'second_pass_iir'])


def apply_iir(x, a, zi=None):
    x = x.astype(cupy.float64)
    a = a.astype(cupy.float64)
    out = x.copy()

    if zi is not None:
        zi = zi.astype(cupy.float64)

    k = a.size
    n = x.size
    block_sz = 32
    n_blocks = (n + block_sz - 1) // block_sz

    correction = cupy.eye(k)
    correction = cupy.c_[correction[::-1], cupy.empty((k, block_sz))]
    carries = cupy.empty((n_blocks - 1, k), dtype=cupy.float64)

    corr_kernel = IIR_MODULE.get_function('compute_correction_factors')
    first_pass_kernel = IIR_MODULE.get_function('first_pass_iir')
    second_pass_kernel = IIR_MODULE.get_function('second_pass_iir')

    corr_kernel((k,), (1,), (block_sz, k, a, correction))
    first_pass_kernel((n_blocks,), (block_sz // 2,),
                      (block_sz, k, n, correction, out, carries))

    if zi is not None:
        if carries.size == 0:
            carries = zi
        else:
            if zi.ndim == 1:
                zi = cupy.expand_dims(zi, 0)
            carries = cupy.r_[zi[-k:, :], carries]

    print(carries)

    if n_blocks > 1 or zi is not None:
        starting_group = 1 if zi is None else 0
        second_pass_kernel((1,), (block_sz,),
                           (block_sz, k, n, starting_group,
                            int(zi is not None), correction, carries, out))
    return out
