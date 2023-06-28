
import cupy
import numpy as np

from cupy_backends.cuda.api import runtime


if runtime.is_hip:
    KERNEL_BASE = r"""
    #include <hip/hip_runtime.h>
"""
else:
    KERNEL_BASE = r"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
"""

KD_KERNEL = KERNEL_BASE + r'''
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

__device__ long long sb(
        const long long s_level, const int n,
        const int num_levels, const long long s) {
    long long num_settled = (1 << s_level) - 1;
    long long num_remaining = num_levels - s_level;

    long long first_node = num_settled;
    long long nls_s = s - first_node;
    long long num_to_left = nls_s * ((1 << num_remaining) - 1);
    long long num_to_left_last = nls_s * (1 << (num_remaining - 1));

    long long total_last = n - ((1 << (num_levels - 1)) - 1);
    long long num_left = min(total_last, num_to_left_last);
    long long num_missing = num_to_left_last - num_left;

    long long sb_s_l = num_settled + num_to_left - num_missing;
    return sb_s_l;
}

__device__ long long ss(
        const int n, const int num_levels,
        const long long s) {

    if(s >= n) {
        return 0;
    }

    long long level = 63 - __clzll(s + 1);
    long long num_level_subtree = num_levels - level;

    long long first = (s + 1) << (num_level_subtree - 1);
    long long on_last = (1 << (num_level_subtree - 1)) - 1;
    long long fllc_s = first + on_last;

    long long val = fllc_s - n;
    long long hi = 1 << (num_level_subtree - 1);
    long long lowest_level = max(min(val, hi), 0ll);

    long long num_nodes = (1 << num_level_subtree) - 1;
    long long ss_s = num_nodes - lowest_level;
    return ss_s;
}

__global__ void update_tags(
        const int n, const int level, long long* tags) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int level_size = (1 << level) - 1;
    if(idx >= n || idx < level_size) {
        return;
    }

    const int num_levels = 32 - __clz(n);

    long long tag = tags[idx];
    long long left_child = 2 * tag + 1;
    long long right_child = 2 * tag + 2;
    long long subtree_size = ss(n, num_levels, left_child);
    long long segment_begin = sb(level, n, num_levels, tag);
    long long pivot_pos = segment_begin + subtree_size;
    if(idx < pivot_pos) {
        tags[idx] = left_child;
    } else if(idx > pivot_pos) {
        tags[idx] = right_child;
    }

}
'''


KD_MODULE = cupy.RawModule(
    code=KD_KERNEL, options=('-std=c++11',),
    name_expressions=['update_tags'])  # +


def asm_kd_tree(points):
    x = points.copy()
    tags = cupy.zeros(x.shape[0], dtype=cupy.int64)
    length = x.shape[0]
    dims = x.shape[1]
    n_iter = int(np.log2(length))

    block_sz = 128
    n_blocks = (length + block_sz - 1) // block_sz
    update_tags = KD_MODULE.get_function('update_tags')

    for level in range(n_iter):
        dim = level % dims
        x_dim = x[:, dim]
        idx = cupy.lexsort(cupy.c_[x_dim, tags].T)
        x = x[idx]
        tags = tags[idx]
        update_tags((n_blocks,), (block_sz,), (length, level, tags))

    level += 1
    dim = level % dims
    x_dim = x[:, dim]
    idx = cupy.lexsort(cupy.c_[x_dim, tags].T)
    x = x[idx]
    return x
