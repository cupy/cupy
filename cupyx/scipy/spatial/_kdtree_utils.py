
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime

import numpy as np


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
UNSIGNED_TYPES = [cupy.uint8, cupy.uint16, cupy.uint32, cupy.uint64]
COMPLEX_TYPES = [cupy.complex64, cupy.complex128]
TYPES = FLOAT_TYPES + INT_TYPES + UNSIGNED_TYPES + COMPLEX_TYPES  # type: ignore  # NOQA
TYPE_NAMES = [_get_typename(t) for t in TYPES]


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

KNN_KERNEL = KERNEL_BASE + r'''
#include <cupy/math_constants.h>
#include <cupy/carray.cuh>
#include <cupy/complex.cuh>

__device__ unsigned long long abs(unsigned long long x) {
    return x;
}

__device__ unsigned int abs(unsigned int x) {
    return x;
}

__device__ half abs(half x) {
    return __habs(x);
}

template<typename T>
__device__ double compute_distance_inf(
        const T* __restrict__ point1, const T* __restrict__ point2,
        const double* __restrict__ box_bounds,
        const int n_dims, const double p) {

    double dist = -CUDART_INF ? p == CUDART_INF : CUDART_INF;
    for(int i = 0; i < n_dims; i++) {
        double diff = abs(point1[i] - point2[i]);
        double dim_bound = box_bounds[i];

        if(diff > dim_bound - diff) {
            diff = dim_bound - diff;
        }

        if(p == CUDART_INF) {
            dist = max(dist, diff);
        } else {
            dist = min(dist, diff);
        }
    }
    return dist;
}

template<typename T>
__device__ double compute_distance(
        const T* __restrict__ point1, const T* __restrict__ point2,
        const double* __restrict__ box_bounds,
        const int n_dims, const double p) {

    if(abs(p) == CUDART_INF) {
        return compute_distance_inf<T>(point1, point2, box_bounds, n_dims, p);
    }

    double dist = 0.0;
    for(int i = 0; i < n_dims; i++) {
        double diff = abs(point1[i] - point2[i]);
        double dim_bound = box_bounds[i];
        if(diff > dim_bound - diff) {
            diff = dim_bound - diff;
        }
        dist += pow(diff, p);
    }

    dist = pow(dist, 1.0 / p);
    return dist;
}

__device__ double insort(
        const long long curr, const double dist, const int k,
        double* distances, long long* nodes) {

    if(dist > distances[k - 1]) {
        return distances[k - 1];
    }

    long long left = 0;
    long long right = k - 1;

    while(left != right) {
        long long pos = (left + right) / 2;
        if(distances[pos] < dist) {
            left = pos + 1;
        } else {
            right = pos;
        }
    }

    long long node_to_insert = curr;
    double dist_to_insert = dist;
    double dist_to_return = dist;

    for(long long i = left; i < k; i++) {
        long long node_tmp = nodes[i];
        double dist_tmp = distances[i];

        nodes[i] = node_to_insert;
        distances[i] = dist_to_insert;

        if(nodes[i] != k) {
            dist_to_return = max(dist_to_return, distances[i]);
        }

        node_to_insert = node_tmp;
        dist_to_insert = dist_tmp;

    }

    return dist_to_return;
}

template<typename T>
__device__ void compute_knn(
        const int k, const int n, const int n_dims, const double eps,
        const double p, const double dist_bound, const T* __restrict__ point,
        const T* __restrict__ tree, const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        double* distances, long long* nodes) {

    volatile long long prev = -1;
    volatile long long curr = 0;
    volatile double radius = dist_bound;

    while(true) {
        const long long parent = (curr + 1) / 2 - 1;
        if(curr >= n) {
            prev = curr;
            curr = parent;
            continue;
        }

        const long long child = 2 * curr + 1;
        const long long r_child = 2 * curr + 2;

        const bool from_child = prev >= child;
        const T* cur_point = tree + n_dims * curr;

        const double dist = compute_distance(
            point, cur_point, box_bounds, n_dims, p);

        if(!from_child) {
            if(dist <= radius + eps) {
                radius = insort(index[curr], dist, k, distances, nodes);
            }
        }

        const long long cur_level = 63 - __clzll(curr + 1);
        const long long cur_dim = cur_level % n_dims;
        const double dim_bound = box_bounds[cur_dim];
        double curr_dim_dist = abs(point[cur_dim] - cur_point[cur_dim]);

        if(curr_dim_dist > dim_bound - curr_dim_dist) {
            curr_dim_dist = dim_bound - curr_dim_dist;
        }

        long long cur_close_child = child;
        long long cur_far_child = r_child;

        if(point[cur_dim] > cur_point[cur_dim]) {
            cur_close_child = r_child;
            cur_far_child = child;
        }

        long long next = -1;
        if(prev == cur_close_child) {
            next
          = ((cur_far_child < n) && (curr_dim_dist <= radius + eps))
          ? cur_far_child
          : parent;
        } else if (prev == cur_far_child) {
            next = parent;
        } else {
            next = (child < n) ? cur_close_child : parent;
        }

        if(next == -1) {
            return;
        }

        prev = curr;
        curr = next;
    }
}

template<typename T>
__global__ void knn(
        const int k, const int n, const int points_size, const int n_dims,
        const double eps, const double p, const double dist_bound,
        const T* __restrict__ points, const T* __restrict__ tree,
        const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        double* all_distances, long long* all_nodes) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= points_size) {
        return;
    }

    const T* point = points + n_dims * idx;
    double* distances = all_distances + k * idx;
    long long* nodes = all_nodes + k * idx;

    compute_knn<T>(k, n, n_dims, eps, p, dist_bound, point,
                   tree, index, box_bounds, distances, nodes);
}

'''


KD_MODULE = cupy.RawModule(
    code=KD_KERNEL, options=('-std=c++11',),
    name_expressions=['update_tags'])

KNN_MODULE = cupy.RawModule(
    code=KNN_KERNEL, options=('-std=c++11',),
    name_expressions=[f'knn<{x}>' for x in TYPE_NAMES])


def _get_module_func(module, func_name, *template_args):
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


def asm_kd_tree(points):
    """
    Build an array-based KD-Tree from a given set of points.

    Parameters
    ----------
    points: ndarray
        Array input of size (m, n) which contains `m` points with dimension
        `n`.

    Returns
    -------
    tree: ndarray
        An array representation of a left balanced, dimension alternating
        KD-Tree of the input points.
    indices: ndarray
        An index array that maps the original input to its corresponding
        KD-Tree representation.

    Notes
    -----
    This algorithm is derived from [1]_.

    References
    ----------
    .. [1] Wald, I., GPU-friendly, Parallel, and (Almost-)In-Place
           Construction of Left-Balanced k-d Trees, 2022.
           doi:10.48550/arXiv.2211.00120.
    """
    x = points.copy()
    track_idx = cupy.arange(x.shape[0])
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
        track_idx = track_idx[idx]
        update_tags((n_blocks,), (block_sz,), (length, level, tags))

    level += 1
    dim = level % dims
    x_dim = x[:, dim]
    idx = cupy.lexsort(cupy.c_[x_dim, tags].T)
    x = x[idx]
    track_idx = track_idx[idx]
    return x, track_idx


def compute_knn(points, tree, index, boxdata, k=1, eps=0.0, p=2.0,
                distance_upper_bound=cupy.inf):
    max_k = int(np.max(k))
    n_points, n_dims = points.shape
    if n_dims != tree.shape[-1]:
        raise ValueError('The number of dimensions of the query points must '
                         'match with the tree ones. '
                         f'Expected {tree.shape[-1]}, got: {n_dims}')

    if cupy.dtype(points.dtype) is not cupy.dtype(tree.dtype):
        raise ValueError('Query points dtype must match the tree one.')

    distances = cupy.full((n_points, max_k), cupy.inf, dtype=cupy.float64)
    nodes = cupy.full((n_points, max_k), tree.shape[0], dtype=cupy.int64)

    block_sz = 128
    n_blocks = (n_points + block_sz - 1) // block_sz
    knn = _get_module_func(KNN_MODULE, 'knn', points)
    knn((n_blocks,), (block_sz,),
        (max_k, tree.shape[0], n_points, n_dims, eps, p, distance_upper_bound,
         points, tree, index, boxdata, distances, nodes))

    if not isinstance(k, int):
        indices = [k_i - 1 for k_i in k]
        distances = distances[:, indices]
        nodes = nodes[:, indices]
    elif k == 1:
        distances = cupy.squeeze(distances, -1)
        nodes = cupy.squeeze(nodes, -1)
    return distances, nodes
