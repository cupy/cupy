
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


KD_KERNEL = r'''
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

__device__ half max(half a, half b) {
    return __hmax(a, b);
}

__device__ half min(half a, half b) {
    return __hmin(a, b);
}

template<typename T>
__global__ void compute_bounds(
        const int n, const int n_dims,
        const int level, const int level_sz,
        const T* __restrict__ tree,
        T* bounds) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= level_sz) {
        return;
    }

    int level_start = (1 << level) - 1;
    idx += level_start;

    if(idx >= n) {
        return;
    }

    const int l_child = 2 * idx + 1;
    const int r_child = 2 * idx + 2;

    T* this_bounds = bounds + 2 * n_dims * idx;
    T* left_bounds = bounds + 2 * n_dims * l_child;
    T* right_bounds = bounds + 2 * n_dims * r_child;

    if(l_child >= n && r_child >= n) {
        const T* tree_node = tree + n_dims * idx;
        for(int dim = 0; dim < n_dims; dim++) {
            T* dim_bounds = this_bounds + 2 * dim;
            dim_bounds[0] = tree_node[dim];
            dim_bounds[1] = tree_node[dim];
        }
    } else if (l_child >= n || r_child >= n) {
        T* to_copy = right_bounds;
        if(r_child >= n) {
            to_copy = left_bounds;
        }

        for(int dim = 0; dim < n_dims; dim++) {
            T* dim_bounds = this_bounds + 2 * dim;
            T* to_copy_dim_bounds = to_copy + 2 * dim;
            dim_bounds[0] = to_copy_dim_bounds[0];
            dim_bounds[1] = to_copy_dim_bounds[1];
        }
    } else {
        for(int dim = 0; dim < n_dims; dim++) {
            T* dim_bounds = this_bounds + 2 * dim;
            T* left_dim_bounds = left_bounds + 2 * dim;
            T* right_dim_bounds = right_bounds + 2 * dim;

            dim_bounds[0] = min(left_dim_bounds[0], right_dim_bounds[0]);
            dim_bounds[1] = max(left_dim_bounds[1], right_dim_bounds[1]);
        }
    }
}

__global__ void tag_pairs(
        const int n, const int n_pairs,
        const long long* __restrict__ pair_count,
        const long long* __restrict__ pairs,
        const long long* __restrict__ out_off,
        long long* out) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n_pairs) {
        return;
    }

    const long long cur_count = pair_count[idx];
    const long long prev_off = idx == 0 ? 0 : out_off[idx - 1];
    const long long* cur_pairs = pairs + n * idx;
    long long* cur_out = out + prev_off * 2;

    for(int i = 0; i < cur_count; i++) {
        cur_out[2 * i] = idx;
        cur_out[2 * i + 1] = cur_pairs[i];
    }
}
'''

KNN_KERNEL = r'''
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
        const int n_dims, const double p, const int stride) {

    double dist = p == CUDART_INF ? -CUDART_INF : CUDART_INF;
    for(int i = 0; i < n_dims; i++) {
        double diff = abs(point1[i] - point2[i * stride]);
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
        const int n_dims, const double p, const int stride,
        const bool take_root) {

    if(abs(p) == CUDART_INF) {
        return compute_distance_inf<T>(
            point1, point2, box_bounds, n_dims, p, stride);
    }

    double dist = 0.0;
    for(int i = 0; i < n_dims; i++) {
        double diff = abs(point1[i] - point2[i * stride]);
        double dim_bound = box_bounds[i];
        if(diff > dim_bound - diff) {
            diff = dim_bound - diff;
        }
        dist += pow(diff, p);
    }

    if(take_root) {
        dist = pow(dist, 1.0 / p);
    }
    return dist;
}

template<typename T>
__device__ T insort(
        const long long curr, const T dist, const int k, const int n,
        T* distances, long long* nodes, bool check) {

    if(check && dist > distances[k - 1]) {
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
    T dist_to_insert = dist;
    T dist_to_return = dist;

    for(long long i = left; i < k; i++) {
        long long node_tmp = nodes[i];
        T dist_tmp = distances[i];

        nodes[i] = node_to_insert;
        distances[i] = dist_to_insert;

        dist_to_return = max(dist_to_return, distances[i]);
        node_to_insert = node_tmp;
        dist_to_insert = dist_tmp;

    }

    return dist_to_return;
}

template<typename T>
__device__ double min_bound_dist(
        const T* __restrict__ point_bounds, const T point_dim,
        const double dim_bound, const int dim) {
    const T min_bound = point_bounds[0];
    const T max_bound = point_bounds[1];

    double min_dist = abs(min_bound - point_dim);
    min_dist = min(min_dist, dim_bound - min_dist);

    double max_dist = abs(max_bound - point_dim);
    max_dist = min(max_dist, dim_bound - max_dist);
    return min(min_dist, max_dist);
}

template<typename T>
__device__ void compute_knn(
        const int k, const int n, const int n_dims, const double eps,
        const double p, const double dist_bound, const bool periodic,
        const T* __restrict__ point, const T* __restrict__ tree,
        const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const T* __restrict__ tree_bounds,
        double* distances, long long* nodes) {

    volatile long long prev = -1;
    volatile long long curr = 0;
    volatile double radius = !isinf(p) ? pow(dist_bound, p) : dist_bound;
    int visit_count = 0;

    double epsfac = 1.0;
    if(eps != 0) {
        if(p == 2) {
            epsfac = 1.0 / ((1 + eps) * (1 + eps));
        } else if(isinf(p) || p == 1) {
            epsfac = 1.0 / (1 + eps);
        } else {
            epsfac = 1.0 / pow(1 + eps, p);
        }
    }

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

        if(!from_child) {
            const double dist = compute_distance(
                point, cur_point, box_bounds, n_dims, p, 1, false);

            if(dist <= radius) {
                radius = insort<double>(
                    index[curr], dist, k, n, distances, nodes, true);
            }
        }

        const long long cur_level = 63 - __clzll(curr + 1);
        const long long cur_dim = cur_level % n_dims;
        double curr_dim_dist = abs(point[cur_dim] - cur_point[cur_dim]);
        double overflow_dist = box_bounds[cur_dim] - curr_dim_dist;
        bool overflow = curr_dim_dist > overflow_dist;
        curr_dim_dist = overflow ? overflow_dist : curr_dim_dist;
        curr_dim_dist = !isinf(p) ? pow(curr_dim_dist, p) : curr_dim_dist;

        volatile long long cur_close_child = child;
        volatile long long cur_far_child = r_child;

        if(point[cur_dim] > cur_point[cur_dim]) {
            cur_close_child = r_child;
            cur_far_child = child;
        }

        long long next = -1;
        if(prev == cur_close_child) {
            if(periodic) {
                const T* close_child = tree + n_dims * cur_close_child;
                const T* far_child = tree + n_dims * cur_far_child;
                const T* close_bounds = (
                    tree_bounds + 2 * n_dims * cur_close_child + 2 * cur_dim);
                const T* far_bounds = (
                    tree_bounds + 2 * n_dims * cur_far_child + 2 * cur_dim);

                double far_dist = CUDART_INF;
                double close_dist = CUDART_INF;
                double far_bound_dist = CUDART_INF;
                double close_bound_dist = CUDART_INF;

                double curr_dist = compute_distance(
                    point, cur_point, box_bounds, n_dims, p, 1, false);

                if(cur_far_child < n) {
                    far_dist = compute_distance(
                        point, far_child, box_bounds, n_dims, p, 1, false);

                    far_bound_dist = min_bound_dist(
                        far_bounds, point[cur_dim], box_bounds[cur_dim],
                        cur_dim);
                }

                close_dist = compute_distance(
                    point, close_child, box_bounds, n_dims, p, 1, false);

                close_bound_dist = min_bound_dist(
                    close_bounds, point[cur_dim], box_bounds[cur_dim],
                    cur_dim);

                next
                = ((cur_far_child < n) &&
                   ((curr_dim_dist <= radius * epsfac) ||
                    (far_bound_dist <= curr_dim_dist * epsfac) ||
                    (far_dist <= close_dist * epsfac) ||
                    (far_bound_dist <= close_bound_dist + epsfac) ||
                    (far_bound_dist <= radius * epsfac)))
                ? cur_far_child
                : parent;
            } else {
                next
                = ((cur_far_child < n) &&
                   (curr_dim_dist <= radius * epsfac))
                ? cur_far_child
                : parent;
            }
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
        const T* __restrict__ tree_bounds,
        double* all_distances, long long* all_nodes) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= points_size) {
        return;
    }

    const T* point = points + n_dims * idx;
    double* distances = all_distances + k * idx;
    long long* nodes = all_nodes + k * idx;

    compute_knn<T>(k, n, n_dims, eps, p, dist_bound, false, point,
                   tree, index, box_bounds, tree_bounds,
                   distances, nodes);
}

__device__ void adjust_to_box(
        double* point, const int n_dims,
        const double* __restrict__ box_bounds) {
    for(int i = 0; i < n_dims; i++) {
        double dim_value = point[i];
        const double dim_box_bounds = box_bounds[i];
        if(dim_box_bounds > 0) {
            const double r = floor(dim_value / dim_box_bounds);
            double x1 = dim_value - r * dim_box_bounds;
            while(x1 >= dim_box_bounds) x1 -= dim_box_bounds;
            while(x1 < 0) x1 += dim_box_bounds;
            point[i] = x1;
        }
    }
}

__global__ void knn_periodic(
        const int k, const int n, const int points_size, const int n_dims,
        const double eps, const double p, const double dist_bound,
        double* __restrict__ points, const double* __restrict__ tree,
        const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const double* __restrict__ tree_bounds,
        double* all_distances, long long* all_nodes) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= points_size) {
        return;
    }

    double* point = points + n_dims * idx;
    double* distances = all_distances + k * idx;
    long long* nodes = all_nodes + k * idx;

    adjust_to_box(point, n_dims, box_bounds);
    compute_knn<double>(k, n, n_dims, eps, p, dist_bound, true, point,
                        tree, index, box_bounds, tree_bounds,
                        distances, nodes);
}

template<typename T>
__device__ long long compute_query_ball(
        const int n, const int n_dims, const double radius, const double eps,
        const double p, bool periodic, int sort, const T* __restrict__ point,
        const T* __restrict__ tree, const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const T* __restrict__ tree_bounds,
        long long* nodes) {

    volatile long long prev = -1;
    volatile long long curr = 0;
    long long node_count = 0;
    double radius_p = !isinf(p) ? pow(radius, p) : radius;

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

        if(!from_child) {
            const double dist = compute_distance(
                point, cur_point, box_bounds, n_dims, p, 1, false);

            if(dist <= radius_p) {
                if(sort) {
                    insort<long long>(
                        index[curr], index[curr], n, n, nodes, nodes, false);
                } else {
                    nodes[node_count] = index[curr];
                }

                node_count++;
            }
        }

        const long long cur_level = 63 - __clzll(curr + 1);
        const long long cur_dim = cur_level % n_dims;

        double curr_dim_dist = abs(point[cur_dim] - cur_point[cur_dim]);
        double overflow_dist = box_bounds[cur_dim] - curr_dim_dist;
        bool overflow = curr_dim_dist > overflow_dist;
        curr_dim_dist = overflow ? overflow_dist : curr_dim_dist;
        curr_dim_dist = !isinf(p) ? pow(curr_dim_dist, p) : curr_dim_dist;

        volatile long long cur_close_child = child;
        volatile long long cur_far_child = r_child;

        if(point[cur_dim] > cur_point[cur_dim]) {
            cur_close_child = r_child;
            cur_far_child = child;
        }

        long long next = -1;
        if(prev == cur_close_child) {
            if(periodic) {
                const T* close_child = tree + n_dims * cur_close_child;
                const T* far_child = tree + n_dims * cur_far_child;
                const T* close_bounds = (
                    tree_bounds + 2 * n_dims * cur_close_child + 2 * cur_dim);
                const T* far_bounds = (
                    tree_bounds + 2 * n_dims * cur_far_child + 2 * cur_dim);

                double far_dist = CUDART_INF;
                double close_dist = CUDART_INF;
                double far_bound_dist = CUDART_INF;
                double close_bound_dist = CUDART_INF;

                double curr_dist = compute_distance(
                    point, cur_point, box_bounds, n_dims, p, 1, false);

                if(cur_far_child < n) {
                    far_dist = compute_distance(
                        point, far_child, box_bounds, n_dims, p, 1, false);

                    far_bound_dist = min_bound_dist(
                        far_bounds, point[cur_dim], box_bounds[cur_dim],
                        cur_dim);
                }

                close_dist = compute_distance(
                    point, close_child, box_bounds, n_dims, p, 1, false);

                close_bound_dist = min_bound_dist(
                    close_bounds, point[cur_dim], box_bounds[cur_dim],
                    cur_dim);

                next
                = ((cur_far_child < n) &&
                   ((curr_dim_dist <= radius_p * (1 + eps)) ||
                    (far_bound_dist <= curr_dim_dist * (1 + eps)) ||
                    (far_dist <= close_dist * (1 + eps)) ||
                    (far_bound_dist <= close_bound_dist + (1 + eps)) ||
                    (far_bound_dist <= radius_p * (1 + eps))))
                ? cur_far_child
                : parent;
            } else {
                next
                = ((cur_far_child < n) &&
                   (curr_dim_dist <= radius_p * (1 + eps)))
                ? cur_far_child
                : parent;
            }
        } else if (prev == cur_far_child) {
            next = parent;
        } else {
            next = (child < n) ? cur_close_child : parent;
        }

        prev = curr;
        curr = next;

        if(next == -1) {
            return node_count;
        }
    }
}

template<typename T>
__global__ void query_ball(
        const int n, const int points_size, const int n_dims,
        const double radius, const double eps, const double p, const int sort,
        const T* __restrict__ points, const T* __restrict__ tree,
        const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const T* __restrict__ tree_bounds,
        long long* all_nodes, long long* node_count) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= points_size) {
        return;
    }

    const T* point = points + n_dims * idx;
    long long* nodes = all_nodes + n * idx;

    long long count = compute_query_ball<T>(
        n, n_dims, radius, eps, p, false, sort, point, tree, index, box_bounds,
        tree_bounds, nodes);

    node_count[idx] = count;
}

__global__ void query_ball_periodic(
        const int n, const int points_size, const int n_dims,
        const double radius, const double eps, const double p, const int sort,
        double* __restrict__ points, const double* __restrict__ tree,
        const long long* __restrict__ index,
        const double* __restrict__ box_bounds,
        const double* __restrict__ tree_bounds,
        long long* all_nodes, long long* node_count) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= points_size) {
        return;
    }

    double* point = points + n_dims * idx;
    long long* nodes = all_nodes + n * idx;

    adjust_to_box(point, n_dims, box_bounds);
    long long count = compute_query_ball<double>(
        n, n_dims, radius, eps, p, true, sort, point, tree, index, box_bounds,
        tree_bounds, nodes);

    node_count[idx] = count;
}
'''


KD_MODULE = cupy.RawModule(
    code=KD_KERNEL, options=('-std=c++11',),
    name_expressions=['update_tags', 'tag_pairs'] + [
        f'compute_bounds<{x}>' for x in TYPE_NAMES])

KNN_MODULE = cupy.RawModule(
    code=KNN_KERNEL, options=('-std=c++11',),
    name_expressions=['knn_periodic', 'query_ball_periodic'] +
    [f'knn<{x}>' for x in TYPE_NAMES] +
    [f'query_ball<{x}>' for x in TYPE_NAMES])


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
    track_idx = cupy.arange(x.shape[0], dtype=cupy.int64)
    tags = cupy.zeros(x.shape[0], dtype=cupy.int64)
    length = x.shape[0]
    dims = x.shape[1]
    n_iter = int(np.log2(length))

    block_sz = 128
    n_blocks = (length + block_sz - 1) // block_sz
    update_tags = KD_MODULE.get_function('update_tags')
    x_tags = cupy.empty((2, length), dtype=x.dtype)

    level = 0
    for level in range(n_iter):
        dim = level % dims
        x_tags[0, :] = x[:, dim]
        x_tags[1, :] = tags
        idx = cupy.lexsort(x_tags)
        x = x[idx]
        tags = tags[idx]
        track_idx = track_idx[idx]
        update_tags((n_blocks,), (block_sz,), (length, level, tags))

    if n_iter > 1:
        level += 1

    dim = level % dims
    x_tags[0, :] = x[:, dim]
    x_tags[1, :] = tags
    idx = cupy.lexsort(x_tags)
    x = x[idx]
    track_idx = track_idx[idx]
    return x, track_idx


def compute_tree_bounds(tree):
    n, n_dims = tree.shape
    bounds = cupy.empty((n, n_dims, 2), dtype=tree.dtype)
    n_levels = int(np.log2(n))
    compute_bounds = _get_module_func(KD_MODULE, 'compute_bounds', tree)

    block_sz = 128
    for level in range(n_levels, -1, -1):
        level_sz = 2 ** level
        n_blocks = (level_sz + block_sz - 1) // block_sz
        compute_bounds(
            (n_blocks,), (block_sz,),
            (n, n_dims, level, level_sz, tree, bounds))
    return bounds


def compute_knn(points, tree, index, boxdata, bounds, k=1, eps=0.0, p=2.0,
                distance_upper_bound=cupy.inf, adjust_to_box=False):
    max_k = int(np.max(k))
    points_shape = points.shape

    if points.ndim > 2:
        points = points.reshape(-1, points_shape[-1])
        if not points.flags.c_contiguous:
            points = points.copy()

    if points.ndim == 1:
        n_points = 1
        n_dims = points.shape[0]
    else:
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
    knn_fn, fn_args = (
        ('knn', (points,)) if not adjust_to_box else ('knn_periodic', tuple()))
    knn = _get_module_func(KNN_MODULE, knn_fn, *fn_args)
    knn((n_blocks,), (block_sz,),
        (max_k, tree.shape[0], n_points, n_dims, eps, p, distance_upper_bound,
         points, tree, index, boxdata, bounds, distances, nodes))

    if not isinstance(k, int):
        indices = [k_i - 1 for k_i in k]
        distances = distances[:, indices]
        nodes = nodes[:, indices]

    if len(points_shape) > 2:
        distances = distances.reshape(*points_shape[:-1], -1)
        nodes = nodes.reshape(*points_shape[:-1], -1)

    if len(points_shape) == 1:
        distances = cupy.squeeze(distances, 0)
        nodes = cupy.squeeze(nodes, 0)

    if k == 1 and len(points_shape) > 1:
        distances = cupy.squeeze(distances, -1)
        nodes = cupy.squeeze(nodes, -1)

    if not cupy.isinf(p):
        distances = distances ** (1.0 / p)
    return distances, nodes


def find_nodes_in_radius(points, tree, index, boxdata, bounds, r,
                         p=2.0, eps=0, return_sorted=None, return_length=False,
                         adjust_to_box=False, return_tuples=False):
    points_shape = points.shape
    tree_length = tree.shape[0]

    if points.ndim > 2:
        points = points.reshape(-1, points_shape[-1])
        if not points.flags.c_contiguous:
            points = points.copy()

    if points.ndim == 1:
        n_points = 1
        n_dims = points.shape[0]
    else:
        n_points, n_dims = points.shape

    if n_dims != tree.shape[-1]:
        raise ValueError('The number of dimensions of the query points must '
                         'match with the tree ones. '
                         f'Expected {tree.shape[-1]}, got: {n_dims}')

    if points.dtype != tree.dtype:
        raise ValueError('Query points dtype must match the tree one.')

    nodes = cupy.full((n_points, tree_length), tree.shape[0], dtype=cupy.int64)
    total_nodes = cupy.empty((n_points,), cupy.int64)

    return_sorted = 1 if return_sorted is None else return_sorted

    block_sz = 128
    n_blocks = (n_points + block_sz - 1) // block_sz
    query_ball_fn, fn_args = (
        ('query_ball', (points,)) if not adjust_to_box else
        ('query_ball_periodic', tuple()))
    query_ball = _get_module_func(KNN_MODULE, query_ball_fn, *fn_args)
    query_ball((n_blocks,), (block_sz,),
               (tree_length, n_points, n_dims, float(r), eps, float(p),
                int(return_sorted),
                points, tree, index, boxdata, bounds, nodes,
                total_nodes))

    if return_length:
        return total_nodes
    elif not return_tuples:
        split_nodes = cupy.array_split(
            nodes[nodes != tree_length], total_nodes.cumsum().tolist())
        split_nodes = split_nodes[:n_points]
        return split_nodes
    else:
        cum_total = total_nodes.cumsum()
        n_pairs = int(cum_total[-1])
        result = cupy.empty((n_pairs, 2), dtype=cupy.int64)
        tag_pairs = KD_MODULE.get_function('tag_pairs')
        tag_pairs((n_blocks,), (block_sz,),
                  (tree_length, n_points, total_nodes, nodes,
                   cum_total, result))
        return result[result[:, 0] < result[:, 1]]
