
import cupy
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


def _get_module_func(module, func_name, *template_args):
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


PREAMBLE_MODULE = r'''

template<typename T>
__global__ void get_morton_number(
        const int n_points, const T* points, const T* min_val, const T* range,
        int* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const T* point = points + 2 * idx;

    const int Gap08 = 0x00FF00FF;   // Creates 16-bit gap between value bits
    const int Gap04 = 0x0F0F0F0F;   // ... and so on ...
    const int Gap02 = 0x33333333;   // ...
    const int Gap01 = 0x55555555;   // ...

    const int minInt = 0x0;
    const int maxInt = 0x7FFF;

    int mortonNum = 0;

    // Iterate coordinates of point
    for ( int vi = 0; vi < 2; ++vi )
    {
        // Read
        int v = int( ( point[ vi ] - min_val[0] ) / range[0] * 32768.0 );

        if ( v < minInt )
            v = minInt;

        if ( v > maxInt )
            v = maxInt;

        // Create 1-bit gaps between the 10 value bits
        // Ex: 1010101010101010101
        v = ( v | ( v <<  8 ) ) & Gap08;
        v = ( v | ( v <<  4 ) ) & Gap04;
        v = ( v | ( v <<  2 ) ) & Gap02;
        v = ( v | ( v <<  1 ) ) & Gap01;

        // Interleave bits of x-y coordinates
        mortonNum |= ( v << vi );
    }

    out[idx] = mortonNum;
}


template<typename T>
__global__ void compute_distance_2d(
        const int n_points, const T* points, const long long* a_idx,
        const long long* b_idx, int* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const T* p_c = points + 2 * idx;
    const T* p_a = points + 2 * a_idx[0];
    const T* p_b = points + 2 * b_idx[0];

    T abx = p_b[0] - p_a[0];
    T aby = p_b[1] - p_a[1];

    T acx = p_c[0] - p_a[0];
    T acy = p_c[1] - p_a[1];

    T dist = abx * acy - aby * acx;
    int int_dist = __float_as_int( fabs((float) dist) );
    out[idx] = int_dist;
}
'''

_preamble_module = cupy.RawModule(
    code=PREAMBLE_MODULE, options=('-std=c++11',),
    name_expressions=[
        f'get_morton_number<{x}>' for x in ['float', 'double']] + [
            f'compute_distance_2d<{x}>' for x in ['float', 'double']
    ])


def _check_if_coplanar_points(points, pa_idx, pb_idx, pc_idx):
    pa = points[pa_idx]
    pb = points[pb_idx]
    pc = points[pc_idx]

    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1])
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0])
    det = detleft - detright

    is_det_left_pos = cupy.where(detleft > 0, True, False)
    is_det_left_neg = cupy.where(detleft < 0, True, False)

    is_det_right_pos = cupy.where(detright >= 0, True, False)
    is_det_right_neg = cupy.where(detright <= 0, True, False)

    detsum = 0
    if is_det_left_pos:
        if is_det_right_neg:
            return det
        else:
            detsum = detleft + detright
    elif is_det_left_neg:
        if is_det_right_pos:
            return det
        else:
            detsum = -detleft - detright
    else:
        return det

    epsilon = cupy.finfo(points.dtype)
    ccwerrboundA = (3.0 + 16.0 * epsilon) * epsilon
    errbound = ccwerrboundA * detsum

    return cupy.where(
        cupy.logical_or(det >= errbound, -det >= errbound), det,
        cupy.asarray(0, dtype=points.dtype))


def _compute_triangle_orientation(det):
    return cupy.where(det > 0, 1, cupy.where(det < 0, -1, 0))


def _delaunay_triangulation_2d(points):
    n_points = points.shape[0] + 1
    max_triangles = 2 * n_points

    point_vec = cupy.empty((n_points, 2), dtype=points.dtype)
    point_vec[:-1] = points

    triangles = cupy.empty((max_triangles, 3), dtype=cupy.int32)
    triangle_opp = cupy.empty_like(triangles)  # NOQA
    triangle_info = cupy.empty(max_triangles, dtype=cupy.int8)  # NOQA
    counters = cupy.empty(2, dtype=cupy.int32)  # NOQA

    flip = cupy.empty((max_triangles, 2, 2), dtype=cupy.int32)  # NOQA
    tri_msg = cupy.empty((max_triangles, 2), dtype=cupy.int32)  # NOQA
    values = cupy.empty(n_points, dtype=cupy.int32)
    act_tri = cupy.empty(max_triangles, dtype=cupy.int32)  # NOQA
    extra1 = cupy.empty(max_triangles, dtype=cupy.int32)  # NOQA
    extra2 = cupy.empty(max_triangles, dtype=cupy.int32)  # NOQA

    min_val = points.min()
    max_val = points.max()
    range_val = max_val - min_val

    _get_morton_number = _get_module_func(
        _preamble_module, 'get_morton_number', points)

    block_sz = 128
    n_blocks = (points.shape[0] + block_sz - 1) // block_sz

    # Sort the points spatially according to their Morton numbers
    _get_morton_number(
        (block_sz,), (n_blocks,),
        (n_points - 1, points, min_val, range_val, values))

    values[-1] = 2 ** 31 - 1
    points_idx = cupy.argsort(values)
    point_vec = point_vec[points_idx]

    # Find extreme points in the x-axis
    v0 = cupy.argmin(point_vec[:-1, 0])
    v1 = cupy.argmax(point_vec[:-1, 0])

    _compute_distance_2d = _get_module_func(
        _preamble_module, 'compute_distance_2d', points)

    # Find furthest point from v0 and v1, a.k.a the biggest triangle available
    _compute_distance_2d(
        (block_sz,), (n_blocks,), (n_points, point_vec, v0, v1, values))

    breakpoint()
    v2 = cupy.argmax(values[:-1])

    # Check if the three points are not coplanar
    ori = _check_if_coplanar_points(point_vec, v0, v1, v2)

    is_coplanar = cupy.where(ori == 0.0, True, False)
    if is_coplanar:
        raise ValueError(
            'The input is degenerate, the extreme points are close to '
            'coplanar')

    tri_ort = _compute_triangle_orientation(ori)
    if tri_ort == -1:
        v1, v2 = v2, v1

    # Compute the centroid of v0 v1 v2, to be used as the kernel point.
    point_vec[-1] = point_vec[[v0, v1, v2]].mean(0)
