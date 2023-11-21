
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime

# from cupy.cuda import thrust


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

'''

_preamble_module = cupy.RawModule(
    code=PREAMBLE_MODULE, options=('-std=c++11',),
    name_expressions=[f'get_morton_number<{x}>' for x in ['float', 'double']])


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

    # points_idx = cupy.arange(n_points, dtype=cupy.int32)
    _get_morton_number = _get_module_func(
        _preamble_module, 'get_morton_number', points)

    breakpoint()

    block_sz = 128
    n_blocks = (points.shape[0] + block_sz - 1) // block_sz
    _get_morton_number(
        (block_sz,), (n_blocks,),
        (n_points - 1, points, min_val, range_val, values))

    values[-1] = 2 ** 31 - 1
    points_idx = cupy.argsort(values)
    point_vec = point_vec[points_idx]
