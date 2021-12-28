import cupy

from cupy import _core
from cupy.cuda import texture
from cupy.cuda import runtime


_affine_transform_2d_array_kernel = _core.ElementwiseKernel(
    'U texObj, raw float32 m, uint64 width', 'T transformed_image',
    '''
    float3 pixel = make_float3(
        (float)(i / width),
        (float)(i % width),
        1.0f
    );
    float x = dot(pixel, make_float3(m[0],  m[1],  m[2])) + .5f;
    float y = dot(pixel, make_float3(m[3],  m[4],  m[5])) + .5f;
    transformed_image = tex2D<T>(texObj, y, x);
    ''',
    'cupyx_texture_affine_transformation_2d_array',
    preamble='''
    inline __host__ __device__ float dot(float3 a, float3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    ''')


_affine_transform_3d_array_kernel = _core.ElementwiseKernel(
    'U texObj, raw float32 m, uint64 height, uint64 width',
    'T transformed_volume',
    '''
    float4 voxel = make_float4(
        (float)(i / (width * height)),
        (float)((i % (width * height)) / width),
        (float)((i % (width * height)) % width),
        1.0f
    );
    float x = dot(voxel, make_float4(m[0],  m[1],  m[2],  m[3])) + .5f;
    float y = dot(voxel, make_float4(m[4],  m[5],  m[6],  m[7])) + .5f;
    float z = dot(voxel, make_float4(m[8],  m[9],  m[10], m[11])) + .5f;
    transformed_volume = tex3D<T>(texObj, z, y, x);
    ''',
    'cupyx_texture_affine_transformation_3d_array',
    preamble='''
    inline __host__ __device__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    ''')


def _create_texture_object(data,
                           address_mode: str,
                           filter_mode: str,
                           read_mode: str,
                           border_color=0):

    if cupy.issubdtype(data.dtype, cupy.unsignedinteger):
        fmt_kind = runtime.cudaChannelFormatKindUnsigned
    elif cupy.issubdtype(data.dtype, cupy.integer):
        fmt_kind = runtime.cudaChannelFormatKindSigned
    elif cupy.issubdtype(data.dtype, cupy.floating):
        fmt_kind = runtime.cudaChannelFormatKindFloat
    else:
        raise ValueError(f'Unsupported data type {data.dtype}')

    if address_mode == 'nearest':
        address_mode = runtime.cudaAddressModeClamp
    elif address_mode == 'constant':
        address_mode = runtime.cudaAddressModeBorder
    else:
        raise ValueError(
            f'Unsupported address mode {address_mode} '
            '(supported: constant, nearest)')

    if filter_mode == 'nearest':
        filter_mode = runtime.cudaFilterModePoint
    elif filter_mode == 'linear':
        filter_mode = runtime.cudaFilterModeLinear
    else:
        raise ValueError(
            f'Unsupported filter mode {filter_mode} '
            f'(supported: nearest, linear)')

    if read_mode == 'element_type':
        read_mode = runtime.cudaReadModeElementType
    elif read_mode == 'normalized_float':
        read_mode = runtime.cudaReadModeNormalizedFloat
    else:
        raise ValueError(
            f'Unsupported read mode {read_mode} '
            '(supported: element_type, normalized_float)')

    texture_fmt = texture.ChannelFormatDescriptor(
        data.itemsize * 8, 0, 0, 0, fmt_kind)
    # CUDAArray: last dimension is the fastest changing dimension
    array = texture.CUDAarray(texture_fmt, *data.shape[::-1])
    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypeArray, cuArr=array)
    # TODO(the-lay): each dimension can have a different addressing mode
    # TODO(the-lay): border color/value can be defined for up to 4 channels
    tex_desc = texture.TextureDescriptor(
        (address_mode, ) * data.ndim, filter_mode, read_mode,
        borderColors=(border_color, ))
    tex_obj = texture.TextureObject(res_desc, tex_desc)
    array.copy_from(data)

    return tex_obj


def affine_transformation(data,
                          transformation_matrix,
                          output_shape=None,
                          output=None,
                          interpolation: str = 'linear',
                          mode: str = 'constant',
                          border_value=0):
    """
    Apply an affine transformation.

    The method uses texture memory and supports only 2D and 3D float32 arrays
    without channel dimension.

    Args:
        data (cupy.ndarray): The input array or texture object.
        transformation_matrix (cupy.ndarray): Affine transformation matrix.
            Must be a homogeneous and have shape ``(ndim + 1, ndim + 1)``.
        output_shape (tuple of ints): Shape of output. If not specified,
            the input array shape is used. Default is None.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array. If not specified,
            creates the output array with shape of ``output_shape``. Default is
            None.
        interpolation (str): Specifies interpolation mode: ``'linear'`` or
            ``'nearest'``. Default is ``'linear'``.
        mode (str): Specifies addressing mode for points outside of the array:
            (`'constant'``, ``'nearest'``). Default is ``'constant'``.
        border_value: Specifies value to be used for coordinates outside
            of the array for ``'constant'`` mode. Default is 0.

    Returns:
        cupy.ndarray:
            The transformed input.

    .. seealso:: :func:`cupyx.scipy.ndimage.affine_transform`
    """

    ndim = data.ndim
    if (ndim < 2) or (ndim > 3):
        raise ValueError(
            'Texture memory affine transformation is defined only for '
            '2D and 3D arrays without channel dimension.')

    dtype = data.dtype
    if dtype != cupy.float32:
        raise ValueError(f'Texture memory affine transformation is available '
                         f'only for float32 data type (not {dtype})')

    if interpolation not in ['linear', 'nearest']:
        raise ValueError(
            f'Unsupported interpolation {interpolation} '
            f'(supported: linear, nearest)')

    if transformation_matrix.shape != (ndim + 1, ndim + 1):
        raise ValueError('Matrix must be have shape (ndim + 1, ndim + 1)')

    texture_object = _create_texture_object(data,
                                            address_mode=mode,
                                            filter_mode=interpolation,
                                            read_mode='element_type',
                                            border_color=border_value)

    if ndim == 2:
        kernel = _affine_transform_2d_array_kernel
    else:
        kernel = _affine_transform_3d_array_kernel

    if output_shape is None:
        output_shape = data.shape

    if output is None:
        output = cupy.zeros(output_shape, dtype=dtype)
    elif isinstance(output, (type, cupy.dtype)):
        if output != cupy.float32:
            raise ValueError(f'Texture memory affine transformation is '
                             f'available only for float32 data type (not '
                             f'{output})')
        output = cupy.zeros(output_shape, dtype=output)
    elif isinstance(output, cupy.ndarray):
        if output.shape != output_shape:
            raise ValueError('Output shapes do not match')
    else:
        raise ValueError('Output must be None, cupy.ndarray or cupy.dtype')

    kernel(texture_object, transformation_matrix, *output_shape[1:], output)
    return output
