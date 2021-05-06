from itertools import product

import cupy
import numpy

from cupy import _core
from cupy.cuda import texture
from cupy.cuda import runtime


_affine_transform_2d_linear_kernel = _core.ElementwiseKernel(
    'U texObj, raw float32 m, uint64 width', 'T transformed_image',
    '''
    float3 pixel = make_float3(
        (float)(i / width) + .5f,
        (float)(i % width) + .5f,
        1.0f
    );
    float x = dot(pixel, make_float3(m[0],  m[1],  m[2]));
    float y = dot(pixel, make_float3(m[3],  m[4],  m[5]));
    transformed_image = tex2Dfetch<T>(texObj, y, x);
    ''',
    'affine_transformation_2d_linear',
    preamble='''
    inline __host__ __device__ float dot(float3 a, float3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    ''')


_affine_transform_2d_array_kernel = _core.ElementwiseKernel(
    'U texObj, raw float32 m, uint64 width', 'T transformed_image',
    '''
    float3 pixel = make_float3(
        (float)(i / width) + .5f,
        (float)(i % width) + .5f,
        1.0f
    );
    float x = dot(pixel, make_float3(m[0],  m[1],  m[2]));
    float y = dot(pixel, make_float3(m[3],  m[4],  m[5]));
    transformed_image = tex2D<T>(texObj, y, x);
    ''',
    'affine_transformation_2d_array',
    preamble='''
    inline __host__ __device__ float dot(float3 a, float3 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    ''')


_affine_transform_3d_linear_kernel = _core.ElementwiseKernel(
    'U texObj, raw float32 m, uint64 height, uint64 width',
    'T transformed_volume',
    '''
    float4 voxel = make_float4(
        (float)(i / (width * height)) + .5f,
        (float)((i % (width * height)) / width) + .5f,
        (float)((i % (width * height)) % width) + .5f,
        1.0f
    );
    float x = dot(voxel, make_float4(m[0],  m[1],  m[2],  m[3]));
    float y = dot(voxel, make_float4(m[4],  m[5],  m[6],  m[7]));
    float z = dot(voxel, make_float4(m[8],  m[9],  m[10], m[11]));
    transformed_volume = tex3Dfetch<T>(texObj, z, y, x);
    ''',
    'affine_transformation_3d_linear',
    preamble='''
    inline __host__ __device__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    ''')


_affine_transform_3d_array_kernel = _core.ElementwiseKernel(
    'U texObj, raw float32 m, uint64 height, uint64 width',
    'T transformed_volume',
    '''
    float4 voxel = make_float4(
        (float)(i / (width * height)) + .5f,
        (float)((i % (width * height)) / width) + .5f,
        (float)((i % (width * height)) % width) + .5f,
        1.0f
    );
    float x = dot(voxel, make_float4(m[0],  m[1],  m[2],  m[3]));
    float y = dot(voxel, make_float4(m[4],  m[5],  m[6],  m[7]));
    float z = dot(voxel, make_float4(m[8],  m[9],  m[10], m[11]));
    transformed_volume = tex3D<T>(texObj, z, y, x);
    ''',
    'affine_transformation_3d_array',
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
        raise ValueError('Unsupported data type')

    if address_mode == 'wrap':
        address_mode = runtime.cudaAddressModeWrap
    elif address_mode == 'clamp':
        address_mode = runtime.cudaAddressModeClamp
    elif address_mode == 'mirror':
        address_mode = runtime.cudaAddressModeMirror
    elif address_mode == 'border':
        address_mode = runtime.cudaAddressModeBorder
    else:
        raise ValueError(
            'Unsupported address mode '
            '(supported: wrap, clamp, mirror, border)')

    if filter_mode == 'nearest':
        filter_mode = runtime.cudaFilterModePoint
    elif filter_mode == 'linear':
        filter_mode = runtime.cudaFilterModeLinear
    else:
        raise ValueError(
            'Unsupported filter mode (supported: nearest, linear)')

    if read_mode == 'element_type':
        read_mode = runtime.cudaReadModeElementType
    elif read_mode == 'normalized_float':
        read_mode = runtime.cudaReadModeNormalizedFloat
    else:
        raise ValueError(
            'Unsupported read mode '
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


def _translation_matrix(translation, dtype=numpy.float32):
    ndim = len(translation)
    m = numpy.identity(ndim + 1, dtype=dtype)
    m[:ndim, ndim] = translation[:ndim]
    return m


def affine_transformation(data,
                          transformation_matrix,
                          interpolation: str = 'linear',
                          mode: str = 'border',
                          border_value=0,
                          reshape: bool = True,
                          output=None):
    """
    Apply an affine transformation.

    The method uses texture memory and supports only 2D and 3D arrays without
    channel dimension. TODO(the-lay): support up to 4 channels

    Args:
        data (cupy.ndarray or cupy.cuda.texture.TextureObject): The input
            array or texture object. If passed data is an array, the texture
            is created for the transformation and deleted after.
        transformation_matrix (cupy.ndarray): Affine transformation matrix.
            Must be a homogeneous and have shape ``(ndim + 1, ndim + 1)``.
        interpolation (str): Specifies interpolation mode: ``'linear'`` or
            ``'nearest'``. Ignored if data is a TextureObject. Default is
            ``'linear'``.
        mode (str): Specifies addressing mode for points outside of the array:
            (``'wrap'``, ``'clamp'``, ``'mirror'``, `'border'``). Ignored if
            data is a TextureObject. Default is ``'border'``.
        border_value (TODO): Specifies color to be used for coordinates outside of
            the array for ``'border'`` mode. Default is 0.
        reshape (bool): If ``reshape`` is True, the output shape is adapted so
            that the input array is contained completely in the output. Default
            is ``True``.
        output (cupy.ndarray or ~cupy.dtype): The array in which to place the
            output, or the dtype of the returned array.

    Returns:
        cupy.ndarray or None:
            The transformed input. Returns ``None`` if specified output is an
            array.

    .. seealso:: :func:`cupyx.scipy.ndimage.affine_transformation`
    """

    if isinstance(data, cupy.ndarray):
        ndim = data.ndim
        shape = data.shape
        dtype = data.dtype
        if (ndim < 2) and (ndim > 3):
            raise ValueError(
                'Texture memory affine transformation is defined only for '
                '2D and 3D arrays without channel dimension.')

        if interpolation not in ['linear', 'nearest']:
            raise ValueError(
                'Unsupported interpolation (supported: linear, nearest)')

        texture_object = _create_texture_object(data,
                                                address_mode=mode,
                                                filter_mode=interpolation,
                                                read_mode='element_type',
                                                border_color=border_value)

        if ndim == 2:
            kernel = _affine_transform_2d_array_kernel
        else:
            kernel = _affine_transform_3d_array_kernel

    elif isinstance(data, texture.TextureObject):
        cuArr = data.ResDesc.cuArr
        arr = data.ResDesc.arr

        # texture is backed by CUDA array
        if cuArr is not None:
            # infer array properties
            ndim = cuArr.ndim
            shape = tuple(filter(lambda x: x != 0,
                                 (cuArr.depth, cuArr.height, cuArr.width)))
            print(shape)

            texture_desc = cuArr.desc.get_channel_format()
            ch_kind = texture_desc['f']
            if ch_kind == runtime.cudaChannelFormatKindSigned:
                dtype_kind = 'i'
            elif ch_kind == runtime.cudaChannelFormatKindUnsigned:
                dtype_kind = 'u'
            elif ch_kind == runtime.cudaChannelFormatKindFloat:
                dtype_kind = 'f'
            else:
                raise ValueError('Texture has unsupported data type')

            dtype = ''.join([f'{dtype_kind}{texture_desc[channel] // 8}'
                            for channel in ['x', 'y', 'z', 'w']
                            if texture_desc[channel] != 0])
            dtype = cupy.dtype(dtype)

            if ndim == 1:
                raise ValueError('Affine transformation is not defined for 1D')
            elif ndim == 2:
                kernel = _affine_transform_2d_array_kernel
            elif ndim == 3:
                kernel = _affine_transform_3d_array_kernel

        # texture is backed by linear memory
        elif arr is not None:
            ndim = arr.ndim
            shape = arr.shape
            dtype = arr.dtype

            if ndim == 1:
                raise ValueError('Affine transformation is not defined for 1D')
            elif ndim == 2:
                kernel = _affine_transform_2d_linear_kernel
            elif ndim == 3:
                kernel = _affine_transform_3d_linear_kernel
        else:
            raise ValueError('Texture is not bound to any memory')
    else:
        raise ValueError('affine_transformation expects cupy.ndarray '
                         'or TextureObject')

    if transformation_matrix.shape != (ndim + 1, ndim + 1):
        raise ValueError('Matrix must be have shape (ndim + 1, ndim + 1)')

    if reshape:
        input_center = numpy.divide(shape, 2)
        if isinstance(transformation_matrix, cupy.ndarray):
            transformation_matrix = transformation_matrix.get()

        # a wacky way to create n-dimensional list of boundary points
        boundaries = numpy.array(list(product(*zip((0,) * len(shape) + (1,),
                                                   shape + (1,)))))

        # determine array shape after the transformation
        new_boundaries = (transformation_matrix @ boundaries.T)[:-1]
        shape = (new_boundaries.ptp(axis=1) + 0.5).astype(int)

        # determine required shift for transformation matrix
        output_center = numpy.append(shape / 2, 1)
        output_center = (transformation_matrix[:-1] @ output_center)
        center_shift = input_center - output_center
        transformation_matrix = _translation_matrix(center_shift) @ transformation_matrix

        if isinstance(transformation_matrix, numpy.ndarray):
            transformation_matrix = cupy.array(transformation_matrix)

    if output is None:
        output = cupy.empty(shape, dtype=dtype)
    elif isinstance(output, cupy.dtype):
        output = cupy.empty(shape, dtype=output)
    else:
        # TODO(the-lay): check whether output fits given output data array
        pass

    if isinstance(data, cupy.ndarray):
        kernel(texture_object, transformation_matrix, *shape[1:], output)
    elif isinstance(data, texture.TextureObject):
        kernel(data, transformation_matrix, *shape[1:], output)

    return output
