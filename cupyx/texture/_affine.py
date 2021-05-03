import cupy

from cupy.cuda import texture
from cupy.cuda import runtime


def _create_texture_object(data,
                           address_mode: str,
                           filter_mode: str,
                           read_mode: str):

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
    # print(f'{data.shape=}, {data.itemsize=}, {data.dtype=}')
    # CUDAArray: last dimension=fastest changing dimension
    array = texture.CUDAarray(texture_fmt, *data.shape[::-1])
    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypeArray, cuArr=array)
    tex_desc = texture.TextureDescriptor(
        (address_mode, ) * data.ndim, filter_mode, read_mode)
    tex_obj = texture.TextureObject(res_desc, tex_desc)
    array.copy_from(data)

    return tex_obj


@cupy._util.memoize(for_each_device=True)
def _get_affine_transform_kernel(ndim: int):

    if ndim == 2:
        return cupy.ElementwiseKernel(
            'U texObj, raw float32 m, uint64 width',
            'T transformed_image',
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
            'affine_transformation_2d',
            preamble='''
            inline __host__ __device__ float dot(float3 a, float3 b)
            {
                return a.x * b.x + a.y * b.y + a.z * b.z;
            }
            ''')

    elif ndim == 3:
        return cupy.ElementwiseKernel(
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
            'affine_transformation_3d',
            preamble='''
            inline __host__ __device__ float dot(float4 a, float4 b)
            {
                return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            ''')
    else:
        raise ValueError(
            'Texture memory affine transformation is defined '
            'only for single channel 2D and 3D arrays')


def affine_transformation(data,
                          transformation_matrix,
                          interpolation: str = 'linear',
                          reshape: bool = False,
                          output=None):

    if (data.ndim < 2) and (data.ndim > 3):
        raise ValueError(
            'Texture memory affine transformation is defined '
            'only for single channel 2D and 3D arrays')

    if interpolation not in ['linear', 'nearest']:
        raise ValueError(
            'Unsupported interpolation (supported: linear, nearest)')

    texture_object = _create_texture_object(data,
                                            address_mode='border',
                                            filter_mode=interpolation,
                                            read_mode='element_type')

    if reshape:
        # TODO(the-lay): add reshape functionality
        pass

    if output is None:
        output = cupy.empty_like(data, data.dtype)
    else:
        # TODO(the-lay): check whether input data array fits into output
        pass

    kernel = _get_affine_transform_kernel(data.ndim)

    kernel(texture_object, transformation_matrix, *data.shape[1:], output)

    del texture_object
    return output
