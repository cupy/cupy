import pytest
import unittest

import numpy

import cupy
from cupy import testing
from cupy.cuda import runtime
from cupy.cuda.texture import (ChannelFormatDescriptor, CUDAarray,
                               ResourceDescriptor, TextureDescriptor,
                               TextureObject, TextureReference,
                               SurfaceObject)

if cupy.cuda.runtime.is_hip:
    pytest.skip('HIP texture support is not yet ready',
                allow_module_level=True)


@testing.gpu
@testing.parameterize(*testing.product({
    'xp': ('numpy', 'cupy'),
    'stream': (True, False),
    'dimensions': ((67, 0, 0), (67, 19, 0), (67, 19, 31)),
    'n_channels': (1, 2, 4),
    'dtype': (numpy.float16, numpy.float32, numpy.int8, numpy.int16,
              numpy.int32, numpy.uint8, numpy.uint16, numpy.uint32),
}))
class TestCUDAarray(unittest.TestCase):
    def test_array_gen_cpy(self):
        xp = numpy if self.xp == 'numpy' else cupy
        stream = None if not self.stream else cupy.cuda.Stream()
        width, height, depth = self.dimensions
        n_channel = self.n_channels

        dim = 3 if depth != 0 else 2 if height != 0 else 1
        shape = (depth, height, n_channel*width) if dim == 3 else \
                (height, n_channel*width) if dim == 2 else \
                (n_channel*width,)

        # generate input data and allocate output buffer
        if self.dtype in (numpy.float16, numpy.float32):
            arr = xp.random.random(shape).astype(self.dtype)
            kind = runtime.cudaChannelFormatKindFloat
        else:  # int
            arr = xp.random.randint(100, size=shape, dtype=self.dtype)
            if self.dtype in (numpy.int8, numpy.int16, numpy.int32):
                kind = runtime.cudaChannelFormatKindSigned
            else:
                kind = runtime.cudaChannelFormatKindUnsigned
        arr2 = xp.zeros_like(arr)

        assert arr.flags['C_CONTIGUOUS']
        assert arr2.flags['C_CONTIGUOUS']

        # create a CUDA array
        ch_bits = [0, 0, 0, 0]
        for i in range(n_channel):
            ch_bits[i] = arr.dtype.itemsize*8
        ch = ChannelFormatDescriptor(*ch_bits, kind)
        cu_arr = CUDAarray(ch, width, height, depth)

        # need to wait for the current stream to finish initialization
        if stream is not None:
            s = cupy.cuda.get_current_stream()
            e = s.record()
            stream.wait_event(e)

        # copy from input to CUDA array, and back to output
        cu_arr.copy_from(arr, stream)
        cu_arr.copy_to(arr2, stream)

        # check input and output are identical
        if stream is not None:
            stream.synchronize()
        assert (arr == arr2).all()


source_texobj = r'''
extern "C"{
__global__ void copyKernel1Dfetch(float* output,
                                  cudaTextureObject_t texObj,
                                  int width)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Read from texture and write to global memory
    if (x < width)
        output[x] = tex1Dfetch<float>(texObj, x);
}

__global__ void copyKernel1D(float* output,
                             cudaTextureObject_t texObj,
                             int width)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Read from texture and write to global memory
    float u = x;
    if (x < width)
        output[x] = tex1D<float>(texObj, u);
}

__global__ void copyKernel2D(float* output,
                             cudaTextureObject_t texObj,
                             int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Read from texture and write to global memory
    float u = x;
    float v = y;
    if (x < width && y < height)
        output[y * width + x] = tex2D<float>(texObj, u, v);
}

__global__ void copyKernel3D(float* output,
                             cudaTextureObject_t texObj,
                             int width, int height, int depth)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Read from texture and write to global memory
    float u = x;
    float v = y;
    float w = z;
    if (x < width && y < height && z < depth)
        output[z*width*height+y*width+x] = tex3D<float>(texObj, u, v, w);
}

__global__ void copyKernel3D_4ch(float* output_x,
                                 float* output_y,
                                 float* output_z,
                                 float* output_w,
                                 cudaTextureObject_t texObj,
                                 int width, int height, int depth)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    float4 data;

    // Read from texture, separate channels, and write to global memory
    float u = x;
    float v = y;
    float w = z;
    if (x < width && y < height && z < depth) {
        data = tex3D<float4>(texObj, u, v, w);
        output_x[z*width*height+y*width+x] = data.x;
        output_y[z*width*height+y*width+x] = data.y;
        output_z[z*width*height+y*width+x] = data.z;
        output_w[z*width*height+y*width+x] = data.w;
    }
}
}
'''


source_texref = r'''
extern "C"{
texture<float, cudaTextureType1D, cudaReadModeElementType> texref1D;
texture<float, cudaTextureType2D, cudaReadModeElementType> texref2D;
texture<float, cudaTextureType3D, cudaReadModeElementType> texref3D;
texture<float4, cudaTextureType3D, cudaReadModeElementType> texref3Df4;

__global__ void copyKernel1Dfetch(float* output,
                                  int width)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Read from texture and write to global memory
    if (x < width)
        output[x] = tex1Dfetch(texref1D, x);
}

__global__ void copyKernel1D(float* output,
                             int width)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    // Read from texture and write to global memory
    float u = x;
    if (x < width)
        output[x] = tex1D(texref1D, u);
}

__global__ void copyKernel2D(float* output,
                             int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Read from texture and write to global memory
    float u = x;
    float v = y;
    if (x < width && y < height)
        output[y * width + x] = tex2D(texref2D, u, v);
}

__global__ void copyKernel3D(float* output,
                             int width, int height, int depth)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Read from texture and write to global memory
    float u = x;
    float v = y;
    float w = z;
    if (x < width && y < height && z < depth)
        output[z*width*height+y*width+x] = tex3D(texref3D, u, v, w);
}

__global__ void copyKernel3D_4ch(float* output_x,
                                 float* output_y,
                                 float* output_z,
                                 float* output_w,
                                 int width, int height, int depth)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
    float4 data;

    // Read from texture, separate channels, and write to global memory
    float u = x;
    float v = y;
    float w = z;
    if (x < width && y < height && z < depth) {
        data = tex3D(texref3Df4, u, v, w);
        output_x[z*width*height+y*width+x] = data.x;
        output_y[z*width*height+y*width+x] = data.y;
        output_z[z*width*height+y*width+x] = data.z;
        output_w[z*width*height+y*width+x] = data.w;
    }
}
}
'''


@testing.gpu
@testing.parameterize(*testing.product({
    'dimensions': ((64, 0, 0), (64, 32, 0), (64, 32, 19)),
    'mem_type': ('CUDAarray', 'linear', 'pitch2D'),
    'target': ('object', 'reference'),
}))
class TestTexture(unittest.TestCase):
    def test_fetch_float_texture(self):
        width, height, depth = self.dimensions
        dim = 3 if depth != 0 else 2 if height != 0 else 1

        if (self.mem_type == 'linear' and dim != 1) or \
           (self.mem_type == 'pitch2D' and dim != 2):
            pytest.skip('The test case {0} is inapplicable for {1} and thus '
                        'skipped.'.format(self.dimensions, self.mem_type))

        # generate input data and allocate output buffer
        shape = (depth, height, width) if dim == 3 else \
                (height, width) if dim == 2 else \
                (width,)

        # prepare input, output, and texture memory
        tex_data = cupy.random.random(shape, dtype=cupy.float32)
        real_output = cupy.zeros_like(tex_data)
        ch = ChannelFormatDescriptor(32, 0, 0, 0,
                                     runtime.cudaChannelFormatKindFloat)
        assert tex_data.flags['C_CONTIGUOUS']
        assert real_output.flags['C_CONTIGUOUS']
        if self.mem_type == 'CUDAarray':
            arr = CUDAarray(ch, width, height, depth)
            expected_output = cupy.zeros_like(tex_data)
            assert expected_output.flags['C_CONTIGUOUS']
            # test bidirectional copy
            arr.copy_from(tex_data)
            arr.copy_to(expected_output)
        else:  # linear are pitch2D are backed by ndarray
            arr = tex_data
            expected_output = tex_data

        # create resource and texture descriptors
        if self.mem_type == 'CUDAarray':
            res = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
        elif self.mem_type == 'linear':
            res = ResourceDescriptor(runtime.cudaResourceTypeLinear,
                                     arr=arr,
                                     chDesc=ch,
                                     sizeInBytes=arr.size*arr.dtype.itemsize)
        else:  # pitch2D
            # In this case, we rely on the fact that the hand-picked array
            # shape meets the alignment requirement. This is CUDA's limitation,
            # see CUDA Runtime API reference guide. "TexturePitchAlignment" is
            # assumed to be 32, which should be applicable for most devices.
            res = ResourceDescriptor(runtime.cudaResourceTypePitch2D,
                                     arr=arr,
                                     chDesc=ch,
                                     width=width,
                                     height=height,
                                     pitchInBytes=width*arr.dtype.itemsize)
        address_mode = (runtime.cudaAddressModeClamp,
                        runtime.cudaAddressModeClamp)
        tex = TextureDescriptor(address_mode, runtime.cudaFilterModePoint,
                                runtime.cudaReadModeElementType)

        if self.target == 'object':
            # create a texture object
            texobj = TextureObject(res, tex)
            mod = cupy.RawModule(code=source_texobj)
        else:  # self.target == 'reference'
            mod = cupy.RawModule(code=source_texref)
            texref_name = 'texref'
            texref_name += '3D' if dim == 3 else '2D' if dim == 2 else '1D'
            texrefPtr = mod.get_texref(texref_name)
            # bind texture ref to resource
            texref = TextureReference(texrefPtr, res, tex)  # noqa

        # get and launch the kernel
        ker_name = 'copyKernel'
        ker_name += '3D' if dim == 3 else '2D' if dim == 2 else '1D'
        ker_name += 'fetch' if self.mem_type == 'linear' else ''
        ker = mod.get_function(ker_name)
        block = (4, 4, 2) if dim == 3 else (4, 4) if dim == 2 else (4,)
        grid = ()
        args = (real_output,)
        if self.target == 'object':
            args = args + (texobj,)
        if dim >= 1:
            grid_x = (width + block[0] - 1)//block[0]
            grid = grid + (grid_x,)
            args = args + (width,)
        if dim >= 2:
            grid_y = (height + block[1] - 1)//block[1]
            grid = grid + (grid_y,)
            args = args + (height,)
        if dim == 3:
            grid_z = (depth + block[2] - 1)//block[2]
            grid = grid + (grid_z,)
            args = args + (depth,)
        ker(grid, block, args)

        # validate result
        assert (real_output == expected_output).all()


@testing.gpu
@testing.parameterize(*testing.product({
    'target': ('object', 'reference'),
}))
class TestTextureVectorType(unittest.TestCase):
    def test_fetch_float4_texture(self):
        width = 47
        height = 39
        depth = 11
        n_channel = 4

        # generate input data and allocate output buffer
        in_shape = (depth, height, n_channel*width)
        out_shape = (depth, height, width)

        # prepare input, output, and texture memory
        tex_data = cupy.random.random(in_shape, dtype=cupy.float32)
        real_output_x = cupy.zeros(out_shape, dtype=cupy.float32)
        real_output_y = cupy.zeros(out_shape, dtype=cupy.float32)
        real_output_z = cupy.zeros(out_shape, dtype=cupy.float32)
        real_output_w = cupy.zeros(out_shape, dtype=cupy.float32)
        ch = ChannelFormatDescriptor(32, 32, 32, 32,
                                     runtime.cudaChannelFormatKindFloat)
        arr = CUDAarray(ch, width, height, depth)
        arr.copy_from(tex_data)

        # create resource and texture descriptors
        res = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
        address_mode = (runtime.cudaAddressModeClamp,
                        runtime.cudaAddressModeClamp)
        tex = TextureDescriptor(address_mode, runtime.cudaFilterModePoint,
                                runtime.cudaReadModeElementType)

        if self.target == 'object':
            # create a texture object
            texobj = TextureObject(res, tex)
            mod = cupy.RawModule(code=source_texobj)
        else:  # self.target == 'reference'
            mod = cupy.RawModule(code=source_texref)
            texrefPtr = mod.get_texref('texref3Df4')
            # bind texture ref to resource
            texref = TextureReference(texrefPtr, res, tex)  # noqa

        # get and launch the kernel
        ker_name = 'copyKernel3D_4ch'
        ker = mod.get_function(ker_name)
        block = (4, 4, 2)
        grid = ((width + block[0] - 1)//block[0],
                (height + block[1] - 1)//block[1],
                (depth + block[2] - 1)//block[2])
        args = (real_output_x, real_output_y, real_output_z, real_output_w)
        if self.target == 'object':
            args = args + (texobj,)
        args = args + (width, height, depth)
        ker(grid, block, args)

        # validate result
        assert (real_output_x == tex_data[..., 0::4]).all()
        assert (real_output_y == tex_data[..., 1::4]).all()
        assert (real_output_z == tex_data[..., 2::4]).all()
        assert (real_output_w == tex_data[..., 3::4]).all()


source_surfobj = r"""
extern "C" {
__global__ void writeKernel1D(cudaSurfaceObject_t surf,
                              int width)
{
    unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;

    if (w < width)
    {
        float value = w;
        value *= 3.0;
        surf1Dwrite(value, surf, w * 4);
    }
}

__global__ void writeKernel2D(cudaSurfaceObject_t surf,
                              int width, int height)
{
    unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int h = blockIdx.y * blockDim.y + threadIdx.y;

    if (w < width && h < height)
    {
        float value = h * width + w;
        value *= 3.0;
        surf2Dwrite(value, surf, w * 4, h);
    }
}

__global__ void writeKernel3D(cudaSurfaceObject_t surf,
                              int width, int height, int depth)
{
    unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int h = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int d = blockIdx.z * blockDim.z + threadIdx.z;

    if (w < width && h < height && d < depth)
    {
        float value = d * width * height + h * width + w;
        value *= 3.0;
        surf3Dwrite(value, surf, w * 4, h, d);
    }
}
}
"""


@testing.gpu
@testing.parameterize(*testing.product({
    'dimensions': ((64, 0, 0), (64, 32, 0), (64, 32, 32)),
}))
class TestSurface(unittest.TestCase):
    def test_write_float_surface(self):
        width, height, depth = self.dimensions
        dim = 3 if depth != 0 else 2 if height != 0 else 1

        # generate input data and allocate output buffer
        shape = (depth, height, width) if dim == 3 else \
                (height, width) if dim == 2 else \
                (width,)

        # prepare input, output, and surface memory
        real_output = cupy.zeros(shape, dtype=cupy.float32)
        assert real_output.flags['C_CONTIGUOUS']
        ch = ChannelFormatDescriptor(32, 0, 0, 0,
                                     runtime.cudaChannelFormatKindFloat)
        expected_output = cupy.arange(numpy.prod(shape), dtype=cupy.float32)
        expected_output = expected_output.reshape(shape) * 3.0
        assert expected_output.flags['C_CONTIGUOUS']

        # create resource descriptor
        # note that surface memory only support CUDA array
        arr = CUDAarray(ch, width, height, depth,
                        runtime.cudaArraySurfaceLoadStore)
        arr.copy_from(real_output)  # init to zero
        res = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)

        # create a surface object; currently we don't support surface reference
        surfobj = SurfaceObject(res)
        mod = cupy.RawModule(code=source_surfobj)

        # get and launch the kernel
        ker_name = 'writeKernel'
        ker_name += '3D' if dim == 3 else '2D' if dim == 2 else '1D'
        ker = mod.get_function(ker_name)
        block = (4, 4, 2) if dim == 3 else (4, 4) if dim == 2 else (4,)
        grid = ()
        args = (surfobj,)
        if dim >= 1:
            grid_x = (width + block[0] - 1)//block[0]
            grid = grid + (grid_x,)
            args = args + (width,)
        if dim >= 2:
            grid_y = (height + block[1] - 1)//block[1]
            grid = grid + (grid_y,)
            args = args + (height,)
        if dim == 3:
            grid_z = (depth + block[2] - 1)//block[2]
            grid = grid + (grid_z,)
            args = args + (depth,)
        ker(grid, block, args)

        # validate result
        arr.copy_to(real_output)
        assert (real_output == expected_output).all()
