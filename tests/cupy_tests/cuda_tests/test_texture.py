# import pytest
import unittest

import numpy

import cupy
from cupy import testing
from cupy.cuda import runtime
from cupy.cuda.texture import (ChannelFormatDescriptor, CUDAarray,
                               ResourceDescriptor, TextureDescriptor,
                               TextureObject)


stream_for_async_cpy = cupy.cuda.Stream()
dev = cupy.cuda.Device(runtime.getDevice())


@testing.gpu
@testing.parameterize(*testing.product({
    'xp': ('numpy', 'cupy'),
    'stream': (True, False),
    'dimensions': ((67, 0, 0), (67, 19, 0), (67, 19, 31)),
    'dtype': (numpy.float16, numpy.float32, numpy.int8, numpy.int16,
              numpy.int32, numpy.uint8, numpy.uint16, numpy.uint32),
}))
class TestCUDAarray(unittest.TestCase):
    def test_array_gen_cpy(self):
        xp = numpy if self.xp == 'numpy' else cupy
        stream = None if not self.stream else stream_for_async_cpy
        width, height, depth = self.dimensions
        dim = 3 if depth != 0 else 2 if height != 0 else 1
        shape = (depth, height, width) if dim == 3 else \
                (height, width) if dim == 2 else \
                (width,)

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
        ch = ChannelFormatDescriptor(arr.dtype.itemsize*8, 0, 0, 0, kind)
        cu_arr = CUDAarray(ch, width, height, depth)

        # copy from input to CUDA array, and back to output
        cu_arr.copy_from(arr, stream)
        cu_arr.copy_to(arr2, stream)

        # check input and output are identical
        if stream is not None:
            dev.synchronize()
        assert (arr == arr2).all()


source = r'''
extern "C"{
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
}
'''


@testing.gpu
@testing.parameterize(*testing.product({
    'dimensions': ((67, 0, 0), (67, 19, 0), (67, 19, 31)),
}))
class TestTexture(unittest.TestCase):
    def test_fetch_float_texture_CUDAarray(self):
        width, height, depth = self.dimensions
        dim = 3 if depth != 0 else 2 if height != 0 else 1

        # generate input data and allocate output buffer
        shape = (depth, height, width) if dim == 3 else \
                (height, width) if dim == 2 else \
                (width,)

        # prepare input, output, and texture, and test bidirectional copy
        tex_data = cupy.random.random(shape, dtype=cupy.float32)
        real_output = cupy.zeros_like(tex_data)
        expected_output = cupy.zeros_like(tex_data)
        ch = ChannelFormatDescriptor(32, 0, 0, 0,
                                     runtime.cudaChannelFormatKindFloat)
        arr = CUDAarray(ch, width, height, depth)
        assert tex_data.flags['C_CONTIGUOUS']
        assert expected_output.flags['C_CONTIGUOUS']
        arr.copy_from(tex_data)
        arr.copy_to(expected_output)

        # create a texture object
        res = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
        address_mode = (runtime.cudaAddressModeClamp,
                        runtime.cudaAddressModeClamp)
        tex = TextureDescriptor(address_mode, runtime.cudaFilterModePoint,
                                runtime.cudaReadModeElementType)
        texobj = TextureObject(res, tex)

        # get and launch the kernel
        mod = cupy.RawModule(source)
        ker_name = 'copyKernel'
        ker_name += '3D' if dim == 3 else '2D' if dim == 2 else '1D'
        ker = mod.get_function(ker_name)
        block = (4, 4, 2) if dim == 3 else (4, 4) if dim == 2 else (4,)
        grid = ()
        args = (real_output, texobj, width)
        if dim >= 1:
            grid_x = (width + block[0] - 1)//block[0]
            grid = grid + (grid_x,)
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
