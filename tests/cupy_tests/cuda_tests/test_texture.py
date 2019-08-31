import pytest
import unittest

import numpy

import cupy
cupy.cuda.set_allocator(None)
from cupy import testing
from cupy.cuda import runtime
from cupy.cuda.texture import (ChannelFormatDescriptor, CUDAArray,
                               ResourceDescriptor, TextureDescriptor,
                               TextureObject)


stream_for_async_cpy = cupy.cuda.Stream()
dev = cupy.cuda.Device(runtime.getDevice())


@testing.parameterize(*testing.product({
    'xp': ('numpy', 'cupy'),
    'stream': (None, stream_for_async_cpy),
    #'dimensions': [(67, 19, 0), ],
    'dimensions': [(67, 0, 0), (67, 19, 0), (67, 19, 31)],
    })
)
class TestCUDAArray(unittest.TestCase):
    def test_array_gen_cpy(self):
        xp = numpy if self.xp == 'numpy' else cupy
        stream = self.stream
        width, height, depth = self.dimensions
        dim = 3 if depth != 0 else 2 if height != 0 else 1

        # generate input data and allocate output buffer
        if dim == 3:
            arr = xp.random.random((depth, height, width)).astype(cupy.float32) # 3D random array
        elif dim == 2:
            arr = xp.random.random((height, width)).astype(cupy.float32) # 2D random array
        else:
            arr = xp.random.random((width,)).astype(cupy.float32) # 1D random array
        arr2 = xp.zeros_like(arr)

        assert arr.flags['C_CONTIGUOUS']
        assert arr2.flags['C_CONTIGUOUS']

        # create a CUDA array
        ch = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
        cu_arr = CUDAArray(ch, width, height, depth)
        
        # copy from input to CUDA array, and back to output
        cu_arr.copy_from(arr, stream)
        cu_arr.copy_to(arr2, stream)

        # check input and output are identical
        if stream is not None:
            dev.synchronize()
        assert (arr == arr2).all()


source=r'''
extern "C"{
__global__ void copyKernel(float* output,
                           cudaTextureObject_t texObj,
                           int width, int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Read from texture and write to global memory
    float u = x;
    float v = y;
    output[y * width + x] = tex2D<float>(texObj, u, v);
}
}
'''


# TODO(leofang): why this test fails if it is run AFTER the tests above???
@testing.gpu
class TestTexture(unittest.TestCase):
    def test_fetch_texture_CUDAArray(self):
        width = 8
        height = 16
        #depth = 4

        #arr = cupy.random.random((depth, height, width)).astype(cupy.float32) # 2D random array
        #arr2 = cupy.zeros_like(arr)
        #assert arr.flags['C_CONTIGUOUS']
        #assert arr2.flags['C_CONTIGUOUS']
        ## create a CUDA array
        #ch = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
        #cu_arr = CUDAArray(ch, width, height, depth)
        ## copy from input to CUDA array, and back to output
        #cu_arr.copy_from(arr)
        #cu_arr.copy_to(arr2)
        #assert (arr == arr2).all() 

        # prepare input, output, and texture, and test bidirectional copy
        tex_data = cupy.arange(width*height, dtype=cupy.float32)
        tex_data = tex_data.reshape(height, width)
        real_output = cupy.zeros_like(tex_data)
        expected_output = cupy.zeros_like(tex_data)
        ch = ChannelFormatDescriptor(32, 0, 0, 0,
                                     runtime.cudaChannelFormatKindFloat)
        arr = CUDAArray(ch, width, height)
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
        ker = cupy.RawKernel(source, 'copyKernel')
        block_x = width
        block_y = height
        grid_x = 1
        grid_y = 1
        ker((grid_x, grid_y), (block_x, block_y), (real_output, texobj, width, height))
        
        # validate result
        print(real_output)
        assert (real_output == expected_output).all()
