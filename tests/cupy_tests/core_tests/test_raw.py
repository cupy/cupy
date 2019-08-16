import unittest
import pytest

import cupy


_test_source1 = r'''
extern "C" __global__
void test_sum(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
}
'''

# test compiling and invoking multiple kernels in one single .cubin
_test_source2 = r'''
extern "C"{

__global__ void test_sum(const float* x1, const float* x2, float* y, \
                         unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] + x2[tid];
    }
}

__global__ void test_multiply(const float* x1, const float* x2, float* y, \
                              unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] * x2[tid];
    }
}

}
'''

# test C macros
_test_source3 = r'''
#ifndef PRECISION
    #define PRECISION 2
#endif

#if PRECISION == 2
    #define TYPE double
#elif PRECISION == 1
    #define TYPE float
#else
    #error precision not supported
#endif

extern "C"{

__global__ void test_sum(const TYPE* x1, const TYPE* x2, TYPE* y, \
                         unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] + x2[tid];
    }
}

__global__ void test_multiply(const TYPE* x1, const TYPE* x2, TYPE* y, \
                              unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        y[tid] = x1[tid] * x2[tid];
    }
}

}
'''


class TestRaw(unittest.TestCase):

    def setUp(self):
        self.kern = cupy.RawKernel(_test_source1, 'test_sum')
        self.mod2 = cupy.RawModule(_test_source2)
        self.mod3 = cupy.RawModule(_test_source3, ("-DPRECISION=2",))

    def test_basic(self):
        x1 = cupy.arange(100, dtype=cupy.float32).reshape(10, 10)
        x2 = cupy.ones((10, 10), dtype=cupy.float32)
        y = cupy.zeros((10, 10), dtype=cupy.float32)
        self.kern((10,), (10,), (x1, x2, y))
        assert (y == x1 + x2).all()

    def test_kernel_attributes(self):
        attrs = self.kern.attributes
        for attribute in ['binary_version',
                          'cache_mode_ca',
                          'const_size_bytes',
                          'local_size_bytes',
                          'max_dynamic_shared_size_bytes',
                          'max_threads_per_block',
                          'num_regs',
                          'preferred_shared_memory_carveout',
                          'ptx_version',
                          'shared_size_bytes']:
            assert attribute in attrs
        assert self.kern.num_regs > 0
        assert self.kern.max_threads_per_block > 0
        assert self.kern.shared_size_bytes == 0

    def test_module(self):
        module = self.mod2
        ker_sum = module.get_function('test_sum')
        ker_times = module.get_function('test_multiply')

        N = 10
        x1 = cupy.arange(N**2, dtype=cupy.float32).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=cupy.float32)
        y = cupy.zeros((N, N), dtype=cupy.float32)

        ker_sum((N,), (N,), (x1, x2, y, N**2))
        assert cupy.allclose(y, x1 + x2)

        ker_times((N,), (N,), (x1, x2, y, N**2))
        assert cupy.allclose(y, x1 * x2)

    def test_compiler_flag(self):
        module = self.mod3
        ker_sum = module.get_function('test_sum')
        ker_times = module.get_function('test_multiply')

        N = 10
        x1 = cupy.arange(N**2, dtype=cupy.float64).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=cupy.float64)
        y = cupy.zeros((N, N), dtype=cupy.float64)

        ker_sum((N,), (N,), (x1, x2, y, N**2))
        assert cupy.allclose(y, x1 + x2)

        ker_times((N,), (N,), (x1, x2, y, N**2))
        assert cupy.allclose(y, x1 * x2)

    def test_invalid_compiler_flag(self):
        with pytest.raises(cupy.cuda.compiler.CompileException) as ex:
            cupy.RawModule(_test_source3, ("-DPRECISION=3",))
        assert 'precision not supported' in str(ex)
