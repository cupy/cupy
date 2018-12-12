import unittest

import cupy


_test_source = r'''
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
    def test_basic(self):
        kern = cupy.RawKernel(_test_source, 'test_sum')
        x1 = cupy.arange(100, dtype=cupy.float32).reshape(10, 10)
        x2 = cupy.ones((10, 10), dtype=cupy.float32)
        y = cupy.zeros((10, 10), dtype=cupy.float32)
        kern((10,), (10,), (x1, x2, y))
        assert (y == x1 + x2).all()

    # test compiling using the compile() method and invoking multiple kernels
    # in one single .cubin
    def test_multiple(self):
        ker = cupy.RawKernel(_test_source2, None)
        # module = ker.compile(_test_source2)
        module = ker.compile()
        ker_sum = module.get_function('test_sum')
        ker_times = module.get_function('test_multiply')

        N = 10
        x1 = cupy.arange(N**2, dtype=cupy.float32).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=cupy.float32)
        y = cupy.zeros((N, N), dtype=cupy.float32)

        ker_sum((N,), (N,), (x1, x2, y, N**2))
        assert (y == x1 + x2).all()

        ker_times((N,), (N,), (x1, x2, y, N**2))
        assert (y == x1 * x2).all()

    # test setting C macros using compiler options
    def test_macro(self):
        ker = cupy.RawKernel(_test_source3, None, ("-DPRECISION=1",))
        # module = ker.compile(_test_source3, ("-DPRECISION=1",))
        module = ker.compile()
        ker_sum = module.get_function('test_sum')
        ker_times = module.get_function('test_multiply')

        N = 10
        x1 = cupy.arange(N**2, dtype=cupy.float32).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=cupy.float32)
        y = cupy.zeros((N, N), dtype=cupy.float32)

        ker_sum((N,), (N,), (x1, x2, y, N**2))
        assert (y == x1 + x2).all()

        ker_times((N,), (N,), (x1, x2, y, N**2))
        assert (y == x1 * x2).all()
