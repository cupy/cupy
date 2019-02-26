import unittest

import cupy


_test_source = r'''
extern "C" __global__
void test_sum(const float* x1, const float* x2, float* y) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    y[tid] = x1[tid] + x2[tid];
}
'''


class TestRaw(unittest.TestCase):

    def setUp(self):
        self.kern = cupy.RawKernel(_test_source, 'test_sum')

    def test_basic(self):
        x1 = cupy.arange(100, dtype=cupy.float32).reshape(10, 10)
        x2 = cupy.ones((10, 10), dtype=cupy.float32)
        y = cupy.zeros((10, 10), dtype=cupy.float32)
        self.kern((10,), (10,), (x1, x2, y))
        assert (y == x1 + x2).all()

    def test_kernel_attributes(self):
        attributes = self.kern.attributes

        for key in ['binaryVersion',
                    'cacheModeCA',
                    'constSizeBytes',
                    'localSizeBytes',
                    'maxDynamicSharedSizeBytes',
                    'maxThreadsPerBlock',
                    'numRegs',
                    'preferredShmemCarveout',
                    'ptxVersion',
                    'sharedSizeBytes']:
            assert key in attributes

        assert attributes['numRegs'] > 0
        assert attributes['maxThreadsPerBlock'] > 0
        assert attributes['sharedSizeBytes'] == 0
