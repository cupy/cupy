import unittest

import cupy
from cupy import cuda
from cupy import testing
import numpy
from numpy import testing as np_testing


@testing.gpu
class TestArrayGet(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.stream = cuda.Stream.null

    def check_get(self, f, stream):
        a_gpu = f(cupy)
        a_cpu = a_gpu.get(stream)
        if stream:
            stream.synchronize()
        b_cpu = f(numpy)
        np_testing.assert_array_equal(a_cpu, b_cpu)

    @testing.for_all_dtypes()
    def test_contiguous_array(self, dtype):
        def contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype)
        self.check_get(contiguous_array, None)

    @testing.for_all_dtypes()
    def test_non_contiguous_array(self, dtype):
        def non_contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype)[0::2]
        self.check_get(non_contiguous_array, None)

    @testing.for_all_dtypes()
    def test_contiguous_array_stream(self, dtype):
        def contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype)
        self.check_get(contiguous_array, self.stream)

    @testing.for_all_dtypes()
    def test_non_contiguous_array_stream(self, dtype):
        def non_contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype)[0::2]
        self.check_get(non_contiguous_array, self.stream)

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    def test_get_multigpu(self, dtype):
        with cuda.Device(1):
            src = testing.shaped_arange((2, 3), xp=cupy, dtype=dtype)
            src = cupy.asfortranarray(src)
        with cuda.Device(0):
            dst = src.get()
        expected = testing.shaped_arange((2, 3), xp=numpy, dtype=dtype)
        np_testing.assert_array_equal(dst, expected)
