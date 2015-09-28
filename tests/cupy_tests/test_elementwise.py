import unittest

import cupy
from cupy import cuda
from cupy import testing
from cupy import elementwise


@testing.gpu
class TestElementwise(unittest.TestCase):

    _multiprocess_can_split_ = True

    def check_copy(self, dtype, src_id, dst_id):
        with cuda.Device(src_id):
            src = testing.shaped_arange((2, 3, 4), dtype=dtype)
        with cuda.Device(dst_id):
            dst = cupy.empty((2, 3, 4), dtype=dtype)
        elementwise.copy(src, dst)
        testing.assert_allclose(src, dst)

    @testing.for_all_dtypes()
    def test_copy(self, dtype):
        device_id = cuda.Device().id
        self.check_copy(dtype, device_id, device_id)

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    def test_copy_multigpu(self, dtype):
        with self.assertRaises(ValueError):
            self.check_copy(dtype, 0, 1)

