import unittest

from cupy import testing


@testing.gpu
class TestKind(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(type_check=False)
    def test_asfortranarray1(self, xp, dtype):
        x = xp.zeros((2, 3), dtype)
        ret = xp.asfortranarray(x)
        self.assertTrue(x.flags.c_contiguous)
        self.assertTrue(ret.flags.f_contiguous)
        return ret.strides

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(type_check=False)
    def test_asfortranarray2(self, xp, dtype):
        x = xp.zeros((2, 3, 4), dtype)
        ret = xp.asfortranarray(x)
        self.assertTrue(x.flags.c_contiguous)
        self.assertTrue(ret.flags.f_contiguous)
        return ret.strides

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal(type_check=False)
    def test_asfortranarray3(self, xp, dtype):
        x = xp.zeros((2, 3, 4), dtype)
        ret = xp.asfortranarray(xp.asfortranarray(x))
        self.assertTrue(x.flags.c_contiguous)
        self.assertTrue(ret.flags.f_contiguous)
        return ret.strides
