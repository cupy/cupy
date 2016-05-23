import unittest

from cupy import testing


@testing.gpu
class TestGanerate(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_r_1(self, xp, dtype):
        a = testing.shaped_arange((3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 4), xp, dtype)
        return xp.r_[a, b]

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_r_2(self, xp, dtype):
        a = testing.shaped_arange((1, 3), xp, dtype)
        b = testing.shaped_arange((1, 3), xp, dtype)
        return xp.r_[a, 0, 0, b]

    def test_r_3(self, xp):
        with self.assertRaises(NotImplementedError):
            testing.r_[-1:1:6j, [0] * 3, 5, 6]

    @testing.for_all_dtypes(name='dtype')
    def test_r_4(self, xp, dtype):
        a = testing.shaped_arange((1, 3), xp, dtype)
        with self.assertRaises(NotImplementedError):
            testing.r_['-1', a, a]

    def test_r_5(self, xp):
        with self.assertRaises(NotImplementedError):
            testing.r_['0,2', [1, 2, 3], [4, 5, 6]]

    def test_r_6(self, xp):
        with self.assertRaises(NotImplementedError):
            testing.r_['0,2,0', [1, 2, 3], [4, 5, 6]]

    def test_r_7(self, xp):
        with self.assertRaises(NotImplementedError):
            testing.r_['r', [1, 2, 3], [4, 5, 6]]

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_c_1(self, xp, dtype):
        a = testing.shaped_arange((4, 2), xp, dtype)
        b = testing.shaped_reverse_arange((4, 3), xp, dtype)
        return xp.c_[a, b]
