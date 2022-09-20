import numpy

from cupy import testing


class TestUfuncOuter:

    @testing.numpy_cupy_array_equal()
    def test_add_outer(self, xp):
        x = testing.shaped_random((2, 3), xp=xp, dtype=numpy.int32, seed=0)
        y = testing.shaped_random((4, 1, 5), xp=xp, dtype=numpy.int32, seed=1)
        return xp.add.outer(x, y)

    @testing.numpy_cupy_array_equal()
    def test_add_outer_scalar(self, xp):
        return xp.add.outer(2, 3)
