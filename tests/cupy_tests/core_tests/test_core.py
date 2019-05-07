import unittest

import cupy
from cupy.core import core
from cupy import testing


class TestSize(unittest.TestCase):

    def tearDown(self):
        # Free huge memory for slow test
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_size(self, xp, dtype):
        a = xp.ndarray((2, 3), dtype=dtype)
        return xp.size(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_size_axis(self, xp, dtype):
        a = xp.ndarray((2, 3), dtype=dtype)
        return xp.size(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_size_axis_error(self, xp, dtype):
        a = xp.ndarray((2, 3), dtype=dtype)
        return xp.size(a, axis=3)

    @testing.numpy_cupy_equal()
    @testing.slow
    def test_size_huge(self, xp):
        a = xp.ndarray(2 ** 32, 'b')  # 4 GiB
        return xp.size(a)


_orders = {
    order_arg: order_expect
    for order_expect, order_args in [
        ('C', ['C', 'c', 'CONTIGUOUS', '', None]),
        ('F', ['F', 'f', 'FORTRAN']),
    ]
    for order_arg in order_args
}


class TestOrder(unittest.TestCase):

    @testing.for_orders(_orders.keys())
    def test_ndarray(self, order):
        order_expect = _orders[order]
        a = core.ndarray((2, 3), order=order)
        expect_c = order_expect == 'C'
        expect_f = order_expect == 'F'
        assert a.flags.c_contiguous == expect_c
        assert a.flags.f_contiguous == expect_f
