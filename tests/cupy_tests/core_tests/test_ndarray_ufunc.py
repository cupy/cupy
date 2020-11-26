import unittest

import numpy as np

import cupy
from cupy import testing


@testing.gpu
class TestArrayUfunc(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_unary_op(self, dtype):
        a = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        outa = np.sin(a)
        # numpy operation produced a cupy array
        assert isinstance(outa, cupy.ndarray)
        b = a.get()
        outb = np.sin(b)
        assert np.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_unary_op_out(self, dtype):
        a = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        b = a.get()
        outb = np.sin(b)
        # pre-make output with same type as input
        outa = cupy.array(np.array([0, 1, 2]), dtype=outb.dtype)
        np.sin(a, out=outa)
        assert np.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_op(self, dtype):
        a1 = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        outa = np.add(a1, a2)
        # numpy operation produced a cupy array
        assert isinstance(outa, cupy.ndarray)
        b1 = a1.get()
        b2 = a2.get()
        outb = np.add(b1, b2)
        assert np.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_op_out(self, dtype):
        a1 = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        outa = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        np.add(a1, a2, out=outa)
        b1 = a1.get()
        b2 = a2.get()
        outb = np.add(b1, b2)
        assert np.allclose(outa.get(), outb)

    @testing.for_all_dtypes()
    def test_binary_mixed_op(self, dtype):
        a1 = cupy.array(np.array([0, 1, 2]), dtype=dtype)
        a2 = cupy.array(np.array([0, 1, 2]), dtype=dtype).get()
        with self.assertRaises(TypeError):
            # attempt to add cupy and numpy arrays
            np.add(a1, a2)
        with self.assertRaises(TypeError):
            # check reverse order
            np.add(a2, a1)
        with self.assertRaises(TypeError):
            # reject numpy output from cupy
            np.add(a1, a1, out=a2)
        with self.assertRaises(TypeError):
            # reject cupy output from numpy
            np.add(a2, a2, out=a1)
        with self.assertRaises(ValueError):
            # bad form for out=
            # this is also an error with numpy array
            np.sin(a1, out=())
        with self.assertRaises(ValueError):
            # bad form for out=
            # this is also an error with numpy array
            np.sin(a1, out=(a1, a1))

    @testing.numpy_cupy_array_equal()
    def test_indexing(self, xp):
        a = cupy.testing.shaped_arange((3, 1), xp)[:, :, None]
        b = cupy.testing.shaped_arange((3, 2), xp)[:, None, :]
        return a * b

    @testing.numpy_cupy_array_equal()
    def test_shares_memory(self, xp):
        a = cupy.testing.shaped_arange((1000, 1000), xp, 'int64')
        b = xp.transpose(a)
        a += b
        return a
