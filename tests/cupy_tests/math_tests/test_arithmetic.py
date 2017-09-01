import itertools
import numpy
import unittest

import cupy
from cupy import testing


@testing.gpu
class TestArithmetic(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-4)
    def check_binary(self, name, xp, dtype, no_complex=False):
        if no_complex and numpy.dtype(dtype).kind == 'c':
            return dtype(True)
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'e', 'f', 'd', 'F', 'D'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_negative(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'e', 'f', 'd', 'F', 'D'])
    @testing.numpy_cupy_allclose(atol=1e-4)
    def check_binary_negative(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        b = xp.array([4, 3, 2, 1, -1, -2], dtype=dtype)
        return getattr(xp, name)(a, b)

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-4)
    def check_binary_negative_no_complex(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        b = xp.array([4, 3, 2, 1, -1, -2], dtype=dtype)
        return getattr(xp, name)(a, b)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_binary_negative_float(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        b = xp.array([4, 3, 2, 1, -1, -2], dtype=dtype)
        return getattr(xp, name)(a, b)

    def check_raises_with_numpy_input(self, nargs, name):
        # Check TypeError is raised if numpy.ndarray is given as input
        func = getattr(cupy, name)
        for input_xp_list in itertools.product(*[[numpy, cupy]] * nargs):
            if all(xp is cupy for xp in input_xp_list):
                # We don't test all-cupy-array inputs here
                continue
            arys = [xp.array([2, -3]) for xp in input_xp_list]
            with self.assertRaises(TypeError):
                func(*arys)

    def test_add(self):
        self.check_binary('add')
        self.check_raises_with_numpy_input(2, 'add')

    def test_reciprocal(self):
        with testing.NumpyError(divide='ignore', invalid='ignore'):
            self.check_unary('reciprocal')
        self.check_raises_with_numpy_input(1, 'reciprocal')

    def test_multiply(self):
        self.check_binary('multiply')
        self.check_raises_with_numpy_input(2, 'multiply')

    def test_divide(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary('divide')
        self.check_raises_with_numpy_input(2, 'divide')

    def test_divide_negative(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_negative('divide')

    def test_power(self):
        self.check_binary('power')
        self.check_raises_with_numpy_input(2, 'power')

    def test_power_negative(self):
        self.check_binary_negative_float('power')

    def test_subtract(self):
        self.check_binary('subtract')
        self.check_raises_with_numpy_input(2, 'subtract')

    def test_true_divide(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary('true_divide')
        self.check_raises_with_numpy_input(2, 'true_divide')

    def test_true_divide_negative(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_negative('true_divide')

    def test_floor_divide(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary('floor_divide', no_complex=True)
        self.check_raises_with_numpy_input(2, 'floor_divide')

    def test_floor_divide_negative(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_negative_no_complex('floor_divide')

    def test_fmod(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary('fmod', no_complex=True)
        self.check_raises_with_numpy_input(2, 'fmod')

    def test_fmod_negative(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_negative_no_complex('fmod')

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        b, c = xp.modf(a)
        d = xp.empty((2, 7), dtype=dtype)
        d[0] = b
        d[1] = c
        return d

    def test_remainder(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary('remainder', no_complex=True)
        self.check_raises_with_numpy_input(2, 'remainder')

    def test_remainder_negative(self):
        with testing.NumpyError(divide='ignore'):
            self.check_binary_negative_no_complex('remainder')

    def test_conj(self):
        self.check_unary('conj')
        self.check_raises_with_numpy_input(1, 'conj')

    def test_angle(self):
        self.check_unary('angle')
        self.check_unary_negative('angle')
        self.check_raises_with_numpy_input(1, 'angle')

    def test_real(self):
        self.check_unary('real')
        self.check_raises_with_numpy_input(1, 'real')

    def test_imag(self):
        self.check_unary('imag')
        self.check_raises_with_numpy_input(1, 'imag')
