import itertools
import numpy
import unittest

import cupy
from cupy import testing


float_types = [numpy.float16, numpy.float32, numpy.float64]
complex_types = [numpy.complex, numpy.complex64, numpy.complex128]
signed_int_types = [numpy.int8, numpy.int16, numpy.int32, numpy.int64]
unsigned_int_types = [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]
int_types = signed_int_types + unsigned_int_types
all_types = [numpy.bool] + float_types + int_types + complex_types
negative_types = ([numpy.bool] + float_types + signed_int_types
                  + complex_types)
negative_no_complex_types = [numpy.bool] + float_types + signed_int_types
no_complex_types = [numpy.bool] + float_types + int_types


@testing.gpu
@testing.parameterize(*(
    testing.product({
        'nargs': [1],
        'name': ['reciprocal', 'conj', 'angle', 'real', 'imag'],
    }) + testing.product({
        'nargs': [2],
        'name': [
            'add', 'multiply', 'divide', 'power', 'subtract', 'true_divide',
            'floor_divide', 'fmod', 'remainder'],
    })
))
class TestArithmeticRaisesWithNumpyInput(unittest.TestCase):

    def test_raises_with_numpy_input(self):
        nargs = self.nargs
        name = self.name

        # Check TypeError is raised if numpy.ndarray is given as input
        func = getattr(cupy, name)
        for input_xp_list in itertools.product(*[[numpy, cupy]] * nargs):
            if all(xp is cupy for xp in input_xp_list):
                # We don't test all-cupy-array inputs here
                continue
            arys = [xp.array([2, -3]) for xp in input_xp_list]
            with self.assertRaises(TypeError):
                func(*arys)


@testing.gpu
@testing.parameterize(*(
    testing.product({
        'arg1': [testing.shaped_arange((2, 3), numpy, dtype=d)
                 for d in all_types],
        'name': ['conj', 'angle', 'real', 'imag'],
    }) + testing.product({
        'arg1': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in negative_types],
        'name': ['angle'],
    }) + testing.product({
        'arg1': [testing.shaped_arange((2, 3), numpy, dtype=d) + 1
                 for d in all_types],
        'name': ['reciprocal'],
    })
))
class TestArithmeticUnary(unittest.TestCase):

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_unary(self, xp):
        arg1 = self.arg1
        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)
        return getattr(xp, self.name)(arg1)


@testing.gpu
@testing.parameterize(*(
    testing.product({
        # TODO(unno): boolean subtract causes DeprecationWarning in numpy>=1.13
        'arg1': [testing.shaped_arange((2, 3), numpy, dtype=d)
                 for d in all_types],
        'arg2': [testing.shaped_reverse_arange((2, 3), numpy, dtype=d)
                 for d in all_types],
        'name': ['add', 'multiply', 'power', 'subtract'],
    }) + testing.product({
        'arg1': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in negative_types],
        'arg2': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in negative_types],
        'name': ['divide', 'true_divide', 'subtract'],
    }) + testing.product({
        'arg1': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in float_types],
        'arg2': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in float_types],
        'name': ['power', 'true_divide', 'subtract'],
    }) + testing.product({
        'arg1': [testing.shaped_arange((2, 3), numpy, dtype=d)
                 for d in no_complex_types],
        'arg2': [testing.shaped_reverse_arange((2, 3), numpy, dtype=d)
                 for d in no_complex_types],
        'name': ['floor_divide', 'fmod', 'remainder'],
    }) + testing.product({
        'arg1': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in negative_no_complex_types],
        'arg2': [numpy.array([-3, -2, -1, 1, 2, 3], dtype=d)
                 for d in negative_no_complex_types],
        'name': ['floor_divide', 'fmod', 'remainder'],
    })
))
class TestArithmeticBinary(unittest.TestCase):

    @testing.numpy_cupy_allclose(atol=1e-4)
    def test_binary(self, xp):
        arg1 = self.arg1
        arg2 = self.arg2

        # TODO(niboshi): Fix this: power(0, 1j)
        #     numpy => 1+0j
        #     cupy => 0j
        if (self.name == 'power'
                and numpy.asarray(arg1 == 0).any()
                and numpy.asarray(numpy.iscomplex(arg2)).any()):
            return xp.array(True)

        # TODO(niboshi): Fix this: xp.power(0j, False)
        #     numpy => 1+0j
        #     cupy => 0j
        if (self.name == 'power'
                and numpy.asarray(arg1).dtype in complex_types
                and numpy.asarray(arg1 == 0j).any()
                and numpy.asarray(arg2).dtype == numpy.bool
                and numpy.asarray(arg2 == False).any()):
            return xp.array(True)

        # TODO(niboshi): Fix this: xp.add(0j, xp.array([2.], 'f')).dtype
        #     numpy => complex64
        #     cupy => complex128
        if (isinstance(arg1, complex)
                and numpy.asarray(arg2).dtype in (numpy.float16, numpy.float32)):
            return xp.array(True)

        # TODO(niboshi): Fix this: xp.add(xp.array(True), True)
        #     numpy => True
        #     cupy => array(2)
        if (numpy.asarray(arg1).dtype == numpy.bool
                and numpy.asarray(arg2).dtype == numpy.bool
                and (isinstance(arg1, bool) or isinstance(arg2, bool))):
            return xp.array(True)

        if isinstance(arg1, numpy.ndarray):
            arg1 = xp.asarray(arg1)
        if isinstance(arg2, numpy.ndarray):
            arg2 = xp.asarray(arg2)

        func = getattr(xp, self.name)
        with testing.NumpyError(divide='ignore'):
            with numpy.warnings.catch_warnings():
                numpy.warnings.filterwarnings('ignore')
                y = func(arg1, arg2)

        # TODO(niboshi): Fix this. Treat NaN and inf interchangeably
        if y.dtype in (float_types + complex_types):
            y[y == numpy.inf] = numpy.nan
            y[y == -numpy.inf] = numpy.nan

        return y


class TestArithmeticModf(unittest.TestCase):

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        b, c = xp.modf(a)
        d = xp.empty((2, 7), dtype=dtype)
        d[0] = b
        d[1] = c
        return d
