import unittest

import numpy

from cupy import testing
from cupy_tests.core_tests.fusion_tests import fusion_utils


class FusionUnaryUfuncTestBase(unittest.TestCase):

    def generate_inputs(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp, dtype, scale=10, seed=0)
        return (x,), {}


class FusionBinaryUfuncTestBase(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, dtype2, scale=10, seed=1)
        return (x, y), {}


@testing.gpu
@testing.parameterize(*testing.product({
    'func': [
        'bitwise_and', 'bitwise_or', 'bitwise_xor', 'left_shift', 'right_shift'
    ]
}))
class TestFusionBitwiseBinary(FusionBinaryUfuncTestBase):

    @testing.for_int_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_bitwise(self, xp, dtype1, dtype2):
        def impl(x, y):
            if ((x.dtype == 'uint64' and y.dtype.kind == 'i')
                    or (y.dtype == 'uint64' and x.dtype.kind == 'i')):
                # Skip TypeError case.
                return
            return getattr(xp, self.func)(x, y)
        return impl


class TestFusionBitwiseUnary(FusionUnaryUfuncTestBase):

    @testing.for_int_dtypes()
    @fusion_utils.check_fusion()
    def test_invert(self, xp, dtype):
        return lambda x: xp.invert(x)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': [
        'greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal',
        'logical_and', 'logical_or', 'logical_xor',
        'maximum', 'minimum', 'fmax', 'fmin',
    ]
}))
class TestFusionComparisonBinary(FusionBinaryUfuncTestBase):

    @testing.for_all_dtypes_combination(
        no_complex=True, names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_comparison(self, xp, dtype1, dtype2):
        return lambda x, y: getattr(xp, self.func)(x, y)


@testing.gpu
class TestFusionComparisonUnary(FusionUnaryUfuncTestBase):

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_comparison(self, xp, dtype):
        return lambda x: xp.logical_not(x)


@testing.gpu
class TestFusionArrayContents(FusionUnaryUfuncTestBase):

    def generate_inputs(self, xp, has_nan, dtype):
        if numpy.dtype(dtype).kind not in ('f', 'c'):
            return super(TestFusionArrayContents, self).generate_inputs(
                xp, dtype)

        nan = numpy.nan
        inf = dtype(float('inf'))

        if has_nan:
            x = xp.array([-3, nan, -1, nan, 0, nan, inf], dtype=dtype)
        else:
            x = xp.array([-3, inf, -1, -inf, 0, 1, 2], dtype=dtype)
        return (x,), {}

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion(generate_inputs_args=(False,))
    def test_isfinite(self, xp, dtype):
        return lambda x: xp.isfinite(x)

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion(generate_inputs_args=(False,))
    def test_isinf(self, xp, dtype):
        return lambda x: xp.isinf(x)

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion(generate_inputs_args=(True,))
    def test_isnan(self, xp, dtype):
        return lambda x: xp.isnan(x)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': [
        'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
        'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
    ],
}))
class TestFusionTrigonometricUnary(unittest.TestCase):

    def generate_inputs(self, xp, dtype):
        if numpy.dtype(dtype).kind not in ('f', 'c'):
            x = xp.array([0, 1])
        else:
            x = testing.shaped_random((3, 4), xp, dtype, scale=1, seed=0)
        return (x,), {}

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_trigonometric(self, xp, dtype):
        def impl(x):
            with numpy.errstate(divide='ignore', invalid='ignore'):
                return getattr(xp, self.func)(x)
        return impl


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['arctan2', 'hypot']
}))
class TestFusionTrigonometricBinary(FusionBinaryUfuncTestBase):

    @testing.for_all_dtypes_combination(
        no_complex=True, names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_trigonometric(self, xp, dtype1, dtype2):
        return lambda x, y: getattr(xp, self.func)(x, y)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['deg2rad', 'rad2deg', 'degrees', 'radians']
}))
class TestFusionDegRad(FusionUnaryUfuncTestBase):

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_trigonometric(self, xp, dtype):
        return lambda x: getattr(xp, self.func)(x)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['around', 'round', 'round_', 'rint', 'floor', 'ceil', 'trunc',
             'fix']
}))
class TestFusionRounding(FusionUnaryUfuncTestBase):

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_rounding(self, xp, dtype):
        return lambda x: getattr(xp, self.func)(x)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['exp', 'expm1', 'exp2', 'log', 'log10', 'log2', 'log1p']
}))
class TestFusionExpLogUnary(unittest.TestCase):

    def generate_inputs(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp, dtype, scale=10, seed=0) + 1
        return (x,), {}

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_explog(self, xp, dtype):
        return lambda x: getattr(xp, self.func)(x)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['logaddexp', 'logaddexp2']
}))
class TestFusionExpLogBinary(FusionBinaryUfuncTestBase):

    @testing.for_all_dtypes_combination(
        no_complex=True, names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_explog(self, xp, dtype1, dtype2):
        return lambda x, y: getattr(xp, self.func)(x, y)


@testing.gpu
class TestFusionLdexp(FusionBinaryUfuncTestBase):

    @testing.for_float_dtypes(name='dtype1')
    @testing.for_dtypes(['i', 'l'], name='dtype2')
    @fusion_utils.check_fusion()
    def test_explog(self, xp, dtype1, dtype2):
        return lambda x, y: xp.ldexp(x, y)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['signbit', 'frexp']
}))
class TestFusionFloatingUnary(FusionUnaryUfuncTestBase):

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_floating_point_routine(self, xp, dtype):
        return lambda x: getattr(xp, self.func)(x)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['copysign', 'nextafter']
}))
class TestFusionFloatingBinary(FusionBinaryUfuncTestBase):

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_complex=True)
    @fusion_utils.check_fusion()
    def test_floating_point_routine(self, xp, dtype1, dtype2):
        return lambda x, y: getattr(xp, self.func)(x, y)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['reciprocal', 'negative', 'angle', 'conj', 'real', 'imag']
}))
class TestArithmeticUnary(FusionUnaryUfuncTestBase):

    def generate_inputs(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp, dtype, scale=10, seed=0)
        x[x == 0] = 1
        return (x,), {}

    @testing.for_all_dtypes(no_bool=True)
    @fusion_utils.check_fusion()
    def test_arithmetic(self, xp, dtype):
        return lambda x: getattr(xp, self.func)(x)


@testing.gpu
class TestModf(FusionUnaryUfuncTestBase):

    def generate_inputs(self, xp, dtype):
        x = testing.shaped_random((3, 4), xp, dtype, scale=10, seed=0)
        return (x,), {}

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_arithmetic(self, xp, dtype):
        return lambda x: xp.modf(x)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['add', 'subtract', 'multiply', 'power']
}))
class TestArithmeticBinary(FusionBinaryUfuncTestBase):

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=5, seed=0)
        y = testing.shaped_random((3, 4), xp, dtype2, scale=5, seed=0)
        return (x, y), {}

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_complex=True, no_bool=True)
    @fusion_utils.check_fusion()
    def test_arithmetic(self, xp, dtype1, dtype2):
        # TODO(unno): boolean subtract causes DeprecationWarning in numpy>=1.13
        return lambda x, y: getattr(xp, self.func)(x, y)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['divide', 'true_divide', 'floor_divide', 'fmod', 'remainder']
}))
class TestDivide(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, dtype2, scale=10, seed=1)
        y[y == 0] = 1
        return (x, y), {}

    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_complex=True)
    @fusion_utils.check_fusion()
    def test_divide(self, xp, dtype1, dtype2):
        return lambda x, y: getattr(xp, self.func)(x, y)


@testing.gpu
class TestDivmod(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=0)
        y = testing.shaped_random((3, 4), xp, dtype2, scale=10, seed=1)
        y[y == 0] = 1
        return (x, y), {}

    @testing.with_requires('numpy>=1.13')
    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_complex=True)
    @fusion_utils.check_fusion()
    def test_divmod(self, xp, dtype1, dtype2):
        return lambda x, y: xp.divmod(x, y)


@testing.gpu
class TestFusionMisc(FusionUnaryUfuncTestBase):

    @testing.with_requires('numpy>=1.11.2')
    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_sqrt(self, xp, dtype):
        return lambda x: xp.sqrt(x)

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_cbrt(self, xp, dtype):
        return lambda x: xp.cbrt(x)

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_square(self, xp, dtype):
        return lambda x: xp.square(x)

    @testing.for_all_dtypes(no_complex=True, no_bool=True)
    @fusion_utils.check_fusion()
    def test_absolute(self, xp, dtype):
        return lambda x: xp.absolute(x)

    @testing.for_all_dtypes(no_complex=True, no_bool=True)
    @fusion_utils.check_fusion()
    def test_abs(self, xp, dtype):
        return lambda x: xp.abs(x)

    @testing.for_all_dtypes(no_complex=True, no_bool=True)
    @fusion_utils.check_fusion()
    def test_sign(self, xp, dtype):
        return lambda x: xp.sign(x)

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_clip(self, xp, dtype):
        return lambda x: xp.clip(x, dtype(2), dtype(4))


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['i0', 'sinc']
}))
class TestFusionSpecialMath(FusionUnaryUfuncTestBase):

    # TODO(imanishi): Fix for integer tests
    @testing.for_float_dtypes()
    @fusion_utils.check_fusion()
    def test_special_math(self, xp, dtype):
        return lambda x: getattr(xp, self.func)(x)


class TestFusionManipulation(unittest.TestCase):

    def generate_inputs(self, xp, dtype1, dtype2):
        cond = testing.shaped_random((3, 4), xp, 'bool_', seed=0)
        x = testing.shaped_random((3, 4), xp, dtype1, scale=10, seed=1)
        y = testing.shaped_random((3, 4), xp, dtype2, scale=10, seed=2)
        return (cond, x, y), {}

    @testing.for_all_dtypes_combination(names=('dtype1', 'dtype2'))
    @fusion_utils.check_fusion()
    def test_where(self, xp, dtype1, dtype2):
        return lambda cond, x, y: xp.where(cond, x, y)

    # TODO(imanishi): Supoort complex dtypes
    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_complex=True)
    @fusion_utils.check_fusion(accept_error=(TypeError,))
    def test_copyto(self, xp, dtype1, dtype2):
        return lambda cond, x, y: xp.copyto(x, y)

    # TODO(imanishi): Supoort complex dtypes
    @testing.for_all_dtypes_combination(
        names=('dtype1', 'dtype2'), no_complex=True)
    @fusion_utils.check_fusion(accept_error=(TypeError,))
    def test_copyto_where(self, xp, dtype1, dtype2):
        return lambda cond, x, y: xp.where(x, y, where=cond)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['sum', 'prod', 'amax', 'amin', 'max', 'min']
}))
class TestFusionNumericalReduction(FusionUnaryUfuncTestBase):

    @testing.for_all_dtypes()
    @fusion_utils.check_fusion()
    def test_reduction(self, xp, dtype):
        return lambda x: getattr(xp, self.func)(x)


@testing.gpu
@testing.parameterize(*testing.product({
    'func': ['all', 'any']
}))
class TestFusionLogicalReduction(FusionUnaryUfuncTestBase):

    @testing.for_all_dtypes(no_complex=True)
    @fusion_utils.check_fusion()
    def test_reduction(self, xp, dtype):
        return lambda x: getattr(xp, self.func)(x)
