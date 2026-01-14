from __future__ import annotations

import numpy
import pytest

from cupy import testing


ml_dtypes = pytest.importorskip('ml_dtypes')
BF16 = numpy.dtype(ml_dtypes.bfloat16)
TOL = float(ml_dtypes.finfo(BF16).eps)

# Interesting test values including special values
TEST_VALUES = numpy.array(
    [0.0, 1, -1, 2, 100,
     0.5, -0.5, 10.0, -10.0,
     numpy.inf, -numpy.inf, numpy.nan],
    dtype=BF16)


@pytest.mark.parametrize('func', [
    'positive', 'negative', 'absolute', 'sqrt', 'conjugate',
    'isnan', 'isinf', 'isfinite', 'signbit',
])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_unary(xp, func):
    a = xp.array(TEST_VALUES, dtype=BF16)
    return getattr(xp, func)(a)


@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_angle(xp):
    # XXX(seberg): CuPy doesn't handle NaN correctly, so don't test.
    a = xp.array(TEST_VALUES[:-1], dtype=BF16)
    return xp.angle(a, deg=False)


@pytest.mark.parametrize('func', [
    'add', 'subtract', 'multiply', 'divide', 'power', 'floor_divide',
    'remainder',
    # Comparisons:
    'greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal'
])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_binary(xp, func):
    x = xp.asarray(TEST_VALUES)
    if func in {'floor_divide', 'remainder'}:
        # XXX(seberg): CUDA handles some non-finite values differently
        x = x[xp.isfinite(x)]

    y = x[:, None]  # Broadcasting to test all combinations
    with numpy.errstate(all='ignore'):
        return getattr(xp, func)(x, y)


@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_clip(xp):
    # XXX(seberg): Unimplemented in ml_dtypes if this fails add loop.
    a = xp.array([0.0, 2.0, 5.0, 10.0, -5.0], dtype=BF16)
    return xp.clip(a, 1, 4)


@pytest.mark.parametrize('shapes', [
    ((2, 3), (3, 2)),  # matrix @ matrix
    ((3,), (3,)),      # vector @ vector
])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_matmul(xp, shapes):
    shape_a, shape_b = shapes
    a = testing.shaped_arange(shape_a, xp, dtype=BF16)
    b = testing.shaped_arange(shape_b, xp, dtype=BF16)
    return xp.matmul(a, b)


@pytest.mark.parametrize('func', ['sum', 'prod', 'min', 'max', 'mean'])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_reductions(xp, func):
    a = xp.asarray(TEST_VALUES.reshape(3, 4))
    with numpy.errstate(all='ignore'):
        return getattr(xp, func)(a, axis=1)


@pytest.mark.parametrize('from_dtype', [
    numpy.float16, numpy.float32, numpy.float64, numpy.int32])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_astype_to_bfloat16(xp, from_dtype):
    a = xp.asarray(TEST_VALUES)
    if from_dtype == numpy.int32:
        a = a[xp.isfinite(a)]
    return a.astype(from_dtype).astype(BF16)


@pytest.mark.parametrize('to_dtype', [
    numpy.float16, numpy.float32, numpy.float64, numpy.int32])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_astype_from_bfloat16(xp, to_dtype):
    a = xp.asarray(TEST_VALUES)
    if to_dtype == numpy.int32:
        a = a[xp.isfinite(a)]
    return a.astype(to_dtype)


