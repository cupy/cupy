from __future__ import annotations

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing


if not runtime.is_hip and cupy.cuda.get_local_runtime_version() < 12020:
    # We may be ablel to enable this, but for now skip to pass tests.
    pytest.skip(allow_module_level=True,
                reason="bfloat16 is missing some features")

if numpy.lib.NumpyVersion(numpy.__version__) < "2.1.2":
    pytest.skip(allow_module_level=True,
                reason="bfloat16 not enabled due to NumPy bug.")


ml_dtypes = pytest.importorskip('ml_dtypes')
BF16 = numpy.dtype(ml_dtypes.bfloat16)
TOL = float(ml_dtypes.finfo(BF16).eps)

# Interesting test values including special values
TEST_VALUES = numpy.array(
    [0.0, 1, -1, 2, 100,
     0.5, -0.5, 10.0, -10.0,
     numpy.inf, -numpy.inf, numpy.nan],
    dtype=BF16)


@pytest.mark.parametrize("start,stop,step", [
    (0, 10, 1),
    (0, 10, 2),
    (0, 10, -1),
    (0, 50, 0.5),
])
@testing.numpy_cupy_allclose()
def test_arange(xp, start, stop, step):
    return xp.arange(start, stop, step, dtype=BF16)


@pytest.mark.parametrize('func', [
    'positive', 'negative', 'absolute', 'sqrt', 'conjugate',
    'isnan', 'isinf', 'isfinite', 'signbit',
    # Exponential and logarithmic functions:
    'exp', 'expm1', 'exp2', 'log', 'log10', 'log2', 'log1p',
    # Trigonometric functions:
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
    'deg2rad', 'rad2deg',
    # Hyperbolic functions:
    'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
    # Rounding:
    'rint', 'floor', 'ceil', 'trunc', 'fix',
    # Misc:
    'cbrt', 'square', 'fabs', 'sign', 'reciprocal',
])
@numpy.errstate(all='ignore')
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_unary(xp, func):
    a = xp.array(TEST_VALUES, dtype=BF16)
    if func in {'sign'}:
        a = a[~xp.isnan(a)]  # cupy v14 behaves poorly for NaN
    return getattr(xp, func)(a)


@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_sinc(xp):
    a = xp.array(TEST_VALUES, dtype=BF16)
    a = a[~xp.isnan(a)]  # cupy v14 behaves poorly for NaN

    res = xp.sinc(a)
    if xp == numpy:
        res = res.astype(BF16)
    return res


@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
@numpy.errstate(over='ignore')
def test_i0(xp):
    a = xp.array(TEST_VALUES, dtype=BF16)
    # CuPy inf's go to NaNs, ignore that
    a = a[~xp.isinf(a)]

    res = xp.i0(a)
    if xp == numpy:
        # NumPy always returns float64.
        res = res.astype(BF16)
    return res


@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_angle(xp):
    # XXX(seberg): CuPy doesn't handle NaN correctly, so don't test.
    a = xp.array(TEST_VALUES[:-1], dtype=BF16)
    res = xp.angle(a, deg=False)
    if xp == numpy:
        res = res.astype(BF16)  # some versions of NumPy return float32
    return res


@pytest.mark.parametrize('func', [
    'add', 'subtract', 'multiply', 'divide', 'power', 'floor_divide',
    'remainder',
    'hypot', 'arctan2', 'logaddexp', 'logaddexp2', 'heaviside', 'fmod',
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


@pytest.mark.parametrize('func', [
    'greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal',
    'copysign', 'nextafter', 'maximum', 'minimum', 'fmax', 'fmin',
])
@testing.numpy_cupy_allclose(rtol=0, atol=0)
def test_binary_exact(xp, func):
    x = xp.asarray(TEST_VALUES)
    y = x[:, None]  # Broadcasting to test all combinations
    with numpy.errstate(all='ignore'):
        return getattr(xp, func)(x, y)


@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_clip(xp):
    # XXX(seberg): Unimplemented in ml_dtypes if this fails add loop.
    a = xp.array([0.0, 2.0, 5.0, 10.0, -5.0], dtype=BF16)
    return xp.clip(a, 1, 4)


@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_modf(xp):
    a = xp.array(TEST_VALUES, dtype=BF16)
    a = a[xp.isfinite(a)]
    frac, integer = xp.modf(a)
    # Stack the results to compare both outputs
    return xp.stack([frac, integer])


@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_frexp(xp):
    a = xp.array(TEST_VALUES, dtype=BF16)
    a = a[xp.isfinite(a)]
    mantissa, exponent = xp.frexp(a)
    # Return mantissa for comparison (exponent is int)
    return mantissa


@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_ldexp(xp):
    a = xp.array([1.0, 2.0, 0.5, -1.0, 0.0], dtype=BF16)
    b = xp.array([1, 2, -1, 3, 0], dtype=xp.int32)
    return xp.ldexp(a, b)


@pytest.mark.parametrize('shapes', [
    # Make size large enough that float32 compute type should matter
    ((10, 20_000), (20_000, 10)),
    ((30_000,), (30_000,)),
])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
@pytest.mark.skipif(runtime.is_hip, reason="Matrix version crashed CI.")
def test_matmul(xp, shapes):
    shape_a, shape_b = shapes
    a = testing.shaped_arange(shape_a, xp, dtype=BF16)
    b = testing.shaped_arange(shape_b, xp, dtype=BF16)
    res = xp.matmul(a, b)
    if xp == numpy:
        # NumPy returns float32 we don't
        res = res.astype(BF16)
    return res


@pytest.mark.parametrize('func', [
    'sum', 'prod', 'min', 'max', 'mean',
    'nansum', 'nanmin', 'nanmax',
    # For some reason, NumPy result includes NaN:
    pytest.param('nanprod', marks=[pytest.mark.xfail]),
])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_reductions(xp, func):
    a = xp.asarray(TEST_VALUES.reshape(3, 4))
    with numpy.errstate(all='ignore'):
        return getattr(xp, func)(a, axis=1)


@pytest.mark.parametrize('func', [
    'sum', 'prod', 'mean', 'nansum',
    # For some reason, NumPy result includes NaN:
    pytest.param('nanprod', marks=[pytest.mark.xfail]),
])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_reductions_dtype(xp, func):
    a = xp.asarray(TEST_VALUES.reshape(3, 4))
    with numpy.errstate(all='ignore'):
        return getattr(xp, func)(a, axis=1, dtype=BF16)


@pytest.mark.parametrize('func', ['sum', 'prod', 'min', 'max', 'mean'])
@pytest.mark.parametrize('data', [
    numpy.arange(1, 10_000),
    numpy.random.uniform(0.5, 2, size=10_000),
])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_reductions_large(xp, func, data):
    # Mainly for sum test that we use a float32 compute type, this makes
    # sense, but NumPy + bfloat16 isn't great about it (at least yet).
    # NOTE(seberg): Using values that underflow to 0 in product returns
    # NaN on CuPy when testing (but also for float32).
    a = xp.asarray(data, dtype=BF16)
    if xp == numpy:
        a = a.astype(numpy.float32)
        with numpy.errstate(over='ignore'):
            res = getattr(xp, func)(a).astype(BF16)
    else:
        res = getattr(xp, func)(a)
    return res


@pytest.mark.parametrize('from_dtype', [
    numpy.float16, numpy.float32, numpy.float64, numpy.int8, numpy.int32])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_astype_to_bfloat16(xp, from_dtype):
    a = xp.asarray(TEST_VALUES)
    if numpy.dtype(from_dtype).kind == "i":
        a = a[xp.isfinite(a)]
    return a.astype(from_dtype).astype(BF16)


@pytest.mark.parametrize('to_dtype', [
    numpy.float16, numpy.float32, numpy.float64, numpy.int8, numpy.int32])
@testing.numpy_cupy_allclose(rtol=TOL, atol=TOL)
def test_astype_from_bfloat16(xp, to_dtype):
    a = xp.asarray(TEST_VALUES)
    if numpy.dtype(to_dtype).kind == "i":
        a = a[xp.isfinite(a)]
    return a.astype(to_dtype)


@pytest.mark.parametrize('func', [
    lambda x: x + ml_dtypes.bfloat16(numpy.inf),
    lambda x: x + x,
    lambda x: cupy.sqrt(x),
    lambda x: x.astype(numpy.float32),
    lambda x: x + ml_dtypes.bfloat16(2),
    lambda x: x.sum(),
])
def test_fusion_basic(func):
    # NOTE(seberg): As of writing v14.0 no attempt made for old-fusion.
    arr = cupy.asarray(TEST_VALUES)
    expected = func(arr)

    fused = cupy._core.new_fusion.Fusion(func, 'bf16_fuse_test')
    actual = fused(arr)
    cupy.testing.assert_allclose(actual, expected, rtol=TOL, atol=TOL)
