import math
import unittest

import numpy
import pytest

import cupy
from cupy import testing
import cupyx.scipy.stats  # NOQA
from cupyx.scipy import stats
from cupyx.scipy.stats import distributions

try:
    import scipy.stats  # NOQA
except ImportError:
    pass


@testing.gpu
class TestEntropyBasic(unittest.TestCase):
    def test_entropy_positive(self):
        # See ticket SciPy's gh-497
        pk = cupy.asarray([0.5, 0.2, 0.3])
        qk = cupy.asarray([0.1, 0.25, 0.65])
        eself = stats.entropy(pk, pk)
        edouble = stats.entropy(pk, qk)
        assert 0.0 == eself
        assert edouble >= 0.0

    def test_entropy_base(self):
        pk = cupy.ones(16, float)
        s = stats.entropy(pk, base=2.0)
        assert abs(s - 4.0) < 1.0e-5

        qk = cupy.ones(16, float)
        qk[:8] = 2.0
        s = stats.entropy(pk, qk)
        s2 = stats.entropy(pk, qk, base=2.0)
        assert abs(s / s2 - math.log(2.0)) < 1.0e-5

    def test_entropy_zero(self):
        # Test for SciPy PR-479
        s = stats.entropy(cupy.asarray([0, 1, 2]))
        expected = 0.63651416829481278
        assert abs(float(s) - expected) < 1e-12

    def test_entropy_2d(self):
        pk = cupy.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = cupy.asarray([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        testing.assert_array_almost_equal(
            stats.entropy(pk, qk), [0.1933259, 0.18609809]
        )

    def test_entropy_2d_zero(self):
        pk = cupy.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = cupy.asarray([[0.0, 0.1], [0.3, 0.6], [0.5, 0.3]])
        testing.assert_array_almost_equal(stats.entropy(pk, qk),
                                          [cupy.inf, 0.18609809])

        pk[0][0] = 0.0
        testing.assert_array_almost_equal(
            stats.entropy(pk, qk), [0.17403988, 0.18609809]
        )

    def test_entropy_base_2d_nondefault_axis(self):
        pk = cupy.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        testing.assert_array_almost_equal(
            stats.entropy(pk, axis=1),
            cupy.asarray([0.63651417, 0.63651417, 0.66156324]),
        )

    def test_entropy_2d_nondefault_axis(self):
        pk = cupy.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = cupy.asarray([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        testing.assert_array_almost_equal(
            stats.entropy(pk, qk, axis=1),
            cupy.asarray([0.231049, 0.231049, 0.127706]),
        )

    def test_entropy_raises_value_error(self):
        pk = cupy.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = cupy.asarray([[0.1, 0.2], [0.6, 0.3]])
        with pytest.raises(ValueError):
            stats.entropy(pk, qk)


@testing.parameterize(*(
    testing.product({
        'shape': [(64, ), (16, 15), (14, 4, 10)],
        'base': [None, 10],
        'axis': [None, 0, -1],
        'use_qk': [False, True],
        'normalize': [False, True],
    })
))
@testing.gpu
@testing.with_requires('scipy>=1.4.0')
class TestEntropy(unittest.TestCase):

    def _entropy(self, xp, scp, dtype, shape, use_qk, base, axis, normalize):
        pk = testing.shaped_random(shape, xp, dtype=dtype)
        is_float16 = pk.dtype.char == 'e'
        if use_qk:
            qk = testing.shaped_random(shape, xp, dtype=dtype)
        else:
            qk = None

        if normalize and pk.dtype.kind != 'c':
            # if we don't normalize pk and qk, entropy will do it internally
            norm_axis = 0 if axis is None else axis
            pk = distributions._normalize(pk, norm_axis)
            if qk is not None:
                qk = distributions._normalize(qk, norm_axis)
        res = scp.stats.entropy(pk, qk=qk, base=base, axis=axis)

        float_type = xp.float32 if pk.dtype.char in 'ef' else xp.float64
        if res.ndim > 0:
            # verify expected dtype
            assert res.dtype == float_type

        # Cast back to the floating precision of the input so that the
        # correct rtol is used by numpy_cupy_allclose
        res = xp.asarray(res, xp.float16 if is_float16 else float_type)
        return res

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol={cupy.float16: 1e-3,
                                       cupy.float32: 1e-6,
                                       'default': 1e-15},
                                 scipy_name='scp')
    def test_entropy(self, xp, scp, dtype):
        return self._entropy(xp, scp, dtype, self.shape, self.use_qk,
                             self.base, self.axis, self.normalize)

    @testing.for_complex_dtypes()
    def test_entropy_complex(self, dtype):
        for xp, scp in zip([numpy, cupy], [scipy, cupyx.scipy]):
            with pytest.raises(TypeError):
                return self._entropy(xp, scp, dtype, self.shape, self.use_qk,
                                     self.base, self.axis, self.normalize)
