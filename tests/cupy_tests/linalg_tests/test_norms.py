import unittest

import numpy

from cupy import testing


@testing.gpu
class TestTrace(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_trace(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return a.trace(1, 3, 2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_trace(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return xp.trace(a, 1, 3, 2)


@testing.parameterize(*testing.product({
    'shape': [(1,), (2,)],
    'ord': [-numpy.Inf, -2, -1, 0, 1, 2, 3, numpy.Inf],
    'axis': [0, None],
    'keepdims': [True, False],
}) + testing.product({
    'shape': [(1, 2), (2, 2)],
    'ord': [-numpy.Inf, -1, 1, numpy.Inf, 'fro'],
    'axis': [(0, 1), None],
    'keepdims': [True, False],
}) + testing.product({
    'shape': [(2, 2, 2)],
    'ord': [-numpy.Inf, -2, -1, 0, 1, 2, 3, numpy.Inf],
    'axis': [0, 1, 2],
    'keepdims': [True, False],
}) + testing.product({
    'shape': [(2, 2, 2)],
    'ord': [-numpy.Inf, -1, 1, numpy.Inf, 'fro'],
    'axis': [(0, 1), (0, 2), (1, 2)],
    'keepdims': [True, False],
})
)
@testing.gpu
@testing.with_requires('numpy>=1.11.2')  # The old version dtype is strange
class TestNorm(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_norm(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        with testing.NumpyError(divide='ignore'):
            return xp.linalg.norm(a, self.ord, self.axis, self.keepdims)
