import unittest
import pytest

import numpy
import cupy
try:
    import scipy.spatial  # NOQA
    import scipy.spatial.distance  # NOQA
    scipy_available = True
except ModuleNotFoundError:
    scipy_available = False
import cupyx.scipy.spatial.distance  # NOQA
try:
    import pylibraft  # NOQA
    pylibraft_available = True
except ModuleNotFoundError:
    pylibraft_available = False
from cupy import testing


@testing.with_requires("scipy")
@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64'],
    'rows': [20, 100],
    'cols': [20, 100],
    'metric': ['euclidean', 'cityblock', 'canberra', 'chebyshev',
               'hamming', 'correlation', 'jensenshannon', 'russellrao',
               "minkowski", "cosine", "sqeuclidean"],
    'p': [2.0],
    'order': ["C", "F"]
}))
@pytest.mark.skipif(cupy.cuda.runtime.is_hip, reason="tests for CUDA only")
@pytest.mark.skipif(not scipy_available or not pylibraft_available,
                    reason='requires scipy and pylibraft')
class TestCdist(unittest.TestCase):

    def _make_matrix(self, xp, dtype, order):
        shape = (self.rows, self.cols)
        return testing.shaped_random(shape, xp, dtype=dtype,
                                     scale=1, order=order)

    @testing.numpy_cupy_array_almost_equal(decimal=4, scipy_name='scp')
    def test_cdist_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        b = self._make_matrix(xp, self.dtype, self.order)

        # RussellRao expects boolean arrays
        if self.metric == "russellrao":
            a[a < 0.5] = 0
            a[a >= 0.5] = 1
            b[b < 0.5] = 0
            b[b >= 0.5] = 1

        # JensenShannon expects probability arrays
        elif self.metric == "jensenshannon":
            a_n = a
            b_n = b
            if xp == cupy:
                a_n = a_n.get()
                b_n = b_n.get()

            # l1 normalization is different between cupy and numpy
            # so use numpy.
            norm = numpy.sum(a_n, axis=1)
            a = (a_n.T / norm).T
            if xp == cupy:
                a = cupy.asarray(a)

            norm = numpy.sum(b_n, axis=1)
            b = (b_n.T / norm).T
            if xp == cupy:
                b = cupy.asarray(b)

        if self.metric == 'minkowski':
            out = scp.spatial.distance.cdist(a, b, metric=self.metric,
                                             p=self.p).astype(self.dtype)
        else:
            out = scp.spatial.distance.cdist(a, b, metric=self.metric)\
                .astype(self.dtype)

        print(str(out.shape))
        return out


@testing.with_requires("scipy")
@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64'],
    'rows': [20, 100],
    'cols': [20, 100],
    'p': [1.0, 2.0, 3.0],
    'order': ["C", "F"]
}))
@pytest.mark.skipif(cupy.cuda.runtime.is_hip, reason="tests for CUDA only")
@pytest.mark.skipif(not scipy_available or not pylibraft_available,
                    reason='requires scipy and pylibraft')
class TestPdist:

    def _make_matrix(self, xp, dtype, order):
        shape = (self.rows, self.cols)
        return testing.shaped_random(shape, xp, dtype=dtype,
                                     scale=1, order=order)

    @testing.numpy_cupy_array_almost_equal(decimal=4, scipy_name='scp')
    def test_pdist_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.pdist(a).astype(self.dtype)
        return out


@testing.with_requires("scipy")
@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64'],
    'rows': [20, 100],
    'cols': [20, 100],
    'p': [1.0, 2.0, 3.0],
    'order': ["C", "F"]
}))
@pytest.mark.skipif(cupy.cuda.runtime.is_hip, reason="tests for CUDA only")
@pytest.mark.skipif(not scipy_available or not pylibraft_available,
                    reason='requires scipy and pylibraft')
class TestDistanceMatrix(unittest.TestCase):

    def _make_matrix(self, xp, dtype, order):
        shape = (self.rows, self.cols)
        return testing.shaped_random(shape, xp, dtype=dtype,
                                     scale=1, order=order)

    @testing.numpy_cupy_array_almost_equal(decimal=4, scipy_name='scp')
    def test_distance_matrix_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance_matrix(a, a, p=self.p).astype(self.dtype)
        return out


@testing.with_requires("scipy")
@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64'],
    'cols': [20, 100],
    'p': [1.0, 2.0, 3.0],
    'order': ["C", "F"]
}))
@pytest.mark.skipif(cupy.cuda.runtime.is_hip, reason="tests for CUDA only")
@pytest.mark.skipif(not scipy_available or not pylibraft_available,
                    reason='requires scipy and pylibraft')
class TestDistanceFunction(unittest.TestCase):

    def _make_matrix(self, xp, dtype, order):
        shape = (1, self.cols) if xp is cupy else (self.cols,)
        return testing.shaped_random(shape, xp, dtype=dtype,
                                     scale=1, order=order)

    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_minkowski_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.minkowski(a, a, p=self.p)
        return out

    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_canberra_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.canberra(a, a)
        return out

    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_chebyshev_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.chebyshev(a, a)
        return out

    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_cityblock_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.cityblock(a, a)
        return out

    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', type_check=False)
    def test_correlation_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.correlation(a, a)
        return xp.asarray(out)

    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', type_check=False)
    def test_cosine_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.cosine(a, a)
        return xp.asarray(out)

    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_hamming_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.hamming(a, a)
        return out

    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_euclidean_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.euclidean(a, a)
        return out

    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_jensenshannon_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.jensenshannon(a, a)
        return out

    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', type_check=False)
    def test_russellrao_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        a[a < 0.5] = 0
        a[a >= 0.5] = 1
        out = scp.spatial.distance.russellrao(a, a)
        return xp.asarray(out)

    @testing.numpy_cupy_equal(scipy_name='scp')
    def test_sqeuclidean_(self, xp, scp):

        a = self._make_matrix(xp, self.dtype, self.order)
        out = scp.spatial.distance.sqeuclidean(a, a)
        return out
