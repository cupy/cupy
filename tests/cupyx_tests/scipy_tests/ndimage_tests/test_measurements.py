import unittest
import pytest

import numpy

import cupy
from cupy import testing
import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass


def _generate_binary_structure(rank, connectivity):
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return numpy.array(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    return output <= connectivity


@testing.parameterize(*testing.product({
    'ndim': [1, 2, 3, 4],
    'size': [50, 100],
    'density': [0.2, 0.3, 0.4],
    'connectivity': [None, 2, 3],
    'x_dtype': [bool, numpy.int8, numpy.int32, numpy.int64,
                numpy.float32, numpy.float64],
    'output': [None, numpy.int32, numpy.int64],
    'o_type': [None, 'ndarray']
}))
@testing.gpu
@testing.with_requires('scipy')
class TestLabel(unittest.TestCase):

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_label(self, xp, scp):
        size = int(pow(self.size, 1 / self.ndim))
        x_shape = range(size, size + self.ndim)
        x = xp.zeros(x_shape, dtype=self.x_dtype)
        x[testing.shaped_random(x_shape, xp) < self.density] = 1
        if self.connectivity is None:
            structure = None
        else:
            structure = _generate_binary_structure(self.ndim,
                                                   self.connectivity)
        if self.o_type == 'ndarray' and self.output is not None:
            output = xp.empty(x_shape, dtype=self.output)
            num_features = scp.ndimage.label(x, structure=structure,
                                             output=output)
            return output
        labels, num_features = scp.ndimage.label(x, structure=structure,
                                                 output=self.output)
        return labels


@testing.gpu
@testing.with_requires('scipy')
class TestLabelSpecialCases(unittest.TestCase):

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_label_empty(self, xp, scp):
        x = xp.empty(0)
        labels, num_features = scp.ndimage.label(x)
        return labels

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_label_0d_zero(self, xp, scp):
        x = xp.zeros([])
        labels, num_features = scp.ndimage.label(x)
        return labels

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_label_0d_one(self, xp, scp):
        x = xp.ones([])
        labels, num_features = scp.ndimage.label(x)
        return labels

    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_label_swirl(self, xp, scp):
        x = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
             [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
             [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
             [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
             [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
             [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
             [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        x = xp.array(x)
        labels, num_features = scp.ndimage.label(x)
        return labels


@testing.gpu
@testing.parameterize(*testing.product({
    'op': ['sum', 'mean', 'variance', 'standard_deviation'],
}))
@testing.with_requires('scipy')
class TestNdimage(unittest.TestCase):

    def _make_image(self, shape, xp, dtype):
        if dtype == xp.bool_:
            return testing.shaped_random(shape, xp, dtype=xp.bool_)
        else:
            return testing.shaped_arange(shape, xp, dtype=dtype)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_almost_equal(scipy_name='scp')
    def test_ndimage_single_dim(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        label = testing.shaped_random((100,), xp, dtype=xp.int32, scale=3)
        index = xp.array([0, 1, 2])
        return getattr(scp.ndimage, self.op)(image, label, index)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_almost_equal(scipy_name='scp')
    def test_ndimage_multi_dim(self, xp, scp, dtype):
        image = self._make_image((8, 8, 8), xp, dtype)
        label = testing.shaped_random((8, 8, 8), xp, dtype=xp.int32, scale=3)
        index = xp.array([0, 1, 2])
        return getattr(scp.ndimage, self.op)(image, label, index)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_ndimage_zero_dim(self, xp, scp, dtype):
        image = self._make_image((), xp, dtype)
        label = testing.shaped_random((), xp, dtype=xp.int32, scale=4)
        index = xp.array([1, 2, 3])
        return getattr(scp.ndimage, self.op)(image, label, index)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ndimage_only_input(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        return getattr(scp.ndimage, self.op)(image)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ndimage_no_index(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        label = testing.shaped_random((100,), xp, dtype=xp.int32, scale=3)
        return getattr(scp.ndimage, self.op)(image, label)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_ndimage_scalar_index(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        label = testing.shaped_random((100,), xp, dtype=xp.int32, scale=3)
        return getattr(scp.ndimage, self.op)(image, label, 1)

    @testing.for_dtypes([cupy.complex64, cupy.complex128])
    def test_ndimage_wrong_dtype(self, dtype):
        image = self._make_image((100,), cupy, dtype)
        label = cupy.random.randint(1, 4, dtype=cupy.int32)
        index = cupy.array([1, 2, 3])
        with pytest.raises(TypeError):
            getattr(cupyx.scipy.ndimage, self.op)(image, label, index)

    def test_ndimage_wrong_label_shape(self):
        image = self._make_image((100,), cupy, cupy.int32)
        label = cupy.random.randint(1, 3, dtype=cupy.int32, size=50)
        index = cupy.array([1, 2, 3])
        with pytest.raises(ValueError):
            getattr(cupyx.scipy.ndimage, self.op)(image, label, index)

    def test_ndimage_wrong_image_type(self):
        image = list(range(100))
        label = cupy.random.randint(1, 3, dtype=cupy.int32, size=100)
        index = cupy.array([1, 2, 3])
        with pytest.raises(TypeError):
            getattr(cupyx.scipy.ndimage, self.op)(image, label, index)

    def test_ndimage_wrong_label_type(self):
        image = self._make_image((100,), cupy, cupy.int32)
        label = numpy.random.randint(1, 3, dtype=numpy.int32, size=100)
        index = cupy.array([1, 2, 3])
        with pytest.raises(TypeError):
            getattr(cupyx.scipy.ndimage, self.op)(image, label, index)

    def test_ndimage_wrong_index_type(self):
        image = self._make_image((100,), cupy, cupy.int32)
        label = cupy.random.randint(1, 3, dtype=cupy.int32, size=100)
        index = [1, 2, 3]
        with pytest.raises(TypeError):
            getattr(cupyx.scipy.ndimage, self.op)(image, label, index)

    @testing.numpy_cupy_array_almost_equal(scipy_name='scp')
    def test_ndimage_zero_values(self, xp, scp):
        image = xp.array([])
        label = xp.array([])
        index = xp.array([])
        return getattr(scp.ndimage, self.op)(image, label, index)
