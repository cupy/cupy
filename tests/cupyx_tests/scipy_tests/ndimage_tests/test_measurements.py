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
class TestStats(unittest.TestCase):

    def _make_image(self, shape, xp, dtype):
        if dtype == xp.bool_:
            return testing.shaped_random(shape, xp, dtype=xp.bool_)
        else:
            return testing.shaped_arange(shape, xp, dtype=dtype)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_single_dim(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        labels = testing.shaped_random((100,), xp, dtype=xp.int32, scale=4)
        index = xp.array([1, 2, 3])
        op = getattr(scp.ndimage, self.op)
        return op(image, labels, index)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_multi_dim(self, xp, scp, dtype):
        image = self._make_image((8, 8, 8), xp, dtype)
        labels = testing.shaped_random((8, 8, 8), xp, dtype=xp.int32, scale=4)
        index = xp.array([1, 2, 3])
        op = getattr(scp.ndimage, self.op)
        return op(image, labels, index)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_broadcast_labels(self, xp, scp, dtype):
        # 1d label will be broadcast to 2d
        image = self._make_image((16, 6), xp, dtype)
        labels = xp.asarray([1, 0, 2, 2, 2, 0], dtype=xp.int32)
        op = getattr(scp.ndimage, self.op)
        return op(image, labels)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_broadcast_labels2(self, xp, scp, dtype):
        # 1d label will be broadcast to 2d
        image = self._make_image((16, 6), xp, dtype)
        labels = xp.asarray([1, 0, 2, 2, 2, 0], dtype=xp.int32)
        index = 2
        op = getattr(scp.ndimage, self.op)
        return op(image, labels, index)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_dim(self, xp, scp, dtype):
        image = self._make_image((), xp, dtype)
        labels = testing.shaped_random((), xp, dtype=xp.int32, scale=4)
        index = xp.array([1, 2, 3])
        op = getattr(scp.ndimage, self.op)
        return op(image, labels, index)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_only_input(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        op = getattr(scp.ndimage, self.op)
        return op(image)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_no_index(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        labels = testing.shaped_random((100,), xp, dtype=xp.int32, scale=4)
        op = getattr(scp.ndimage, self.op)
        return op(image, labels)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_scalar_index(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        labels = testing.shaped_random((100,), xp, dtype=xp.int32, scale=4)
        op = getattr(scp.ndimage, self.op)
        return op(image, labels, 1)

    @testing.for_complex_dtypes()
    def test_invalid_image_dtype(self, dtype):
        image = self._make_image((100,), cupy, dtype)
        labels = testing.shaped_random((100,), cupy, dtype=cupy.int32, scale=4)
        index = cupy.array([1, 2, 3])
        op = getattr(cupyx.scipy.ndimage, self.op)
        with pytest.raises(TypeError):
            op(image, labels, index)

    def test_invalid_image_type(self):
        image = list(range(100))
        labels = testing.shaped_random((100,), cupy, dtype=cupy.int32, scale=4)
        index = cupy.array([1, 2, 3])
        op = getattr(cupyx.scipy.ndimage, self.op)
        with pytest.raises(TypeError):
            op(image, labels, index)

    def test_invalid_labels_shape(self):
        image = self._make_image((100,), cupy, cupy.int32)
        labels = testing.shaped_random((50,), cupy, dtype=cupy.int32, scale=4)
        index = cupy.array([1, 2, 3])
        op = getattr(cupyx.scipy.ndimage, self.op)
        with pytest.raises(ValueError):
            op(image, labels, index)

    def test_invalid_labels_type(self):
        image = self._make_image((100,), cupy, cupy.int32)
        labels = numpy.random.randint(0, 4, dtype=numpy.int32, size=100)
        index = cupy.array([1, 2, 3])
        op = getattr(cupyx.scipy.ndimage, self.op)
        with pytest.raises(TypeError):
            op(image, labels, index)

    def test_invalid_index_type(self):
        image = self._make_image((100,), cupy, cupy.int32)
        labels = testing.shaped_random((100,), cupy, dtype=cupy.int32, scale=4)
        index = [1, 2, 3]
        op = getattr(cupyx.scipy.ndimage, self.op)
        with pytest.raises(TypeError):
            op(image, labels, index)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal(scipy_name='scp')
    def test_no_values(self, xp, scp, dtype):
        image = xp.array([], dtype=dtype)
        labels = xp.array([])
        index = xp.array([])
        op = getattr(scp.ndimage, self.op)
        return op(image, labels, index)
