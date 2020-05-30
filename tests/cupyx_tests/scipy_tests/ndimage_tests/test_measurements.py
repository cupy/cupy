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

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_almost_equal(scipy_name='scp')
    def test_ndimage_single_dim(self, xp, scp, dtype):
        image = xp.arange(100, dtype=dtype)
        label = testing.shaped_random((100,), xp, dtype=xp.int32, scale=3)
        index = xp.array([0, 1, 2])
        return getattr(scp.ndimage, self.op)(image, label, index)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_almost_equal(scipy_name='scp')
    def test_ndimage_multi_dim(self, xp, scp, dtype):
        image = xp.arange(512, dtype=dtype).reshape(8, 8, 8)
        label = testing.shaped_random((8, 8, 8), xp, dtype=xp.int32, scale=3)
        index = xp.array([0, 1, 2])
        return getattr(scp.ndimage, self.op)(image, label, index)

    @testing.for_all_dtypes(no_bool=True, no_complex=True, no_float16=True)
    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', decimal=5)
    def test_ndimage_only_input(self, xp, scp, dtype):
        image = xp.arange(100, dtype=dtype)
        return getattr(scp.ndimage, self.op)(image)

    @testing.for_all_dtypes(no_bool=True, no_complex=True, no_float16=True)
    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', decimal=5)
    def test_ndimage_no_index(self, xp, scp, dtype):
        image = xp.arange(100, dtype=dtype)
        label = testing.shaped_random((100,), xp, dtype=xp.int32, scale=3)
        return getattr(scp.ndimage, self.op)(image, label)

    @testing.for_all_dtypes(no_bool=True, no_complex=True, no_float16=True)
    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', decimal=4)
    def test_ndimage_scalar_index(self, xp, scp, dtype):
        image = xp.arange(100, dtype=dtype)
        label = testing.shaped_random((100,), xp, dtype=xp.int32, scale=3)
        return getattr(scp.ndimage, self.op)(image, label, 1)

    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', decimal=1)
    def test_ndimage_only_input_float16(self, xp, scp):
        image = xp.arange(100, dtype=xp.float16)
        return getattr(scp.ndimage, self.op)(image)

    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', decimal=1)
    def test_ndimage_no_index_float16(self, xp, scp):
        image = xp.arange(50, dtype=xp.float16)
        label = testing.shaped_random((50,), xp, dtype=xp.int32, scale=3)
        return getattr(scp.ndimage, self.op)(image, label)

    @testing.numpy_cupy_array_almost_equal(scipy_name='scp', decimal=1)
    def test_ndimage_scalar_index_float16(self, xp, scp):
        image = xp.arange(100, dtype=xp.float16)
        label = testing.shaped_random((100,), xp, dtype=xp.int32, scale=3)
        return getattr(scp.ndimage, self.op)(image, label, 1)

    @testing.for_dtypes([cupy.bool_, cupy.complex64, cupy.complex128])
    def test_ndimage_wrong_dtype(self, dtype):
        image = cupy.arange(100).astype(dtype)
        label = cupy.random.randint(1, 4, dtype=cupy.int32)
        index = cupy.array([1, 2, 3])
        with pytest.raises(TypeError):
            getattr(cupyx.scipy.ndimage, self.op)(image, label, index)

    def test_ndimage_wrong_label_shape(self):
        image = cupy.arange(100, dtype=cupy.int32)
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
        image = cupy.arange(100, dtype=cupy.int32)
        label = numpy.random.randint(1, 3, dtype=numpy.int32, size=100)
        index = cupy.array([1, 2, 3])
        with pytest.raises(TypeError):
            getattr(cupyx.scipy.ndimage, self.op)(image, label, index)

    def test_ndimage_wrong_index_type(self):
        image = cupy.arange(100, dtype=cupy.int32)
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
