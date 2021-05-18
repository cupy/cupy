import warnings

import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing
from cupy import _util
from cupy._core import _accelerator
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
class TestLabel:

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
class TestLabelSpecialCases:

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
    'op': ['sum', 'mean', 'variance', 'standard_deviation', 'center_of_mass'],
}))
@testing.with_requires('scipy')
class TestStats:

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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', _util.PerformanceWarning)
            result = op(image, labels, index)
        if self.op == 'center_of_mass':
            assert isinstance(result, list)
            # assert isinstance(result[0], tuple)
            assert len(result[0]) == image.ndim
            result = xp.asarray(result)
        return result

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_multi_dim(self, xp, scp, dtype):
        image = self._make_image((8, 8, 8), xp, dtype)
        labels = testing.shaped_random((8, 8, 8), xp, dtype=xp.int32, scale=4)
        index = xp.array([1, 2, 3])
        op = getattr(scp.ndimage, self.op)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', _util.PerformanceWarning)
            result = op(image, labels, index)
        if self.op == 'center_of_mass':
            assert isinstance(result, list)
            # assert isinstance(result[0], tuple)
            assert len(result[0]) == image.ndim
            result = xp.asarray(result)
        return result

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_broadcast_labels(self, xp, scp, dtype):
        # 1d label will be broadcast to 2d
        image = self._make_image((16, 6), xp, dtype)
        labels = xp.asarray([1, 0, 2, 2, 2, 0], dtype=xp.int32)
        op = getattr(scp.ndimage, self.op)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', _util.PerformanceWarning)
            result = op(image, labels)
        return result

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_broadcast_labels2(self, xp, scp, dtype):
        # 1d label will be broadcast to 2d
        image = self._make_image((16, 6), xp, dtype)
        labels = xp.asarray([1, 0, 2, 2, 2, 0], dtype=xp.int32)
        index = 2
        op = getattr(scp.ndimage, self.op)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', _util.PerformanceWarning)
            result = op(image, labels, index)
        return result

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_zero_dim(self, xp, scp, dtype):
        image = self._make_image((), xp, dtype)
        labels = testing.shaped_random((), xp, dtype=xp.int32, scale=4)
        index = xp.array([1, 2, 3])
        op = getattr(scp.ndimage, self.op)
        if self.op == 'center_of_mass':
            # SciPy doesn't handle 0-dimensional array input for center_of_mass
            with pytest.raises(IndexError):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', _util.PerformanceWarning)
                    op(image, labels, index)
            return xp.array([])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', _util.PerformanceWarning)
            result = op(image, labels, index)
        return result

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_only_input(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        op = getattr(scp.ndimage, self.op)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', _util.PerformanceWarning)
            result = op(image)
        return result

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_no_index(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        labels = testing.shaped_random((100,), xp, dtype=xp.int32, scale=4)
        op = getattr(scp.ndimage, self.op)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', _util.PerformanceWarning)
            result = op(image, labels)
        return result

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_scalar_index(self, xp, scp, dtype):
        image = self._make_image((100,), xp, dtype)
        labels = testing.shaped_random((100,), xp, dtype=xp.int32, scale=4)
        op = getattr(scp.ndimage, self.op)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', _util.PerformanceWarning)
            result = op(image, labels, 1)
        return result

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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', _util.PerformanceWarning)
            result = op(image, labels, index)
        return result


@testing.gpu
@testing.parameterize(*testing.product({
    'op': ['maximum', 'median', 'minimum', 'maximum_position',
           'minimum_position', 'extrema'],
    'labels': [None, 5, 50],
    'index': [None, 1, 'all', 'subset'],
    'shape': [(512,), (32, 64)],
    'enable_cub': [True, False],
}))
@testing.with_requires('scipy')
class TestMeasurementsSelect:

    @pytest.fixture(autouse=True)
    def with_accelerators(self):
        old_accelerators = _accelerator.get_routine_accelerators()
        if self.enable_cub:
            _accelerator.set_routine_accelerators(['cub'])
        else:
            _accelerator.set_routine_accelerators([])
        yield
        _accelerator.set_routine_accelerators(old_accelerators)

    def _hip_skip_invalid_condition(self):
        if (runtime.is_hip
                and self.op == 'extrema'
                and (self.index is None
                     or (self.index == 1 and self.labels in [None, 5])
                     or (self.index in ['all', 'subset']
                         and self.labels is None))):
            pytest.xfail('ROCm/HIP may have a bug')

    # no_bool=True due to https://github.com/scipy/scipy/issues/12836
    @testing.for_all_dtypes(no_complex=True, no_bool=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_measurements_select(self, xp, scp, dtype):
        self._hip_skip_invalid_condition()

        shape = self.shape
        rstate = numpy.random.RandomState(0)
        # scale must be small enough to avoid potential integer overflow due to
        # https://github.com/scipy/scipy/issues/12836
        x = testing.shaped_random(shape, xp=xp, dtype=dtype, scale=32)
        non_unique = xp.unique(x).size < x.size

        if (self.op in ['minimum_position', 'maximum_position'] and
                non_unique and self.index is not None):
            # skip cases with non-unique min or max position
            return xp.array([])

        if self.labels is None:
            labels = self.labels
        else:
            labels = rstate.choice(self.labels, x.size).reshape(shape) + 1
            labels = xp.asarray(labels)
        if self.index is None or isinstance(self.index, int):
            index = self.index
        elif self.index == 'all':
            if self.labels is not None:
                index = xp.arange(1, self.labels + 1, dtype=cupy.intp)
            else:
                index = None
        elif self.index == 'subset':
            if self.labels is not None:
                index = xp.arange(1, self.labels + 1, dtype=cupy.intp)[1::2]
            else:
                index = None
        func = getattr(scp.ndimage, self.op)
        result = func(x, labels, index)
        if self.op == 'extrema':
            if non_unique and self.index is not None:
                # omit comparison of minimum_position, maximum_position
                result = [xp.asarray(r) for r in result[:2]]
            else:
                result = [xp.asarray(r) for r in result]
        else:
            if isinstance(result, list):
                # convert list of coordinate tuples to an array for comparison
                result = xp.asarray(result)
        return result


@testing.gpu
@testing.parameterize(*testing.product({
    'labels': [None, 4, 6],
    'index': [None, [0, 2], [3, 1, 0], [1]],
    'shape': [(200,), (16, 20)],
}))
@testing.with_requires('scipy')
class TestHistogram():

    def _make_image(self, shape, xp, dtype, scale):
        return testing.shaped_random(shape, xp, dtype=dtype, scale=scale)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_allclose(scipy_name='scp')
    def test_histogram(self, xp, scp, dtype):
        nbins = 5
        minval = 0
        maxval = 10
        image = self._make_image(self.shape, xp, dtype, scale=maxval)
        labels = self.labels
        index = self.index
        if labels is not None:
            labels = testing.shaped_random(self.shape, xp, dtype=xp.int32,
                                           scale=self.labels)
        if index is not None:
            index = xp.array(index)
        op = getattr(scp.ndimage, 'histogram')
        if index is not None and labels is None:
            # cannot give an index array without labels
            with pytest.raises(ValueError):
                op(image, minval, maxval, nbins, labels, index)
            return xp.asarray([])
        result = op(image, minval, maxval, nbins, labels, index)
        if index is None:
            return result
        # stack 1d arrays into a single array for comparison
        return xp.stack(result)


@testing.gpu
@testing.parameterize(*testing.product({
    'labels': [None, 4],
    'index': [None, [0, 2], [3, 1, 0], [1]],
    'shape': [(200,), (16, 20)],
    'dtype': [numpy.float64, 'same'],
    'default': [0, 3],
    'pass_positions': [True, False],
}))
@testing.with_requires('scipy')
class TestLabeledComprehension():

    def _make_image(self, shape, xp, dtype, scale):
        if dtype == xp.bool_:
            return testing.shaped_random(shape, xp, dtype=xp.bool_)
        else:
            return testing.shaped_random(shape, xp, dtype=dtype, scale=scale)

    @testing.for_all_dtypes(no_bool=True, no_complex=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', rtol=1e-4, atol=1e-4)
    def test_labeled_comprehension(self, xp, scp, dtype):
        image = self._make_image(self.shape, xp, dtype, scale=101)
        labels = self.labels
        index = self.index
        if labels is not None:
            labels = testing.shaped_random(self.shape, xp, dtype=xp.int32,
                                           scale=4)
        if index is not None:
            index = xp.array(index)

        if self.pass_positions:
            # simple function that takes a positions argument
            def func(x, pos):
                return xp.sum(x + pos > 50)
        else:
            # simple function to apply to each lable
            func = xp.sum

        op = getattr(scp.ndimage, 'labeled_comprehension')
        dtype == image.dtype if self.dtype == 'same' else self.dtype
        if index is not None and labels is None:
            # cannot give an index array without labels
            with pytest.raises(ValueError):
                op(image, labels, index, func, dtype, self.default,
                   self.pass_positions)
            return xp.asarray([])
        return op(image, labels, index, func, dtype, self.default,
                  self.pass_positions)
