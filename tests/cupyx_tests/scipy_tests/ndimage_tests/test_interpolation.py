import unittest

import numpy

import cupy
from cupy import testing
import cupyx.scipy.ndimage

try:
    import scipy.ndimage
except ImportError:
    pass

try:
    import cv2
except ImportError:
    pass


@testing.parameterize(*testing.product({
    'output': [None, numpy.float64, 'f', float, 'empty'],
    'order': [0, 1],
    'mode': ['constant', 'nearest', 'mirror'],
    'cval': [1.0],
    'prefilter': [True],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestMapCoordinates(unittest.TestCase):

    _multiprocess_can_split = True

    def _map_coordinates(self, xp, scp, a, coordinates):
        map_coordinates = scp.ndimage.map_coordinates
        if self.output == 'empty':
            output = xp.empty(coordinates.shape[1], dtype=a.dtype)
            return_value = map_coordinates(a, coordinates, output, self.order,
                                           self.mode, self.cval,
                                           self.prefilter)
            self.assertTrue(return_value is None or return_value is output)
            return output
        else:
            return map_coordinates(a, coordinates, self.output, self.order,
                                   self.mode, self.cval, self.prefilter)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_map_coordinates_float(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        coordinates = testing.shaped_random((a.ndim, 100), xp, dtype)
        return self._map_coordinates(xp, scp, a, coordinates)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_map_coordinates_int(self, xp, scp, dtype):
        if numpy.lib.NumpyVersion(scipy.__version__) < '1.0.0':
            if dtype in (numpy.dtype('l'), numpy.dtype('q')):
                dtype = numpy.int64
            elif dtype in (numpy.dtype('L'), numpy.dtype('Q')):
                dtype = numpy.uint64

        a = testing.shaped_random((100, 100), xp, dtype)
        coordinates = testing.shaped_random((a.ndim, 100), xp, dtype)
        out = self._map_coordinates(xp, scp, a, coordinates)
        float_out = self._map_coordinates(xp, scp, a.astype(xp.float64),
                                          coordinates) % 1
        half = xp.full_like(float_out, 0.5)
        out[xp.isclose(float_out, half, atol=1e-5)] = 0
        return out


@testing.parameterize(*testing.product({
    'matrix_shape': [(2,), (2, 2), (2, 3), (3, 3)],
    'offset': [0.3, [-1.3, 1.3]],
    'output_shape': [None],
    'output': [None, numpy.float64, 'empty'],
    'order': [0, 1],
    'mode': ['constant', 'nearest', 'mirror'],
    'cval': [1.0],
    'prefilter': [True],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestAffineTransform(unittest.TestCase):

    _multiprocess_can_split = True

    def _affine_transform(self, xp, scp, a, matrix):
        ver = numpy.lib.NumpyVersion(scipy.__version__)
        if ver < '1.0.0' and matrix.ndim == 2 and matrix.shape[1] == 3:
            return xp.empty(0)

        if matrix.shape == (3, 3):
            matrix[-1, 0:-1] = 0
            matrix[-1, -1] = 1
        affine_transform = scp.ndimage.affine_transform
        if self.output == 'empty':
            output = xp.empty_like(a)
            return_value = affine_transform(a, matrix, self.offset,
                                            self.output_shape, output,
                                            self.order, self.mode, self.cval,
                                            self.prefilter)
            self.assertTrue(return_value is None or return_value is output)
            return output
        else:
            return affine_transform(a, matrix, self.offset, self.output_shape,
                                    self.output, self.order, self.mode,
                                    self.cval, self.prefilter)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_affine_transform_float(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        matrix = testing.shaped_random(self.matrix_shape, xp, dtype)
        return self._affine_transform(xp, scp, a, matrix)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_affine_transform_int(self, xp, scp, dtype):
        if numpy.lib.NumpyVersion(scipy.__version__) < '1.0.0':
            if dtype in (numpy.dtype('l'), numpy.dtype('q')):
                dtype = numpy.int64
            elif dtype in (numpy.dtype('L'), numpy.dtype('Q')):
                dtype = numpy.uint64

        a = testing.shaped_random((100, 100), xp, dtype)
        matrix = testing.shaped_random(self.matrix_shape, xp, dtype)
        out = self._affine_transform(xp, scp, a, matrix)
        float_out = self._affine_transform(xp, scp, a.astype(xp.float64),
                                           matrix) % 1
        half = xp.full_like(float_out, 0.5)
        out[xp.isclose(float_out, half, atol=1e-5)] = 0
        return out


@testing.gpu
@testing.with_requires('opencv-python')
class TestAffineTransformOpenCV(unittest.TestCase):

    _multiprocess_can_split = True

    @testing.for_float_dtypes(no_float16=True)
    # The precision of cv2.warpAffine is not good because it uses fixed-point
    # arithmetic.
    @testing.numpy_cupy_allclose(atol=0.2)
    def test_affine_transform_opencv(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        matrix = testing.shaped_random((2, 3), xp, dtype)
        if xp == cupy:
            return cupyx.scipy.ndimage.affine_transform(a, matrix, order=1,
                                                        mode='opencv')
        else:
            return cv2.warpAffine(a, matrix, (a.shape[1], a.shape[0]))


@testing.parameterize(*testing.product({
    'angle': [-10, 1000],
    'axes': [(1, 0)],
    'reshape': [False, True],
    'output': [None, numpy.float64, 'empty'],
    'order': [0, 1],
    'mode': ['constant', 'nearest', 'mirror'],
    'cval': [1.0],
    'prefilter': [True],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestRotate(unittest.TestCase):

    _multiprocess_can_split = True

    def _rotate(self, xp, scp, a):
        rotate = scp.ndimage.rotate
        if self.output == 'empty':
            output = rotate(a, self.angle, self.axes,
                            self.reshape, None, self.order,
                            self.mode, self.cval, self.prefilter)
            return_value = rotate(a, self.angle, self.axes,
                                  self.reshape, output, self.order,
                                  self.mode, self.cval, self.prefilter)
            self.assertTrue(return_value is None or return_value is output)
            return output
        else:
            return rotate(a, self.angle, self.axes,
                          self.reshape, self.output, self.order,
                          self.mode, self.cval, self.prefilter)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_rotate_float(self, xp, scp, dtype):
        a = testing.shaped_random((10, 10), xp, dtype)
        return self._rotate(xp, scp, a)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_rotate_int(self, xp, scp, dtype):
        if numpy.lib.NumpyVersion(scipy.__version__) < '1.0.0':
            if dtype in (numpy.dtype('l'), numpy.dtype('q')):
                dtype = numpy.int64
            elif dtype in (numpy.dtype('L'), numpy.dtype('Q')):
                dtype = numpy.uint64

        a = testing.shaped_random((10, 10), xp, dtype)
        out = self._rotate(xp, scp, a)
        float_out = self._rotate(xp, scp, a.astype(xp.float64)) % 1
        half = xp.full_like(float_out, 0.5)
        out[xp.isclose(float_out, half, atol=1e-5)] = 0
        return out


@testing.parameterize(
    {'axes': (-1, -2)},
    {'axes': (0, 1)},
    {'axes': (2, 0)},
    {'axes': (-2, 2)},
)
@testing.gpu
@testing.with_requires('scipy')
class TestRotateAxes(unittest.TestCase):

    _multiprocess_can_split = True

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_rotate_axes(self, xp, scp, dtype):
        a = testing.shaped_random((10, 10, 10), xp, dtype)
        rotate = scp.ndimage.rotate
        return rotate(a, 1, self.axes, order=1)


@testing.gpu
@testing.with_requires('opencv-python')
class TestRotateOpenCV(unittest.TestCase):

    _multiprocess_can_split = True

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=0.3)
    def test_rotate_opencv(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        if xp == cupy:
            return cupyx.scipy.ndimage.rotate(a, 10, reshape=False,
                                              order=1, mode='opencv')
        else:
            matrix = cv2.getRotationMatrix2D((49.5, 49.5), 10, 1)
            return cv2.warpAffine(a, matrix, (a.shape[1], a.shape[0]))


@testing.parameterize(*testing.product({
    'shift': [0.1, -10, (5, -5)],
    'output': [None, numpy.float64, 'empty'],
    'order': [0, 1],
    'mode': ['constant', 'nearest', 'mirror'],
    'cval': [1.0],
    'prefilter': [True],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestShift(unittest.TestCase):

    _multiprocess_can_split = True

    def _shift(self, xp, scp, a):
        shift = scp.ndimage.shift
        if self.output == 'empty':
            output = xp.empty_like(a)
            return_value = shift(a, self.shift, output, self.order,
                                 self.mode, self.cval, self.prefilter)
            self.assertTrue(return_value is None or return_value is output)
            return output
        else:
            return shift(a, self.shift, self.output, self.order,
                         self.mode, self.cval, self.prefilter)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_shift_float(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return self._shift(xp, scp, a)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_shift_int(self, xp, scp, dtype):
        if numpy.lib.NumpyVersion(scipy.__version__) < '1.0.0':
            if dtype in (numpy.dtype('l'), numpy.dtype('q')):
                dtype = numpy.int64
            elif dtype in (numpy.dtype('L'), numpy.dtype('Q')):
                dtype = numpy.uint64

        a = testing.shaped_random((100, 100), xp, dtype)
        out = self._shift(xp, scp, a)
        float_out = self._shift(xp, scp, a.astype(xp.float64)) % 1
        half = xp.full_like(float_out, 0.5)
        out[xp.isclose(float_out, half, atol=1e-5)] = 0
        return out


@testing.gpu
@testing.with_requires('opencv-python')
class TestShiftOpenCV(unittest.TestCase):

    _multiprocess_can_split = True

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=0.2)
    def test_shift_opencv(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        shift = testing.shaped_random((2,), xp, dtype)
        if xp == cupy:
            return cupyx.scipy.ndimage.shift(a, shift, order=1,
                                             mode='opencv')
        else:
            matrix = numpy.array([[1, 0, shift[1]], [0, 1, shift[0]]])
            return cv2.warpAffine(a, matrix, (a.shape[1], a.shape[0]))


@testing.parameterize(*testing.product({
    'zoom': [0.1, 10, (0.1, 10)],
    'output': [None, numpy.float64, 'empty'],
    'order': [0, 1],
    'mode': ['constant', 'nearest', 'mirror'],
    'cval': [1.0],
    'prefilter': [True],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestZoom(unittest.TestCase):

    _multiprocess_can_split = True

    def _zoom(self, xp, scp, a):
        zoom = scp.ndimage.zoom
        if self.output == 'empty':
            output = zoom(a, self.zoom, None, self.order,
                          self.mode, self.cval, self.prefilter)
            return_value = zoom(a, self.zoom, output, self.order,
                                self.mode, self.cval, self.prefilter)
            self.assertTrue(return_value is None or return_value is output)
            return output
        else:
            return zoom(a, self.zoom, self.output, self.order,
                        self.mode, self.cval, self.prefilter)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_zoom_float(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return self._zoom(xp, scp, a)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_zoom_int(self, xp, scp, dtype):
        if numpy.lib.NumpyVersion(scipy.__version__) < '1.0.0':
            if dtype in (numpy.dtype('l'), numpy.dtype('q')):
                dtype = numpy.int64
            elif dtype in (numpy.dtype('L'), numpy.dtype('Q')):
                dtype = numpy.uint64

        a = testing.shaped_random((100, 100), xp, dtype)
        out = self._zoom(xp, scp, a)
        float_out = self._zoom(xp, scp, a.astype(xp.float64)) % 1
        half = xp.full_like(float_out, 0.5)
        out[xp.isclose(float_out, half, atol=1e-5)] = 0
        return out


@testing.parameterize(
    {'zoom': 3},
    {'zoom': 0.3},
)
@testing.gpu
@testing.with_requires('opencv-python')
class TestZoomOpenCV(unittest.TestCase):

    _multiprocess_can_split = True

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-4)
    def test_zoom_opencv(self, xp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        if xp == cupy:
            return cupyx.scipy.ndimage.zoom(a, self.zoom, order=1,
                                            mode='opencv')
        else:
            output_shape = numpy.rint(numpy.multiply(a.shape, self.zoom))
            return cv2.resize(a, tuple(output_shape.astype(int)))
