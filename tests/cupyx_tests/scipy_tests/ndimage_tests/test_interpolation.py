import numpy
import pytest

import cupy
from cupy.cuda import runtime
from cupy import testing
import cupyx.scipy.ndimage
from cupyx.scipy.ndimage import _util

try:
    import scipy
    import scipy.ndimage
    scipy_version = numpy.lib.NumpyVersion(scipy.__version__)
except ImportError:
    pass

try:
    import cv2
except ImportError:
    pass

# testing these modes can only be tested against SciPy >= 1.6.0+
scipy16_modes = ['wrap', 'grid-wrap', 'reflect', 'grid-mirror',
                 'grid-constant']
# these modes are okay to test on older SciPy
legacy_modes = ['constant', 'nearest', 'mirror']


def _conditional_scipy_version_skip(mode, order):
    if ((mode in scipy16_modes or (mode != 'mirror' and order > 1)) and
            (scipy_version < '1.6.0')):
        pytest.skip(
            'SciPy >= 1.6.0 needed to test this mode/order combination.')


@testing.parameterize(*testing.product({
    'output': [None, numpy.float64, 'f', float, 'empty'],
    'order': [0, 1, 2, 3, 4, 5],
    'mode': ['constant', 'nearest', 'mirror'] + scipy16_modes,
    'cval': [1.0],
    'prefilter': [True],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestMapCoordinates:

    _multiprocess_can_split = True

    def _map_coordinates(self, xp, scp, a, coordinates):
        _conditional_scipy_version_skip(self.mode, self.order)
        map_coordinates = scp.ndimage.map_coordinates
        if self.output == 'empty':
            output = xp.empty(coordinates.shape[1:], dtype=a.dtype)
            return_value = map_coordinates(a, coordinates, output, self.order,
                                           self.mode, self.cval,
                                           self.prefilter)
            assert return_value is None or return_value is output
            return output
        else:
            return map_coordinates(a, coordinates, self.output, self.order,
                                   self.mode, self.cval, self.prefilter)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-4, scipy_name='scp')
    def test_map_coordinates_float(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        coordinates = testing.shaped_random((a.ndim, 100), xp, dtype)
        return self._map_coordinates(xp, scp, a, coordinates)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-4, scipy_name='scp')
    @testing.with_requires('scipy>=1.6.0')
    def test_map_coordinates_complex_float(self, xp, scp, dtype):
        # promote output to a complex dtype
        if self.output == numpy.float64:
            self.output = numpy.complex128
        elif self.output == float:
            self.output = complex
        elif self.output == 'f':
            self.output = 'F'
        a = testing.shaped_random((100, 100), xp, dtype)
        coordinates = testing.shaped_random((a.ndim, 100), xp, xp.float64)
        return self._map_coordinates(xp, scp, a, coordinates)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-4, scipy_name='scp')
    def test_map_coordinates_fortran_order(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        coordinates = testing.shaped_random((a.ndim, 100), xp, dtype)
        a = xp.asfortranarray(a)
        coordinates = xp.asfortranarray(coordinates)
        return self._map_coordinates(xp, scp, a, coordinates)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-4, scipy_name='scp')
    def test_map_coordinates_float_nd_coords(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        coordinates = testing.shaped_random((a.ndim, 10, 10), xp, dtype,
                                            scale=99.0)
        return self._map_coordinates(xp, scp, a, coordinates)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-4, scipy_name='scp')
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
    'order': [0, 1, 2, 3, 4, 5],
    'mode': ['constant', 'nearest', 'mirror'] + scipy16_modes,
}))
@testing.gpu
@testing.with_requires('scipy')
class TestMapCoordinatesHalfInteger:

    def _map_coordinates(self, xp, scp, a, coordinates):
        _conditional_scipy_version_skip(self.mode, self.order)
        map_coordinates = scp.ndimage.map_coordinates
        return map_coordinates(a, coordinates, None, self.order, self.mode)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-4, scipy_name='scp')
    def test_map_coordinates_float(self, xp, scp, dtype):
        # Half integer coordinate rounding test case from:
        # https://github.com/cupy/cupy/issues/4550
        a = testing.shaped_arange((4, 3), xp, dtype)
        coordinates = xp.array([[0.5, 2], [0.5, 1]])
        return self._map_coordinates(xp, scp, a, coordinates)


@testing.parameterize(*testing.product({
    'matrix_shape': [(2,), (2, 2), (2, 3), (3, 3)],
    'offset': [0.3, [-1.3, 1.3]],
    'output_shape': [None],
    'output': [None, numpy.float64, 'empty'],
    'order': [0, 1, 2, 3, 4, 5],
    'mode': legacy_modes + scipy16_modes,
    'cval': [1.0],
    'prefilter': [False, True],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestAffineTransform:

    _multiprocess_can_split = True

    def _affine_transform(self, xp, scp, a, matrix):
        _conditional_scipy_version_skip(self.mode, self.order)
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
            assert return_value is None or return_value is output
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

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    @testing.with_requires('scipy>=1.6.0')
    def test_affine_transform_complex_float(self, xp, scp, dtype):
        if self.output == numpy.float64:
            # must promote output to a complex dtype
            self.output = numpy.complex128
        a = testing.shaped_random((100, 100), xp, dtype)
        matrix = testing.shaped_random(self.matrix_shape, xp, xp.float64)
        return self._affine_transform(xp, scp, a, matrix)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_affine_transform_fortran_order(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        a = xp.asfortranarray(a)
        matrix = testing.shaped_random(self.matrix_shape, xp, dtype)
        matrix = xp.asfortranarray(matrix)
        return self._affine_transform(xp, scp, a, matrix)

    def _hip_skip_invalid_condition(self):
        if (runtime.is_hip
                and self.matrix_shape in [(2,), (2, 2)]
                and self.order in [2, 3, 4, 5]
                and self.output in [None, 'empty']
                and self.prefilter):
            pytest.xfail('ROCm/HIP may have a bug')

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_affine_transform_int(self, xp, scp, dtype):
        self._hip_skip_invalid_condition()

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
@testing.with_requires('scipy')
class TestAffineExceptions:

    def test_invalid_affine_ndim(self):
        ndimage_modules = (scipy.ndimage, cupyx.scipy.ndimage)
        for (xp, ndi) in zip((numpy, cupy), ndimage_modules):
            x = xp.ones((8, 8, 8))
            with pytest.raises(RuntimeError):
                ndi.affine_transform(x, xp.ones((3, 3, 3)))
            with pytest.raises(RuntimeError):
                ndi.affine_transform(x, xp.ones(()))

    def test_invalid_affine_shape(self):
        ndimage_modules = (scipy.ndimage, cupyx.scipy.ndimage)
        for (xp, ndi) in zip((numpy, cupy), ndimage_modules):
            x = xp.ones((8, 8, 8))
            with pytest.raises(RuntimeError):
                ndi.affine_transform(x, xp.ones((0, 3)))
            with pytest.raises(RuntimeError):
                ndi.affine_transform(x, xp.eye(x.ndim - 1))
            with pytest.raises(RuntimeError):
                ndi.affine_transform(x, xp.eye(x.ndim + 2))
            with pytest.raises(RuntimeError):
                ndi.affine_transform(x, xp.eye(x.ndim)[:, :-1])

    @testing.with_requires('scipy>=1.6.0')
    def test_invalid_output_dtype(self):
        # real output array with complex input is not allowed
        ndimage_modules = (scipy.ndimage, cupyx.scipy.ndimage)
        for (xp, ndi) in zip((numpy, cupy), ndimage_modules):
            x = xp.ones((8, 8, 8), dtype=numpy.complex128)
            output = xp.ones_like(x, dtype=x.real.dtype)
            with pytest.raises(RuntimeError):
                ndi.affine_transform(x, xp.ones((0, 3)), output=output)


@testing.gpu
@testing.with_requires('opencv-python')
class TestAffineTransformOpenCV:

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


@testing.parameterize(*(
    testing.product({
        'angle': [-10, 1000],
        'axes': [(1, 0)],
        'reshape': [False, True],
        'output': [None, numpy.float64, 'empty'],
        'order': [0, 1],
        'mode': legacy_modes,
        'cval': [1.0],
        'prefilter': [True],
    }) + testing.product({
        'angle': [-15],
        'axes': [(1, 0)],
        'reshape': [False],
        'output': [None],
        'order': [0, 1, 3],
        'mode': legacy_modes + scipy16_modes,
        'cval': [1.0],
        'prefilter': [True],
    })
))
@testing.gpu
@testing.with_requires('scipy')
class TestRotate:

    _multiprocess_can_split = True

    def _rotate(self, xp, scp, a):
        _conditional_scipy_version_skip(self.mode, self.order)
        rotate = scp.ndimage.rotate
        if self.output == 'empty':
            output = rotate(a, self.angle, self.axes,
                            self.reshape, None, self.order,
                            self.mode, self.cval, self.prefilter)
            return_value = rotate(a, self.angle, self.axes,
                                  self.reshape, output, self.order,
                                  self.mode, self.cval, self.prefilter)
            assert return_value is None or return_value is output
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

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    @testing.with_requires('scipy>=1.6.0')
    def test_rotate_complex_float(self, xp, scp, dtype):
        if self.output == numpy.float64:
            self.output = numpy.complex128
        a = testing.shaped_random((10, 10), xp, dtype)
        return self._rotate(xp, scp, a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_rotate_fortran_order(self, xp, scp, dtype):
        a = testing.shaped_random((10, 10), xp, dtype)
        a = xp.asfortranarray(a)
        return self._rotate(xp, scp, a)

    def _hip_skip_invalid_condition(self):
        if runtime.is_hip:
            if (self.angle in [-10, 1000]
                    and self.mode in ['constant', 'nearest', 'mirror']
                    and self.output == numpy.float64
                    and self.reshape):
                pytest.xfail('ROCm/HIP may have a bug')
            if (self.angle == -15
                    and self.mode in [
                        'nearest', 'grid-wrap', 'reflect', 'grid-mirror']
                    and self.order == 3):
                pytest.xfail('ROCm/HIP may have a bug')

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_rotate_int(self, xp, scp, dtype):
        self._hip_skip_invalid_condition()

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


@testing.gpu
# Scipy older than 1.3.0 raises IndexError instead of ValueError
@testing.with_requires('scipy>=1.3.0')
class TestRotateExceptions:

    def test_rotate_invalid_plane(self):
        ndimage_modules = (scipy.ndimage, cupyx.scipy.ndimage)
        for (xp, ndi) in zip((numpy, cupy), ndimage_modules):
            x = xp.ones((8, 8, 8))
            angle = 15
            with pytest.raises(ValueError):
                ndi.rotate(x, angle, [0, x.ndim])
            with pytest.raises(ValueError):
                ndi.rotate(x, angle, [-(x.ndim + 1), 1])


@testing.parameterize(
    {'axes': (-1, -2)},
    {'axes': (0, 1)},
    {'axes': (2, 0)},
    {'axes': (-2, 2)},
)
@testing.gpu
@testing.with_requires('scipy')
class TestRotateAxes:

    _multiprocess_can_split = True

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_rotate_axes(self, xp, scp, dtype):
        a = testing.shaped_random((10, 10, 10), xp, dtype)
        rotate = scp.ndimage.rotate
        return rotate(a, 1, self.axes, order=1)


@testing.gpu
@testing.with_requires('opencv-python')
class TestRotateOpenCV:

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


@testing.parameterize(*(
    testing.product({
        'shift': [0.1, -10, (5, -5)],
        'output': [None, numpy.float64, 'empty'],
        'order': [0, 1, 3],
        'mode': legacy_modes + scipy16_modes,
        'cval': [1.0],
        'prefilter': [True],
    }) + testing.product({
        'shift': [0.1, ],
        'output': [None, numpy.float64, 'empty'],
        'order': [0, 1, 3],
        'mode': ['constant', ],
        'cval': [cupy.nan, cupy.inf, -cupy.inf],
        'prefilter': [True],
    })
))
@testing.gpu
@testing.with_requires('scipy')
class TestShift:

    _multiprocess_can_split = True

    def _shift(self, xp, scp, a):
        shift = scp.ndimage.shift
        _conditional_scipy_version_skip(self.mode, self.order)
        if self.output == 'empty':
            output = xp.empty_like(a)
            return_value = shift(a, self.shift, output, self.order,
                                 self.mode, self.cval, self.prefilter)
            assert return_value is None or return_value is output
            return output
        else:
            return shift(a, self.shift, self.output, self.order,
                         self.mode, self.cval, self.prefilter)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_shift_float(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return self._shift(xp, scp, a)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    @testing.with_requires('scipy>=1.6.0')
    def test_shift_complex_float(self, xp, scp, dtype):
        if self.output == numpy.float64:
            self.output = numpy.complex128
        a = testing.shaped_random((100, 100), xp, dtype)
        return self._shift(xp, scp, a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_shift_fortran_order(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        a = xp.asfortranarray(a)
        return self._shift(xp, scp, a)

    def _hip_skip_invalid_condition(self):
        if (runtime.is_hip
                and self.cval == 1.0
                and self.order == 3
                and self.output in [None, 'empty']
                and self.shift == 0.1):
            pytest.xfail('ROCm/HIP may have a bug')

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_shift_int(self, xp, scp, dtype):
        self._hip_skip_invalid_condition()

        if self.mode == 'constant' and not xp.isfinite(self.cval):
            if self.output is None or self.output == 'empty':
                # Non-finite cval with integer output array is not supported
                # CuPy exception is tested in TestInterpolationInvalidCval
                return xp.asarray([])

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


# non-finite cval with integer valued output is not allowed for CuPy
@testing.parameterize(*testing.product({
    'output': [None, numpy.float64, numpy.int32, 'empty'],
    'order': [0, 1],
    'mode': ['constant', 'nearest'],
    'cval': [cupy.nan, cupy.inf, -cupy.inf],
}))
@testing.gpu
class TestInterpolationInvalidCval:

    def _prep_output(self, a):
        if self.output == 'empty':
            return cupy.zeros_like(a)
        return self.output

    @testing.for_int_dtypes(no_bool=True)
    def test_shift(self, dtype):
        a = cupy.ones((32,), dtype=dtype)
        shift = cupyx.scipy.ndimage.shift
        output = self._prep_output(a)
        if _util._is_integer_output(output, a) and self.mode == 'constant':
            with pytest.raises(NotImplementedError):
                shift(a, 1, output=output, order=self.order, mode=self.mode,
                      cval=self.cval)
        else:
            shift(a, 1, output=output, order=self.order, mode=self.mode,
                  cval=self.cval)

    @testing.for_int_dtypes(no_bool=True)
    def test_zoom(self, dtype):
        a = cupy.ones((32,), dtype=dtype)
        zoom = cupyx.scipy.ndimage.zoom
        output = self._prep_output(a)
        if _util._is_integer_output(output, a) and self.mode == 'constant':
            with pytest.raises(NotImplementedError):
                # zoom of 1.0 to keep same shape
                zoom(a, 1, output=output, order=self.order, mode=self.mode,
                     cval=self.cval)
        else:
            zoom(a, 1, output=output, order=self.order, mode=self.mode,
                 cval=self.cval)

    @testing.for_int_dtypes(no_bool=True)
    def test_rotate(self, dtype):
        a = cupy.ones((16, 16), dtype=dtype)
        rotate = cupyx.scipy.ndimage.rotate
        output = self._prep_output(a)
        if _util._is_integer_output(output, a) and self.mode == 'constant':
            with pytest.raises(NotImplementedError):
                # rotate by 0 to keep same shape
                rotate(a, 0, output=output, order=self.order, mode=self.mode,
                       cval=self.cval)
        else:
            rotate(a, 0, output=output, order=self.order, mode=self.mode,
                   cval=self.cval)

    @testing.for_int_dtypes(no_bool=True)
    def test_affine(self, dtype):
        a = cupy.ones((16, 16), dtype=dtype)
        affine = cupy.eye(2)
        affine_transform = cupyx.scipy.ndimage.affine_transform
        output = self._prep_output(a)
        if _util._is_integer_output(output, a) and self.mode == 'constant':
            with pytest.raises(NotImplementedError):
                affine_transform(a, affine, output=output, order=self.order,
                                 mode=self.mode, cval=self.cval)
        else:
            affine_transform(a, affine, output=output, order=self.order,
                             mode=self.mode, cval=self.cval)

    @testing.for_int_dtypes(no_bool=True)
    def test_map_coordinates(self, dtype):
        a = cupy.ones((32,), dtype=dtype)
        coords = cupy.arange(32)[cupy.newaxis, :] + 2.5
        map_coordinates = cupyx.scipy.ndimage.map_coordinates
        output = self._prep_output(a)
        if _util._is_integer_output(output, a) and self.mode == 'constant':
            with pytest.raises(NotImplementedError):
                map_coordinates(a, coords, output=output, order=self.order,
                                mode=self.mode, cval=self.cval)
        else:
            map_coordinates(a, coords, output=output, order=self.order,
                            mode=self.mode, cval=self.cval)


@testing.gpu
@testing.with_requires('opencv-python')
class TestShiftOpenCV:

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
    'mode': legacy_modes,
    'cval': [1.0],
    'prefilter': [True],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestZoom:

    _multiprocess_can_split = True

    def _zoom(self, xp, scp, a):
        _conditional_scipy_version_skip(self.mode, self.order)
        zoom = scp.ndimage.zoom
        if self.output == 'empty':
            output = zoom(a, self.zoom, None, self.order,
                          self.mode, self.cval, self.prefilter)
            return_value = zoom(a, self.zoom, output, self.order,
                                self.mode, self.cval, self.prefilter)
            assert return_value is None or return_value is output
            return output
        else:
            return zoom(a, self.zoom, self.output, self.order,
                        self.mode, self.cval, self.prefilter)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_zoom_float(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        return self._zoom(xp, scp, a)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    @testing.with_requires('scipy>=1.6.0')
    def test_zoom_complex_float(self, xp, scp, dtype):
        if self.output == numpy.float64:
            self.output = numpy.complex128
        a = testing.shaped_random((100, 100), xp, dtype)
        return self._zoom(xp, scp, a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def test_zoom_fortran_order(self, xp, scp, dtype):
        a = testing.shaped_random((100, 100), xp, dtype)
        a = xp.asfortranarray(a)
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


@testing.parameterize(*testing.product({
    'shape': [(2, 3), (4, 4)],
    'zoom': [(1, 1), (3, 5), (8, 2), (8, 8)],
    'mode': ['nearest', 'reflect', 'mirror', 'grid-wrap', 'grid-constant'],
}))
@testing.gpu
class TestZoomOrder0IntegerGrid():

    def test_zoom_grid_by_int_order0(self):
        # When grid_mode is True,  order 0 zoom should be the same as
        # replication via a Kronecker product. The only exceptions to this are
        # the non-grid modes 'constant' and 'wrap'.
        size = numpy.prod(self.shape)
        x = cupy.arange(size, dtype=float).reshape(self.shape)
        testing.assert_array_almost_equal(
            cupyx.scipy.ndimage.zoom(
                x, self.zoom, order=0, mode=self.mode, grid_mode=True
            ),
            cupy.kron(x, cupy.ones(self.zoom)),
        )


@testing.parameterize(*testing.product({
    'shape': [(5, 5, 2)],
    'zoom': [(2, 2, 0.5)],  # selected to give output.shape[-1] == 1
    'mode': legacy_modes + scipy16_modes,
    'order': [0, 1, 2, 3, 4, 5],
    'grid_mode': [False, True],
}))
@testing.gpu
class TestZoomOutputSize1():

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    @testing.with_requires('scipy>=1.6.0')
    def test_zoom_output_size1(self, xp, scp, dtype):
        x = xp.zeros(self.shape, dtype=dtype)
        x[1, 1, 1] = 1
        return scp.ndimage.zoom(x, self.zoom, order=self.order, mode=self.mode,
                                grid_mode=self.grid_mode)


@testing.parameterize(
    {'zoom': 3},
    {'zoom': 0.3},
)
@testing.gpu
@testing.with_requires('opencv-python')
class TestZoomOpenCV:

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


@testing.parameterize(*testing.product({
    # these 3 modes have analytical spline boundary conditions
    'mode': ['mirror', 'grid-wrap', 'reflect'],
    'order': [0, 1, 2, 3, 4, 5],
    'dtype': [numpy.uint8, numpy.float64],
    'output': [numpy.float64, numpy.float32],
    'axis': [0, 1, 2, -1],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestSplineFilter1d:
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_spline_filter1d(self, xp, scp):
        if self.mode == 'grid-wrap' and scipy_version < '1.6.0':
            pytest.skip('testing mode grid-wrap requires scipy >= 1.6.0')
        x = testing.shaped_random((16, 12, 11), dtype=self.dtype, xp=xp)
        return scp.ndimage.spline_filter1d(x, order=self.order, axis=self.axis,
                                           output=self.output, mode=self.mode)

    @testing.for_CF_orders(name='array_order')
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_spline_filter1d_output(self, xp, scp, array_order):
        if self.mode == 'grid-wrap' and scipy_version < '1.6.0':
            pytest.skip('testing mode grid-wrap requires scipy >= 1.6.0')
        x = testing.shaped_random((16, 12, 11), dtype=self.dtype, xp=xp,
                                  order=array_order)
        output = xp.empty(x.shape, dtype=self.output, order=array_order)
        scp.ndimage.spline_filter1d(x, order=self.order, axis=self.axis,
                                    output=output, mode=self.mode)
        return output


@testing.parameterize(*testing.product({
    # these 3 modes have analytical spline boundary conditions
    'mode': ['mirror', 'grid-wrap', 'reflect'],
    'order': [0, 1, 2, 3, 4, 5],
    'dtype': [numpy.uint8, numpy.float64],
    'output': [numpy.float64, numpy.float32],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestSplineFilter:
    @testing.numpy_cupy_allclose(atol=1e-4, rtol=1e-4, scipy_name='scp')
    def test_spline_filter(self, xp, scp):
        if self.mode == 'grid-wrap' and scipy_version < '1.6.0':
            pytest.skip('testing mode grid-wrap requires scipy >= 1.6.0')
        x = testing.shaped_random((16, 12, 11), dtype=self.dtype, xp=xp)
        if self.order < 2:
            with pytest.raises(RuntimeError):
                scp.ndimage.spline_filter(x, order=self.order,
                                          output=self.output, mode=self.mode)
            return xp.asarray([])
        return scp.ndimage.spline_filter(x, order=self.order,
                                         output=self.output, mode=self.mode)

    @testing.for_CF_orders(name='array_order')
    @testing.numpy_cupy_allclose(atol=1e-4, rtol=1e-4, scipy_name='scp')
    def test_spline_filter_with_output(self, xp, scp, array_order):
        if self.mode == 'grid-wrap' and scipy_version < '1.6.0':
            pytest.skip('testing mode grid-wrap requires scipy >= 1.6.0')
        x = testing.shaped_random((16, 12, 11), dtype=self.dtype, xp=xp,
                                  order=array_order)
        output = xp.empty(x.shape, dtype=self.output, order=array_order)
        if self.order < 2:
            with pytest.raises(RuntimeError):
                scp.ndimage.spline_filter(x, order=self.order, output=output,
                                          mode=self.mode)
            return xp.asarray([])
        scp.ndimage.spline_filter(x, order=self.order, output=output,
                                  mode=self.mode)
        return output


@testing.parameterize(*testing.product({
    'mode': ['mirror', 'wrap', 'reflect'],
    'order': [2, 3, 4, 5],
    'dtype': [numpy.complex64, numpy.complex128],
    'output': [numpy.complex64, numpy.complex128],
}))
@testing.gpu
@testing.with_requires('scipy')
class TestSplineFilterComplex:

    @testing.with_requires('scipy>=1.6')
    @testing.numpy_cupy_allclose(atol=1e-4, rtol=1e-4, scipy_name='scp')
    def test_spline_filter_complex(self, xp, scp):
        x = testing.shaped_random((16, 12, 11), dtype=self.dtype, xp=xp)
        return scp.ndimage.spline_filter(x, order=self.order,
                                         output=self.output, mode=self.mode)

    # the following test case is for SciPy versions lacking complex support
    @testing.with_requires('scipy<1.6')
    def test_spline_filter_complex2(self):
        if self.mode == 'wrap':
            pytest.skip('mode cannot be tested against SciPy < 1.6')
        cpu_func = scipy.ndimage.spline_filter
        gpu_func = cupyx.scipy.ndimage.spline_filter
        x = testing.shaped_random((16, 12, 11), dtype=self.dtype, xp=numpy)
        x_gpu = cupy.asarray(x)

        kwargs_gpu = dict(order=self.order, output=self.output, mode=self.mode)
        res_gpu = gpu_func(x_gpu, **kwargs_gpu)

        output_real = numpy.empty((1,), dtype=self.output).real.dtype
        kwargs = dict(order=self.order, output=output_real, mode=self.mode)
        res_cpu = cpu_func(x.real, **kwargs) + 1j * cpu_func(x.imag, **kwargs)
        testing.assert_allclose(res_cpu, res_gpu, atol=1e-4, rtol=1e-4)
