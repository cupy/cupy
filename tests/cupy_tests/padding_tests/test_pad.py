import unittest
import warnings

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(
    *testing.product({
        'array': [numpy.arange(6).reshape([2, 3])],
        'pad_width': [1, [1, 2], [[1, 2], [3, 4]]],
        # mode 'mean' is non-exact, so it is tested in a separate class
        'mode': ['constant', 'edge', 'linear_ramp', 'maximum',
                 'minimum', 'reflect', 'symmetric', 'wrap'],
    })
)
@testing.gpu
class TestPadDefault(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_pad_default(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)

        if (xp.dtype(dtype).kind in ['i', 'u'] and
                self.mode == 'linear_ramp'):
            # TODO: can remove this skip once cupy/cupy/#2330 is merged
            return array

        # Older version of NumPy(<1.12) can emit ComplexWarning
        def f():
            return xp.pad(array, self.pad_width, mode=self.mode)

        if xp is numpy:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', numpy.ComplexWarning)
                return f()
        else:
            return f()


@testing.parameterize(
    *testing.product({
        'array': [numpy.arange(6).reshape([2, 3])],
        'pad_width': [1, [1, 2], [[1, 2], [3, 4]]],
    })
)
@testing.gpu
class TestPadDefaultMean(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_almost_equal(decimal=5)
    def test_pad_default(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)

        if xp.dtype(dtype).kind in ['i', 'u']:
            # TODO: can remove this skip once cupy/cupy/#2330 is merged
            return array

        # Older version of NumPy(<1.12) can emit ComplexWarning
        def f():
            return xp.pad(array, self.pad_width, mode='mean')

        if xp is numpy:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', numpy.ComplexWarning)
                return f()
        else:
            return f()


@testing.parameterize(
    # mode='constant'
    {'array': numpy.arange(6).reshape([2, 3]), 'pad_width': 1,
     'mode': 'constant', 'constant_values': 3},
    {'array': numpy.arange(6).reshape([2, 3]),
     'pad_width': [1, 2], 'mode': 'constant',
     'constant_values': [3, 4]},
    {'array': numpy.arange(6).reshape([2, 3]),
     'pad_width': [[1, 2], [3, 4]], 'mode': 'constant',
     'constant_values': [[3, 4], [5, 6]]},
    # mode='reflect'
    {'array': numpy.arange(6).reshape([2, 3]), 'pad_width': 1,
     'mode': 'reflect', 'reflect_type': 'odd'},
    {'array': numpy.arange(6).reshape([2, 3]),
     'pad_width': [1, 2], 'mode': 'reflect', 'reflect_type': 'odd'},
    {'array': numpy.arange(6).reshape([2, 3]),
     'pad_width': [[1, 2], [3, 4]], 'mode': 'reflect',
     'reflect_type': 'odd'},
    # mode='symmetric'
    {'array': numpy.arange(6).reshape([2, 3]), 'pad_width': 1,
     'mode': 'symmetric', 'reflect_type': 'odd'},
    {'array': numpy.arange(6).reshape([2, 3]),
     'pad_width': [1, 2], 'mode': 'symmetric', 'reflect_type': 'odd'},
    {'array': numpy.arange(6).reshape([2, 3]),
     'pad_width': [[1, 2], [3, 4]], 'mode': 'symmetric',
     'reflect_type': 'odd'},
    # mode='minimum'
    {'array': numpy.arange(60).reshape([5, 12]), 'pad_width': 1,
     'mode': 'minimum', 'stat_length': 2},
    {'array': numpy.arange(60).reshape([5, 12]),
     'pad_width': [1, 2], 'mode': 'minimum', 'stat_length': (2, 4)},
    {'array': numpy.arange(60).reshape([5, 12]),
     'pad_width': [[1, 2], [3, 4]], 'mode': 'minimum',
     'stat_length': ((2, 4), (3, 5))},
    {'array': numpy.arange(60).reshape([5, 12]),
     'pad_width': [[1, 2], [3, 4]], 'mode': 'minimum',
     'stat_length': None},
    # mode='maximum'
    {'array': numpy.arange(60).reshape([5, 12]), 'pad_width': 1,
     'mode': 'maximum', 'stat_length': 2},
    {'array': numpy.arange(60).reshape([5, 12]),
     'pad_width': [1, 2], 'mode': 'maximum', 'stat_length': (2, 4)},
    {'array': numpy.arange(60).reshape([5, 12]),
     'pad_width': [[1, 2], [3, 4]], 'mode': 'maximum',
     'stat_length': ((2, 4), (3, 5))},
    {'array': numpy.arange(60).reshape([5, 12]),
     'pad_width': [[1, 2], [3, 4]], 'mode': 'maximum',
     'stat_length': None},
)
@testing.gpu
# Old numpy does not work with multi-dimensional constant_values
class TestPad(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_pad(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)

        # Older version of NumPy(<1.12) can emit ComplexWarning
        def f():
            if self.mode == 'constant':
                return xp.pad(array, self.pad_width, mode=self.mode,
                              constant_values=self.constant_values)
            elif self.mode in ['minimum', 'maximum']:
                return xp.pad(array, self.pad_width, mode=self.mode,
                              stat_length=self.stat_length)
            elif self.mode in ['reflect', 'symmetric']:
                return xp.pad(array, self.pad_width, mode=self.mode,
                              reflect_type=self.reflect_type)

        if xp is numpy:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', numpy.ComplexWarning)
                return f()
        else:
            return f()


@testing.parameterize(
    # mode='mean'
    {'array': numpy.arange(60).reshape([5, 12]), 'pad_width': 1,
     'mode': 'mean', 'stat_length': 2},
    {'array': numpy.arange(60).reshape([5, 12]),
     'pad_width': [1, 2], 'mode': 'mean', 'stat_length': (2, 4)},
    {'array': numpy.arange(60).reshape([5, 12]),
     'pad_width': [[1, 2], [3, 4]], 'mode': 'mean',
     'stat_length': ((2, 4), (3, 5))},
    {'array': numpy.arange(60).reshape([5, 12]),
     'pad_width': [[1, 2], [3, 4]], 'mode': 'mean',
     'stat_length': None},
)
@testing.gpu
# Old numpy does not work with multi-dimensional constant_values
class TestPadMean(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_almost_equal(decimal=5)
    def test_pad(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)

        if xp.dtype(dtype).kind in ['i', 'u']:
            # TODO: can remove this skip once cupy/cupy/#2330 is merged
            return array

        # Older version of NumPy(<1.12) can emit ComplexWarning
        def f():
            return xp.pad(array, self.pad_width, mode=self.mode,
                          stat_length=self.stat_length)

        if xp is numpy:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', numpy.ComplexWarning)
                return f()
        else:
            return f()


@testing.gpu
class TestPadNumpybug(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_pad_highdim_default(self, xp, dtype):
        array = xp.arange(6, dtype=dtype).reshape([2, 3])
        pad_width = [[1, 2], [3, 4]]
        constant_values = [[1, 2], [3, 4]]
        a = xp.pad(array, pad_width, mode='constant',
                   constant_values=constant_values)
        return a


@testing.gpu
class TestPadEmpty(unittest.TestCase):

    @testing.with_requires('numpy>=1.17')
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_pad_empty(self, xp, dtype):
        array = xp.arange(6, dtype=dtype).reshape([2, 3])
        pad_width = 2
        a = xp.pad(array, pad_width=pad_width, mode='empty')
        # omit uninitialized "empty" boundary from the comparison
        return a[pad_width:-pad_width, pad_width:-pad_width]


@testing.gpu
class TestPadCustomFunction(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_pad_via_func(self, xp, dtype):
        def _padwithtens(vector, pad_width, iaxis, kwargs):
            vector[:pad_width[0]] = 10
            vector[-pad_width[1]:] = 10
        a = xp.arange(6, dtype=dtype).reshape(2, 3)
        a = xp.pad(a, 2, _padwithtens)
        return a


@testing.parameterize(
    # mode='constant'
    {'array': [], 'pad_width': 1, 'mode': 'constant', 'constant_values': 3},
    {'array': 1, 'pad_width': 1, 'mode': 'constant', 'constant_values': 3},
    {'array': [0, 1, 2, 3], 'pad_width': 1, 'mode': 'constant',
     'constant_values': 3},
    {'array': [0, 1, 2, 3], 'pad_width': [1, 2], 'mode': 'constant',
     'constant_values': 3},
    # mode='edge'
    {'array': 1, 'pad_width': 1, 'mode': 'edge'},
    {'array': [0, 1, 2, 3], 'pad_width': 1, 'mode': 'edge'},
    {'array': [0, 1, 2, 3], 'pad_width': [1, 2], 'mode': 'edge'},
    # mode='reflect'
    {'array': 1, 'pad_width': 1, 'mode': 'reflect'},
    {'array': [0, 1, 2, 3], 'pad_width': 1, 'mode': 'reflect'},
    {'array': [0, 1, 2, 3], 'pad_width': [1, 2], 'mode': 'reflect'},
)
@testing.gpu
class TestPadSpecial(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_pad_special(self, xp):
        array = xp.array(self.array)

        if self.mode == 'constant':
            a = xp.pad(array, self.pad_width, mode=self.mode,
                       constant_values=self.constant_values)
        elif self.mode in ['edge', 'reflect']:
            a = xp.pad(array, self.pad_width, mode=self.mode)
        return a


@testing.parameterize(
    {'array': [0, 1, 2, 3], 'pad_width': [-1, 1], 'mode': 'constant',
     'kwargs': {'constant_values': 3}},
    {'array': [0, 1, 2, 3], 'pad_width': [[3, 4], [5, 6]], 'mode': 'constant',
     'kwargs': {'constant_values': 3}},
    {'array': [0, 1, 2, 3], 'pad_width': [1], 'mode': 'constant',
     'kwargs': {'notallowedkeyword': 3}},
    # edge
    {'array': [], 'pad_width': 1, 'mode': 'edge',
     'kwargs': {}},
    {'array': [0, 1, 2, 3], 'pad_width': [-1, 1], 'mode': 'edge',
     'kwargs': {}},
    {'array': [0, 1, 2, 3], 'pad_width': [[3, 4], [5, 6]], 'mode': 'edge',
     'kwargs': {}},
    {'array': [0, 1, 2, 3], 'pad_width': [1], 'mode': 'edge',
     'kwargs': {'notallowedkeyword': 3}},
    # mode='reflect'
    {'array': [], 'pad_width': 1, 'mode': 'reflect',
     'kwargs': {}},
    {'array': [0, 1, 2, 3], 'pad_width': [-1, 1], 'mode': 'reflect',
     'kwargs': {}},
    {'array': [0, 1, 2, 3], 'pad_width': [[3, 4], [5, 6]], 'mode': 'reflect',
     'kwargs': {}},
    {'array': [0, 1, 2, 3], 'pad_width': [1], 'mode': 'reflect',
     'kwargs': {'notallowedkeyword': 3}},
)
@testing.gpu
@testing.with_requires('numpy>=1.17')
class TestPadValueError(unittest.TestCase):

    def test_pad_failure(self):
        for xp in (numpy, cupy):
            array = xp.array(self.array)
            with pytest.raises(ValueError):
                xp.pad(array, self.pad_width, self.mode, **self.kwargs)


@testing.parameterize(
    {'array': [0, 1, 2, 3], 'pad_width': [], 'mode': 'constant',
     'kwargs': {'constant_values': 3}},
    # edge
    {'array': [0, 1, 2, 3], 'pad_width': [], 'mode': 'edge',
     'kwargs': {}},
    # mode='reflect'
    {'array': [0, 1, 2, 3], 'pad_width': [], 'mode': 'reflect',
     'kwargs': {}},
)
@testing.gpu
class TestPadTypeError(unittest.TestCase):

    def test_pad_failure(self):
        for xp in (numpy, cupy):
            array = xp.array(self.array)
            with pytest.raises(TypeError):
                xp.pad(array, self.pad_width, self.mode, **self.kwargs)
