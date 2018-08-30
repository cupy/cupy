import unittest
import warnings

import numpy

from cupy import testing


@testing.parameterize(
    *testing.product({
        'array': [numpy.arange(6).reshape([2, 3])],
        'pad_width': [1, [1, 2], [[1, 2], [3, 4]]],
        'mode': ['constant', 'edge', 'reflect'],
    })
)
@testing.gpu
class TestPadDefault(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_pad_default(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)

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
)
@testing.gpu
# Old numpy does not work with multi-dimensional constant_values
@testing.with_requires('numpy>=1.11.1')
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
            elif self.mode == 'reflect':
                return xp.pad(array, self.pad_width, mode=self.mode,
                              reflect_type=self.reflect_type)

        if xp is numpy:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', numpy.ComplexWarning)
                return f()
        else:
            return f()


@testing.gpu
class TestPadNumpybug(unittest.TestCase):

    @testing.with_requires('numpy>=1.11.2')
    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_pad_highdim_default(self, xp, dtype):
        array = xp.arange(6, dtype=dtype).reshape([2, 3])
        pad_width = [[1, 2], [3, 4]]
        constant_values = [[1, 2], [3, 4]]
        a = xp.pad(array, pad_width, mode='constant',
                   constant_values=constant_values)
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
        if self.mode == 'constant':
            a = xp.pad(self.array, self.pad_width, mode=self.mode,
                       constant_values=self.constant_values)
        elif self.mode in ['edge', 'reflect']:
            a = xp.pad(self.array, self.pad_width, mode=self.mode)
        return a


@testing.parameterize(
    {'array': [0, 1, 2, 3], 'pad_width': [-1, 1], 'mode': 'constant',
     'constant_values': 3},
    {'array': [0, 1, 2, 3], 'pad_width': [], 'mode': 'constant',
     'constant_values': 3},
    {'array': [0, 1, 2, 3], 'pad_width': [[3, 4], [5, 6]], 'mode': 'constant',
     'constant_values': 3},
    {'array': [0, 1, 2, 3], 'pad_width': [1], 'mode': 'constant',
     'notallowedkeyword': 3},
    # mode='edge'
    {'array': [], 'pad_width': 1, 'mode': 'edge'},
    {'array': [0, 1, 2, 3], 'pad_width': [-1, 1], 'mode': 'edge'},
    {'array': [0, 1, 2, 3], 'pad_width': [], 'mode': 'edge'},
    {'array': [0, 1, 2, 3], 'pad_width': [[3, 4], [5, 6]], 'mode': 'edge'},
    {'array': [0, 1, 2, 3], 'pad_width': [1], 'mode': 'edge',
     'notallowedkeyword': 3},
    # mode='reflect'
    {'array': [], 'pad_width': 1, 'mode': 'reflect'},
    {'array': [0, 1, 2, 3], 'pad_width': [-1, 1], 'mode': 'reflect'},
    {'array': [0, 1, 2, 3], 'pad_width': [], 'mode': 'reflect'},
    {'array': [0, 1, 2, 3], 'pad_width': [[3, 4], [5, 6]], 'mode': 'reflect'},
    {'array': [0, 1, 2, 3], 'pad_width': [1], 'mode': 'reflect',
     'notallowedkeyword': 3},
)
@testing.gpu
@testing.with_requires('numpy>=1.11.1')  # Old numpy fails differently
class TestPadFailure(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_pad_failure(self, xp):
        a = xp.pad(self.array, self.pad_width, mode=self.mode,
                   constant_values=self.constant_values)
        return a
