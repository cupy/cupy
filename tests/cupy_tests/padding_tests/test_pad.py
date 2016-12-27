import unittest

import numpy

from cupy import testing


@testing.parameterize(
    {'array': numpy.arange(6).reshape([2, 3]), 'pad_width': 1,
     'mode': 'constant'},
    {'array': numpy.arange(6).reshape([2, 3]),
     'pad_width': numpy.array([1, 2]), 'mode': 'constant'},
    {'array': numpy.arange(6).reshape([2, 3]),
     'pad_width': numpy.array([[1, 2], [3, 4]]), 'mode': 'constant'},
    {'array': numpy.ones([4, 5, 6, 7]),
     'pad_width': numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
     'mode': 'constant'},
)
@testing.gpu
class TestPadDefault(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_pad_default(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        a = xp.pad(array, self.pad_width, mode=self.mode)
        return a


@testing.parameterize(
    {'array': numpy.arange(6).reshape([2, 3]), 'pad_width': 1,
     'mode': 'constant', 'constant_values': 3},
    {'array': numpy.arange(6).reshape([2, 3]),
     'pad_width': numpy.array([1, 2]), 'mode': 'constant',
     'constant_values': numpy.array([3, 4])},
    {'array': numpy.arange(6).reshape([2, 3]),
     'pad_width': numpy.array([[1, 2], [3, 4]]), 'mode': 'constant',
     'constant_values': numpy.array([[3, 4], [5, 6]])},
    {'array': numpy.ones([4, 5, 6, 7]),
     'pad_width': numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
     'mode': 'constant',
     'constant_values': numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]])},
)
@testing.gpu
class TestPad(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_pad(self, xp, dtype):
        array = xp.array(self.array, dtype=dtype)
        a = xp.pad(array, self.pad_width, mode=self.mode,
                   constant_values=self.constant_values)
        return a


@testing.parameterize(
    {'array': [], 'pad_width': 1, 'mode': 'constant', 'constant_values': 3},
    {'array': 1, 'pad_width': 1, 'mode': 'constant', 'constant_values': 3},
    {'array': [0, 1, 2, 3], 'pad_width': 1, 'mode': 'constant',
     'constant_values': 3},
    {'array': [0, 1, 2, 3], 'pad_width': [1, 2], 'mode': 'constant',
     'constant_values': 3},
    {'array': [[0, 1, 2], [3, 4, 5]], 'pad_width': [[1, 2], [3, 4]],
     'mode': 'constant', 'constant_values': [[1, 2], [3, 4]]},
)
@testing.gpu
class TestPadSpecial(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_pad_special(self, xp):
        a = xp.pad(self.array, self.pad_width, mode=self.mode,
                   constant_values=self.constant_values)
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
)
@testing.gpu
class TestPadFailure(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_raises()
    def test_pad_failure(self, xp):
        a = xp.pad(self.array, self.pad_width, mode=self.mode,
                   constant_values=self.constant_values)
        return a
