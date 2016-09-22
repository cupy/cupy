import unittest

import numpy

from chainer import testing
from chainer import utils


@testing.parameterize(*testing.product({
    'dtype': [None, numpy.float16, numpy.float32, numpy.float64],
}))
class TestForceArray(unittest.TestCase):

    def test_scalar(self):
        x = utils.force_array(numpy.float32(1), dtype=self.dtype)
        self.assertIsInstance(x, numpy.ndarray)
        if self.dtype is None:
            self.assertEqual(x.dtype, numpy.float32)
        else:
            self.assertEqual(x.dtype, self.dtype)

    def test_0dim_array(self):
        x = utils.force_array(numpy.array(1, numpy.float32), dtype=self.dtype)
        self.assertIsInstance(x, numpy.ndarray)
        if self.dtype is None:
            self.assertEqual(x.dtype, numpy.float32)
        else:
            self.assertEqual(x.dtype, self.dtype)

    def test_array(self):
        x = utils.force_array(numpy.array([1], numpy.float32),
                              dtype=self.dtype)
        self.assertIsInstance(x, numpy.ndarray)
        if self.dtype is None:
            self.assertEqual(x.dtype, numpy.float32)
        else:
            self.assertEqual(x.dtype, self.dtype)


class TestForceType(unittest.TestCase):

    def test_force_type_scalar(self):
        x = numpy.int32(1)
        y = utils.force_type(numpy.dtype(numpy.float32), x)
        self.assertEqual(y.dtype, numpy.float32)

    def test_force_type_array(self):
        x = numpy.array([1], dtype=numpy.int32)
        y = utils.force_type(numpy.dtype(numpy.float32), x)
        self.assertEqual(y.dtype, numpy.float32)

    def test_force_type_array_no_change(self):
        x = numpy.array([1], dtype=numpy.float32)
        y = utils.force_type(numpy.dtype(numpy.float32), x)
        self.assertEqual(y.dtype, numpy.float32)


testing.run_module(__name__, __file__)
