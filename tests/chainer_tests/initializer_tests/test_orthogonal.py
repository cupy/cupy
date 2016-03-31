import unittest

from chainer import initializers
from chainer import testing
from chainer import gradient_check        
import numpy


@testing.parameterize(
    {'shape': ()},
    {'shape': (1,)},
    {'shape': (3, 4)},
    {'shape': (3, 4, 5)}
 )
class TestOrthogonal(unittest.TestCase):

    def setUp(self):
        self.initializer = initializers.Orthogonal(scale=1.0)

    def test_shape(self):
        w = self.initializer(self.shape)
        self.assertTupleEqual(w.shape, self.shape)

    def test_orthogonality(self):
        w = self.initializer(self.shape)
        if self.shape:
            w = w.reshape(self.shape[0], -1)
            dots = numpy.tensordot(w, w, (1, 1))
            gradient_check.assert_allclose(dots, numpy.identity(self.shape[0]))
        else:
            gradient_check.assert_allclose(w, numpy.array([]))


testing.run_module(__name__, __file__)
