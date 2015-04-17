from unittest import TestCase
import numpy

from chainer import Variable
from chainer.functions import identity

class TestFunction(TestCase):
    def test_forward(self):
        xs = (Variable(numpy.array([0]), rank=1),
              Variable(numpy.array([0]), rank=3),
              Variable(numpy.array([0]), rank=2))
        ys = identity(*xs)

        self.assertEqual(len(ys), len(xs))
        for y in ys:
            # rank is (maximum rank in xs) + 2, since Function call
            # automatically inserts Split function.
            self.assertEqual(y.rank, 5)

    def test_backward(self):
        x = Variable(numpy.array([1]))
        y1 = identity(x)
        y2 = identity(x)
        z = y1 + y2

        z.grad = numpy.array([1])
        z.backward()

        self.assertEqual(y1.grad[0], 1)
        self.assertEqual(y2.grad[0], 1)
        self.assertEqual(x.grad[0], 2)
