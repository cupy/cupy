import numpy
import test_sqrt

from chainer import cuda
import chainer.functions as F
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestSquare(test_sqrt.UnaryFunctionsTestBase):

    def make_data(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x, gy

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward_cpu(F.square, numpy.square)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward_gpu(F.square, cuda.cupy.square)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward_cpu(F.square)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward_gpu(F.square)

    def test_label(self):
        self.check_label(F.Square, 'square')


testing.run_module(__name__, __file__)
