from unittest import TestCase
from linear_model import LinearModel
from chainer.optimizers import AdaGrad
from chainer.testing import attr

class TestAdaGrad(TestCase):
    def setUp(self):
        self.optimizer = AdaGrad(0.1)
        self.model = LinearModel(self.optimizer)

    def test_linear_model_cpu(self):
        self.assertGreater(self.model.accuracy(False), 0.7)

    @attr.gpu
    def test_linear_model_gpu(self):
        self.assertGreater(self.model.accuracy(True), 0.7)
