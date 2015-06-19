from unittest import TestCase
from linear_model import LinearModel
from chainer.optimizers import MomentumSGD
from chainer.testing import attr

class TestMomentumSGD(TestCase):
    def setUp(self):
        self.optimizer = MomentumSGD(0.1)
        self.model = LinearModel(self.optimizer)

    def test_linear_model_cpu(self):
        self.assertGreater(self.model.accuracy(False), 0.8)

    @attr.gpu
    def test_linear_model_gpu(self):
        self.assertGreater(self.model.accuracy(True), 0.8)
