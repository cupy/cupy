from unittest import TestCase

import numpy

from chainer.gradient_check import assert_allclose
from chainer.utils import WalkerAlias


class TestWalkerAlias(TestCase):
    def setUp(self):
        self.ps = [5, 3, 4, 1, 2]
        self.sampler = WalkerAlias(self.ps)

    def check_sample(self):
        counts = numpy.zeros(len(self.ps), numpy.float32)
        for _ in range(1000):
            vs = self.sampler.sample((4, 3))
            numpy.add.at(counts, to_cpu(vs), 1)
        counts /= (1000 * 12)
        counts *= sum(self.ps)
        assert_allclose(self.ps, counts, atol=0.1, rtol=0.1)

    def test_sample_cpu(self):
        self.check_sample()

    def test_sample_gpu(self):
        self.sampler.to_gpu()
        self.check_sample()
