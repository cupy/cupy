import unittest

import numpy

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer import utils


class TestWalkerAlias(unittest.TestCase):

    def setUp(self):
        self.ps = numpy.array([5, 3, 4, 1, 2], dtype=numpy.int32)
        self.sampler = utils.WalkerAlias(self.ps)

    def check_sample(self):
        counts = numpy.zeros(len(self.ps), numpy.float32)
        for _ in range(1000):
            vs = self.sampler.sample((4, 3))
            numpy.add.at(counts, cuda.to_cpu(vs), 1)
        counts /= (1000 * 12)
        counts *= sum(self.ps)
        testing.assert_allclose(self.ps, counts, atol=0.1, rtol=0.1)

    def test_sample_cpu(self):
        self.check_sample()

    @attr.gpu
    def test_sample_gpu(self):
        self.sampler.to_gpu()
        self.assertTrue(self.sampler.use_gpu)
        self.check_sample()

    @attr.gpu
    def test_to_cpu(self):
        self.sampler.to_gpu()
        self.sampler.to_cpu()
        self.assertFalse(self.sampler.use_gpu)
        self.check_sample()


testing.run_module(__name__, __file__)
