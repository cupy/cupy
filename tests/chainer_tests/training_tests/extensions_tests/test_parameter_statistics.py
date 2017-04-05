import unittest

import mock
import numpy

from chainer import cuda
from chainer import testing
from chainer.testing import attr
from chainer.training import extensions


@testing.parameterize(
    {'in_x': [1.0, 2.0, 0.0, -1.0, 0.5],
     'target_min': -1.0,
     'target_max': 2.0,
     'target_mean': 0.5,
     'target_std': 1.0,
     'target_zeros': 1,
     'target_percentiles': [-0.9948, -0.9088, -0.3652, 0.5,
                            1.3652, 1.9088, 1.9948],
     'dtype': numpy.float32},
    {'in_x': [0],
     'target_min': 0.0,
     'target_max': 0.0,
     'target_mean': 0.0,
     'target_std': 0.0,
     'target_zeros': 1,
     'target_percentiles': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     'dtype': numpy.float32},
    {'in_x': [],
     'target_min': numpy.nan,
     'target_max': numpy.nan,
     'target_mean': numpy.nan,
     'target_std': numpy.nan,
     'target_zeros': 0,
     'target_percentiles': [numpy.nan, numpy.nan, numpy.nan, numpy.nan,
                            numpy.nan, numpy.nan],
     'dtype': numpy.float32},
)
class TestParameterStatistics(unittest.TestCase):

    def setUp(self):
        self.trainer = mock.MagicMock()
        self.links = mock.MagicMock()
        self.extension = extensions.ParameterStatistics(self.links)
        self.x = numpy.array(self.in_x, dtype=self.dtype)

    def test_statistics_cpu(self):
        report = self.extension.statistics_report(self.x)

        for f, s in report.items():
            if self.x.size == 0:
                self.assertTrue(numpy.isnan(s))
            else:
                self.assertIsInstance(s, self.dtype)
                self.assertEqual(s, getattr(self, 'target_{}'.format(f)))

    @attr.gpu
    def test_statistics_gpu(self):
        report = self.extension.statistics_report(cuda.to_gpu(self.x))

        for f, s in report.items():
            if self.x.size == 0:
                self.assertTrue(cuda.get_array_module(s).isnan(s))
            else:
                self.assertIsInstance(s, cuda.ndarray)
                self.assertEqual(s, getattr(self, 'target_{}'.format(f)))


    def test_percentiles_cpu(self):
        report = self.extension.percentiles_report(self.x)

        for i, tp in enumerate(self.target_percentiles):
            p = report['percentile/{}'.format(i)]
            if self.x.size == 0:
                self.assertTrue(numpy.isnan(p))
            else:
                self.assertIsInstance(p, self.dtype)
                self.assertAlmostEqual(p, tp)

    @attr.gpu
    def test_percentiles_gpu(self):
        report = self.extension.percentiles_report(cuda.to_gpu(self.x))

        for i, tp in enumerate(self.target_percentiles):
            p = report['percentile/{}'.format(i)]
            if self.x.size == 0:
                self.assertTrue(cuda.get_array_module(p).isnan(p))
            else:
                self.assertIsInstance(p, cuda.ndarray)
                self.assertAlmostEqual(cuda.to_cpu(p), tp)

    def test_zeros_cpu(self):
        report = self.extension.zeros_report(self.x)

        z = report['zeros']
        self.assertIsInstance(z, int)
        self.assertEqual(z, self.target_zeros)

    @attr.gpu
    def test_zeros_gpu(self):
        report = self.extension.zeros_report(cuda.to_gpu(self.x))

        z = report['zeros']
        self.assertIsInstance(z, int)
        self.assertEqual(z, self.target_zeros)


@testing.parameterize(
    {'stats': {'key_1': 0, 'key_2': 0}, 'prefix': 'prefix'}
)
class TestParameterStatisticsPostProcess(unittest.TestCase):

    def setUp(self):
        self.links = mock.MagicMock()
        self.extension = extensions.ParameterStatistics(self.links,
                                                        prefix=self.prefix)

    def test_prefix(self):
        for key in self.extension.post_process(self.stats).keys():
            self.assertTrue(key.startswith(self.prefix))


testing.run_module(__name__, __file__)
