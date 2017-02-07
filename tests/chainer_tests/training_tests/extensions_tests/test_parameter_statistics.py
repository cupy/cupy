import unittest

import mock
import numpy

from chainer import testing
from chainer.training import extensions


@testing.parameterize(
    {'x': [1.0, 2.0, 0.0, -1.0, 0.5], 'min': -1.0, 'max': 2.0, 'mean': 0.5,
     'std': 1.0, 'zeros': 1, 'percentiles': [-0.9948, -0.9088, -0.3652, 0.5,
                                             1.3652, 1.9088, 1.9948]},
    {'x': [0], 'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0, 'zeros': 1,
     'percentiles': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
)
class TestParameterStatisticsValid(unittest.TestCase):

    def setUp(self):
        self.trainer = mock.MagicMock()
        self.links = mock.MagicMock()
        self.extension = extensions.ParameterStatistics(self.links)

    def test_call(self):
        self.extension(self.trainer)

    def test_statistics(self):
        statistics = self.extension.statistics(numpy.array(self.x))
        for f, s in statistics.items():
            self.assertEqual(s, getattr(self, f))

    def test_percentiles(self):
        percentiles = self.extension.percentiles(numpy.array(self.x))
        for p, e in zip(percentiles, self.percentiles):
            self.assertAlmostEqual(p, e)

    def test_sparsity(self):
        zeros = self.extension.sparsity(numpy.array(self.x))
        self.assertEqual(zeros, self.zeros)


class TestParameterStatisticsEmpty(unittest.TestCase):

    def setUp(self):
        self.x = []
        self.links = mock.MagicMock()
        self.extension = extensions.ParameterStatistics(self.links)

    def test_statistics(self):
        statistics = self.extension.statistics(numpy.array(self.x))
        for s in statistics.values():
            self.assertTrue(numpy.isnan(s))

    def test_percentiles(self):
        percentiles = self.extension.percentiles(numpy.array(self.x))
        for p in percentiles:
            self.assertTrue(numpy.isnan(p))

    def test_sparsity(self):
        zeros = self.extension.sparsity(numpy.array(self.x))
        self.assertEqual(zeros, 0)


class TestParameterStatisticsNone(unittest.TestCase):

    def setUp(self):
        self.x = None
        self.links = mock.MagicMock()
        self.extension = extensions.ParameterStatistics(self.links)

    def test_none(self):
        with self.assertRaises(ValueError):
            self.extension.sparsity(numpy.array(self.x))


@testing.parameterize(
    {'stats': {'key_1': 0, 'key_2': 0}}
)
class TestParameterStatisticsPostProcess(unittest.TestCase):

    def setUp(self):
        self.prefix = 'prefix'
        self.links = mock.MagicMock()
        self.extension = extensions.ParameterStatistics(self.links,
                                                        prefix=self.prefix)

    def test_prefix(self):
        for key in self.extension.post_process(self.stats).keys():
            self.assertTrue(key.startswith(self.prefix))


testing.run_module(__name__, __file__)
