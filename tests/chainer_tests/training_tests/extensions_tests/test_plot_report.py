import unittest

from chainer.training import extensions


class TestPlotReport(unittest.TestCase):

    def test_available(self):
        try:
            from matplotlib import pyplot  # NOQA
            available = True
        except ImportError:
            available = False

        self.assertEqual(extensions.PlotReport.available(), available)
