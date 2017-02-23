import unittest
import warnings

from chainer.training import extensions


class TestPlotReport(unittest.TestCase):

    def test_available(self):
        try:
            from matplotlib import pyplot  # NOQA
            available = True
        except ImportError:
            available = False

        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(extensions.PlotReport.available(), available)

        # It shows warning only when matplotlib.pyplot is not available
        if available:
            self.assertEqual(len(w), 0)
        else:
            self.assertEqual(len(w), 1)
