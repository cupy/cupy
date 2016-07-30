import unittest

from chainer import testing


class TestImportError(unittest.TestCase):

    def test_import_error(self):
        try:
            import cupy  # noqa
        except Exception as e:
            self.assertIsInstance(e, RuntimeError)


testing.run_module(__name__, __file__)
