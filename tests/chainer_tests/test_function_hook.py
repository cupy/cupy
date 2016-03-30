import unittest

from chainer import function
from chainer import testing


class TestFunctionHook(unittest.TestCase):

    def setUp(self):
        self.h = function.FunctionHook()

    def test_name(self):
        self.assertEqual(self.h.name, 'FunctionHook')

    def test_forward_preprocess(self):
        self.assertTrue(hasattr(self.h, 'forward_preprocess'))

    def test_forward_postprocess(self):
        self.assertTrue(hasattr(self.h, 'forward_postprocess'))

    def test_backward_preprocess(self):
        self.assertTrue(hasattr(self.h, 'backward_preprocess'))

    def test_backward_postprocess(self):
        self.assertTrue(hasattr(self.h, 'backward_postprocess'))


testing.run_module(__name__, __file__)
