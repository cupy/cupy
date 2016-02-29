import unittest
from chainer import function
from chainer import testing


class MockFunctionHook(function.FunctionHook):

    def __init__(self, *args, **kwargs):
        self.call_count = 0
        self.preprocess_count = 0
        self.postprocess_count = 0
        super(MockFunctionHook, self).__init__(*args, **kwargs)

    def __call__(self, function, in_data, out_grad=None):
        self.call_count += 1

    def preprocess(self, function, in_data, out_grad=None):
        self.preprocess_count += 1
        super(MockFunctionHook, self).preprocess(function, in_data, out_grad)

    def postprocess(self, function, in_data, out_grad=None):
        self.postprocess_count += 1
        super(MockFunctionHook, self).postprocess(function, in_data, out_grad)


class TestFunctionHook(unittest.TestCase):

    def setUp(self):
        self.h = MockFunctionHook()

    def test_name(self):
        self.assertEqual(self.h.name, 'FunctionHook')

    def test_forward_preprocess(self):
        self.h.forward_preprocess(None, None)
        self.assertEqual(self.h.preprocess_count, 1)
        self.assertEqual(self.h.call_count, 1)

    def test_forward_postprocess(self):
        self.h.forward_postprocess(None, None)
        self.assertEqual(self.h.postprocess_count, 1)
        self.assertEqual(self.h.call_count, 1)

    def test_backward_preprocess(self):
        self.h.backward_preprocess(None, None, None)
        self.assertEqual(self.h.preprocess_count, 1)
        self.assertEqual(self.h.call_count, 1)

    def test_backward_postprocess(self):
        self.h.backward_postprocess(None, None, None)
        self.assertEqual(self.h.postprocess_count, 1)
        self.assertEqual(self.h.call_count, 1)


testing.run_module(__name__, __file__)
