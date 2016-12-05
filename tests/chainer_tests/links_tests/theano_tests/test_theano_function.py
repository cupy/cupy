import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import links
from chainer.links.theano import theano_function
from chainer import testing
from chainer.testing import attr


if theano_function._available:
    import theano.tensor as T


@unittest.skipUnless(theano_function._available, 'theano is not available')
class TheanoFunctionTestBase(object):

    forward_test_options = {}
    backward_test_options = {'atol': 1e-4}

    def setUp(self):
        self.input_data = [
            numpy.random.uniform(
                -1, 1, d['shape']).astype(getattr(numpy, d['type']))
            for d in self.inputs]
        self.grad_data = [
            numpy.random.uniform(
                -1, 1, d['shape']).astype(getattr(numpy, d['type']))
            for d in self.outputs]

    def make_func(self):
        raise NotImplementedError

    def expect_forward(self):
        raise NotImplementedError

    def check_forward(self, input_data):
        func = self.make_func()
        inputs = [chainer.Variable(data) for data in input_data]
        outputs = func(*inputs)
        if isinstance(outputs, chainer.Variable):
            outputs = (outputs,)
        expect = self.expect_forward()

        self.assertEqual(len(outputs), len(expect))
        for o, e in zip(outputs, expect):
            gradient_check.assert_allclose(
                o.data, e, **self.forward_test_options)

    def test_forward_cpu(self):
        self.check_forward(self.input_data)

    @attr.gpu
    def test_forward_gpu(self):
        inputs = [cuda.to_gpu(x) for x in self.input_data]
        self.check_forward(inputs)

    def check_backward(self, input_data, grad_data):
        func = self.make_func()
        gradient_check.check_backward(
            func, input_data, grad_data, **self.backward_test_options)

    def test_backward_cpu(self):
        self.check_backward(self.input_data, self.grad_data)

    @attr.gpu
    def test_backward_gpu(self):
        inputs = [cuda.to_gpu(x) for x in self.input_data]
        grads = [cuda.to_gpu(x) for x in self.grad_data]
        self.check_backward(inputs, grads)


@testing.parameterize(
    {'inputs': [{'shape': (3, 2), 'type': 'float32'},
                {'shape': (3, 2), 'type': 'float32'}],
     'outputs': [{'shape': (3, 2), 'type': 'float32'}]},
    {'inputs': [{'shape': (3, 2), 'type': 'float32'},
                {'shape': (2,), 'type': 'float32'}],
     'outputs': [{'shape': (3, 2), 'type': 'float32'}]},
    {'inputs': [{'shape': (3, 2), 'type': 'float32'},
                {'shape': (), 'type': 'float32'}],
     'outputs': [{'shape': (3, 2), 'type': 'float32'}]},
    {'inputs': [{'shape': (3, 2), 'type': 'float32'},
                {'shape': (3, 2), 'type': 'float64'}],
     'outputs': [{'shape': (3, 2), 'type': 'float64'}]},
    {'inputs': [{'shape': (3, 2), 'type': 'float16'},
                {'shape': (3, 2), 'type': 'float32'}],
     'outputs': [{'shape': (3, 2), 'type': 'float32'}],
     'forward_test_options': {'atol': 1e-3, 'rtol': 1e-3},
     'backward_test_options': {'eps': 1, 'atol': 1e-3, 'rtol': 1e-3}},
)
class TestTheanoFunction(TheanoFunctionTestBase, unittest.TestCase):

    def make_func(self):
        x = T.TensorType(self.inputs[0]['type'],
                         (False,) * len(self.inputs[0]['shape']))('x')
        y = T.TensorType(self.inputs[1]['type'],
                         (False,) * len(self.inputs[1]['shape']))('y')
        z = x + y
        return links.TheanoFunction([x, y], [z])

    def expect_forward(self):
        x, y = self.input_data
        return x + y,


@testing.parameterize(
    {'inputs': [{'shape': (3, 2), 'type': 'float32'},
                {'shape': (3, 2), 'type': 'float32'}],
     'outputs': [{'shape': (3, 2), 'type': 'float32'},
                 {'shape': (3, 2), 'type': 'float32'}]},
    {'inputs': [{'shape': (3, 2), 'type': 'float32'},
                {'shape': (2,), 'type': 'float32'}],
     'outputs': [{'shape': (3, 2), 'type': 'float32'},
                 {'shape': (3, 2), 'type': 'float32'}]},
    {'inputs': [{'shape': (3, 2), 'type': 'float32'},
                {'shape': (), 'type': 'float32'}],
     'outputs': [{'shape': (3, 2), 'type': 'float32'},
                 {'shape': (3, 2), 'type': 'float32'}]},
)
class TestTheanoFunctionTwoOutputs(TheanoFunctionTestBase, unittest.TestCase):

    def make_func(self):
        x = T.TensorType(self.inputs[0]['type'],
                         (False,) * len(self.inputs[0]['shape']))('x')
        y = T.TensorType(self.inputs[1]['type'],
                         (False,) * len(self.inputs[1]['shape']))('y')
        z = x + y
        w = x - y
        return links.TheanoFunction([x, y], [z, w])

    def expect_forward(self):
        x, y = self.input_data
        return x + y, x - y


@testing.parameterize(
    {'inputs': [{'shape': (3, 2), 'type': 'float32'},
                {'shape': (2,), 'type': 'int32'}],
     'outputs': [{'shape': (2, 2), 'type': 'float32'}]},
    {'inputs': [{'shape': (3, 2), 'type': 'float32'},
                {'shape': (), 'type': 'int32'}],
     'outputs': [{'shape': (2,), 'type': 'float32'}]},
)
class TestTheanoFunctionNonDifferential(
        TheanoFunctionTestBase, unittest.TestCase):

    def make_func(self):
        x = T.TensorType(self.inputs[0]['type'],
                         (False,) * len(self.inputs[0]['shape']))('x')
        i = T.TensorType(self.inputs[1]['type'],
                         (False,) * len(self.inputs[1]['shape']))('y')
        z = x[i]
        return links.TheanoFunction([x, i], z)

    def expect_forward(self):
        x, i = self.input_data
        return x[i],


testing.run_module(__name__, __file__)
