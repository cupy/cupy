import unittest

import numpy
import theano.tensor as T

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TheanoFunctionTestBase(object):

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
        gpu = isinstance(input_data[0], cuda.cupy.ndarray)
        func = self.make_func(gpu)
        inputs = [chainer.Variable(data) for data in input_data]
        outputs = func(*inputs)
        if isinstance(outputs, chainer.Variable):
            outputs = (outputs,)
        expect = self.expect_forward()

        self.assertEqual(len(outputs), len(expect))
        for o, e in zip(outputs, expect):
            gradient_check.assert_allclose(o.data, e)

    def test_forward_cpu(self):
        self.check_forward(self.input_data)

    @attr.gpu
    def test_forward_gpu(self):
        inputs = [cuda.to_gpu(x) for x in self.input_data]
        self.check_forward(inputs)

    def check_backward(self, input_data, grad_data):
        gpu = isinstance(input_data[0], cuda.cupy.ndarray)
        func = self.make_func(gpu)
        gradient_check.check_backward(
            func, input_data, grad_data, atol=1e-4)

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
)
class TestTheanoFunction(TheanoFunctionTestBase, unittest.TestCase):

    def make_func(self, gpu):
        x = T.TensorType(self.inputs[0]['type'],
                         (False,) * len(self.inputs[0]['shape']))('x')
        y = T.TensorType(self.inputs[1]['type'],
                         (False,) * len(self.inputs[1]['shape']))('y')
        z = x + y
        return functions.TheanoFunction([x, y], [z], gpu)

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

    def make_func(self, gpu):
        x = T.TensorType(self.inputs[0]['type'],
                         (False,) * len(self.inputs[0]['shape']))('x')
        y = T.TensorType(self.inputs[1]['type'],
                         (False,) * len(self.inputs[1]['shape']))('y')
        z = x + y
        w = x - y
        return functions.TheanoFunction([x, y], [z, w], gpu)

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

    def make_func(self, gpu):
        x = T.TensorType(self.inputs[0]['type'],
                         (False,) * len(self.inputs[0]['shape']))('x')
        i = T.TensorType(self.inputs[1]['type'],
                         (False,) * len(self.inputs[1]['shape']))('y')
        z = x[i]
        return functions.TheanoFunction([x, i], z, gpu)

    def expect_forward(self):
        x, i = self.input_data
        return x[i],
