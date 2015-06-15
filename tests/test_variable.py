from unittest import TestCase
import numpy as np

from chainer import cuda, Variable, Function
from chainer.gradient_check import assert_allclose

cuda.init()

class Constant(Function):
    def __init__(self, outputs):
        self.outputs = outputs

    def forward_cpu(self, inputs):
        return self.outputs

    def forward_gpu(self, inputs):
        return tuple(map(cuda.to_gpu, self.outputs))

    def backward_cpu(self, inputs, grad_outputs):
        return tuple(map(np.zeros_like, inputs))

    def backward_gpu(self, inputs, grad_outputs):
        return tuple(map(cuda.zeros_like, inputs))

def constant(xs, value):
    return Constant(value)(*xs)

class TestVariable(TestCase):
    def setUp(self):
        self.x = np.random.uniform(-1, 1, 10).astype(np.float32)
        self.a = np.random.uniform(0.1, 10, 10).astype(np.float32)

    def check_len(self, x, gpu):
        if gpu:
            x = cuda.to_gpu(x)
        x = Variable(x)
        self.assertEqual(len(x), 10)

    def test_len_cpu(self): self.check_len(self.x, False)
    def test_len_gpu(self): self.check_len(self.x, True)

    def check_backward(self, inputs, intermediates, outputs, retain_grad):
        for o in outputs:
            o.backward(retain_grad)

        self.assertFalse(any([x.grad is None for x in inputs]))
        intermediate_grads = [x.grad is None for x in intermediates]
        if retain_grad:
            self.assertFalse(any(intermediate_grads))
        else:
            self.assertTrue(all(intermediate_grads))
        self.assertFalse(any([x.grad is None for x in outputs]))

    def create_linear_chain(self, gpu):
        if gpu:
            x = Variable(cuda.to_gpu(self.x))
        else:
            x = Variable(self.x)
        y = constant((x, ), (self.a,))
        z = constant((y, ), (self.a,))
        z.grad = np.zeros_like(z.data)
        return (x, y, z)

    def test_backward_cpu(self):
        x, y, z = self.create_linear_chain(False)
        self.check_backward((x, ), (y, ), (z, ), False)

    def test_backward_gpu(self):
        x, y, z = self.create_linear_chain(True)
        self.check_backward((x, ), (y, ), (z, ), False)

    def test_backward_cpu_retain_grad(self):
        x, y, z = self.create_linear_chain(False)
        self.check_backward((x, ), (y, ), (z, ), True)

    def test_backward_gpu_retain_grad(self):
        x, y, z = self.create_linear_chain(True)
        self.check_backward((x, ), (y, ), (z, ), True)

