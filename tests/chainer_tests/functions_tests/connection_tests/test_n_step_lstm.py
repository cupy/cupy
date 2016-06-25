import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def sigmoid(x):
    return numpy.tanh(x * 0.5) * 0.5 + 0.5


def _split(inputs, pos):
    return inputs[:pos], inputs[pos:]


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
}))
class TestNStepLSTM(unittest.TestCase):

    batches = [4, 3, 2, 1]
    length = len(batches)
    in_size = 3
    out_size = 4
    n_layers = 2
    dropout = 0.0
    seed = 1337

    def setUp(self):
        self.xs = [numpy.random.uniform(-1, 1, (b, self.in_size)).astype('f')
                   for b in self.batches]
        h_shape = (self.n_layers, self.batches[0], self.out_size)
        self.cx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.hx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

        self.ws = []
        self.bs = []
        for i in range(self.n_layers):
            for j in range(8):
                if i == 0 and j < 4:
                    w_in = self.in_size
                else:
                    w_in = self.out_size

                self.ws.append(numpy.random.uniform(
                    -1, 1, (self.out_size, w_in)).astype('f'))
                self.bs.append(numpy.random.uniform(
                    -1, 1, (self.out_size,)).astype('f'))

        self.dys = [numpy.random.uniform(-1, 1, (b, self.out_size)).astype('f')
                    for b in self.batches]
        self.dcy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.dhy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

    def check_forward(
            self, h_data, c_data, xs_data, ws_data, bs_data, volatile):
        h = chainer.Variable(h_data, volatile=volatile)
        c = chainer.Variable(c_data, volatile=volatile)
        xs = [chainer.Variable(x, volatile=volatile) for x in xs_data]
        ws = [chainer.Variable(w, volatile=volatile) for w in ws_data]
        bs = [chainer.Variable(b, volatile=volatile) for b in bs_data]
        hy, cy, ys = functions.n_step_lstm(
            self.n_layers, self.dropout, h, c, ws, bs, xs,
            use_cudnn=self.use_cudnn)

        e_hy = self.hx.copy()
        e_cy = self.cx.copy()
        for ind in range(self.length):
            x = self.xs[ind]
            batch = x.shape[0]
            for layer in range(self.n_layers):
                w = self.ws[layer * 8: layer * 8 + 8]
                b = self.bs[layer * 8: layer * 8 + 8]
                h_prev = e_hy[layer, :batch]
                c_prev = e_cy[layer, :batch]
                i = sigmoid(x.dot(w[0].T) + h_prev.dot(w[4].T) + b[0] + b[4])
                f = sigmoid(x.dot(w[1].T) + h_prev.dot(w[5].T) + b[1] + b[5])
                c_bar = numpy.tanh(
                    x.dot(w[2].T) + h_prev.dot(w[6].T) + b[2] + b[6])
                o = sigmoid(x.dot(w[3].T) + h_prev.dot(w[7].T) + b[3] + b[7])
                e_c = (f * c_prev + i * c_bar)
                e_h = o * numpy.tanh(e_c)
                e_hy[layer, :batch] = e_h
                e_cy[layer, :batch] = e_c

                x = e_h

            gradient_check.assert_allclose(
                ys[ind].data, x, rtol=1e-4, atol=1e-4)

        gradient_check.assert_allclose(hy.data, e_hy, rtol=1e-4, atol=1e-4)
        gradient_check.assert_allclose(cy.data, e_cy, rtol=1e-4, atol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.hx, self.cx, self.xs, self.ws, self.bs, False)

    def test_forward_cpu_volatile(self):
        self.check_forward(self.hx, self.cx, self.xs, self.ws, self.bs, True)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.hx),
                           cuda.to_gpu(self.cx),
                           [cuda.to_gpu(x) for x in self.xs],
                           [cuda.to_gpu(w) for w in self.ws],
                           [cuda.to_gpu(b) for b in self.bs],
                           False)

    @attr.gpu
    def test_forward_gpu_volatile(self):
        self.check_forward(cuda.to_gpu(self.hx),
                           cuda.to_gpu(self.cx),
                           [cuda.to_gpu(x) for x in self.xs],
                           [cuda.to_gpu(w) for w in self.ws],
                           [cuda.to_gpu(b) for b in self.bs],
                           True)

    def check_backward(self, h_data, c_data, xs_data, ws_data, bs_data,
                       dhy_data, dcy_data, dys_data):
        args = tuple([h_data, c_data] + ws_data + bs_data + xs_data)
        grads = tuple([dhy_data, dcy_data] + dys_data)

        def f(*inputs):
            (hx, cx), inputs = _split(inputs, 2)
            ws, inputs = _split(inputs, 8 * self.n_layers)
            bs, inputs = _split(inputs, 8 * self.n_layers)
            xs = inputs
            hy, cy, ys = functions.n_step_lstm(
                self.n_layers, self.dropout, hx, cx, ws, bs, xs)
            return (hy, cy) + ys

        gradient_check.check_backward(
            f, args, grads, eps=1e-2, rtol=1e-4, atol=1e-4)

    def test_backward_cpu(self):
        self.check_backward(self.hx, self.cx, self.xs, self.ws, self.bs,
                            self.dhy, self.dcy, self.dys)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.hx),
                            cuda.to_gpu(self.cx),
                            [cuda.to_gpu(x) for x in self.xs],
                            [cuda.to_gpu(w) for w in self.ws],
                            [cuda.to_gpu(b) for b in self.bs],
                            cuda.to_gpu(self.dhy),
                            cuda.to_gpu(self.dcy),
                            [cuda.to_gpu(dy) for dy in self.dys])


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
}))
@attr.cudnn
class TestNStepLSTMCudnnCall(unittest.TestCase):

    batches = [4, 3, 2, 1]
    length = len(batches)
    in_size = 3
    out_size = 4
    n_layers = 2
    dropout = 0.0
    seed = 1337

    def setUp(self):
        self.xs = [cuda.cupy.random.uniform(
            -1, 1, (b, self.in_size)).astype('f')
            for b in self.batches]
        h_shape = (self.n_layers, self.batches[0], self.out_size)
        self.cx = cuda.cupy.random.uniform(-1, 1, h_shape).astype('f')
        self.hx = cuda.cupy.random.uniform(-1, 1, h_shape).astype('f')

        self.ws = []
        self.bs = []
        for i in range(self.n_layers):
            for j in range(8):
                if i == 0 and j < 4:
                    w_in = self.in_size
                else:
                    w_in = self.out_size

                self.ws.append(cuda.cupy.random.uniform(
                    -1, 1, (self.out_size, w_in)).astype('f'))
                self.bs.append(cuda.cupy.random.uniform(
                    -1, 1, (self.out_size,)).astype('f'))

        self.dys = [cuda.cupy.random.uniform(
            -1, 1, (b, self.out_size)).astype('f')
            for b in self.batches]
        self.dcy = cuda.cupy.random.uniform(-1, 1, h_shape).astype('f')
        self.dhy = cuda.cupy.random.uniform(-1, 1, h_shape).astype('f')
        self.expect = self.use_cudnn and (
            cuda.cudnn.cudnn.getVersion() >= 5000)

    def forward(self, volatile):
        h = chainer.Variable(self.hx, volatile=volatile)
        c = chainer.Variable(self.cx, volatile=volatile)
        xs = [chainer.Variable(x, volatile=volatile) for x in self.xs]
        ws = [chainer.Variable(w, volatile=volatile) for w in self.ws]
        bs = [chainer.Variable(b, volatile=volatile) for b in self.bs]
        return functions.n_step_lstm(
            self.n_layers, self.dropout, h, c, ws, bs, xs,
            use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward_training(self):
        with mock.patch('cupy.cuda.cudnn.RNNForwardTraining') as func:
            self.forward(False)
            self.assertEqual(func.called, self.expect)

    def test_call_cudnn_forward_inference(self):
        with mock.patch('cupy.cuda.cudnn.RNNForwardInference') as func:
            self.forward(True)
            self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        hy, cy, ys = self.forward(False)
        hy.grad = self.dhy
        with mock.patch('cupy.cuda.cudnn.RNNBackwardWeights') as func:
            hy.backward()
            self.assertEqual(func.called, self.expect)


testing.run_module(__name__, __file__)
