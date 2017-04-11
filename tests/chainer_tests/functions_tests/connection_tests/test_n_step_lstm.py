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

    batches = [3, 2, 1]
    length = len(batches)
    in_size = 3
    out_size = 2
    n_layers = 2
    dropout = 0.0

    def setUp(self):
        self.xs = [numpy.random.uniform(-1, 1, (b, self.in_size)).astype('f')
                   for b in self.batches]
        h_shape = (self.n_layers, self.batches[0], self.out_size)
        self.cx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.hx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

        self.ws = []
        self.bs = []
        for i in range(self.n_layers):
            weights = []
            biases = []
            for j in range(8):
                if i == 0 and j < 4:
                    w_in = self.in_size
                else:
                    w_in = self.out_size

                weights.append(numpy.random.uniform(
                    -1, 1, (self.out_size, w_in)).astype('f'))
                biases.append(numpy.random.uniform(
                    -1, 1, (self.out_size,)).astype('f'))
            self.ws.append(weights)
            self.bs.append(biases)

        self.dys = [numpy.random.uniform(-1, 1, (b, self.out_size)).astype('f')
                    for b in self.batches]
        self.dcy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.dhy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

    def check_forward(
            self, h_data, c_data, xs_data, ws_data, bs_data, volatile):
        h = chainer.Variable(h_data, volatile=volatile)
        c = chainer.Variable(c_data, volatile=volatile)
        xs = [chainer.Variable(x, volatile=volatile) for x in xs_data]
        ws = [[chainer.Variable(w, volatile=volatile) for w in ws]
              for ws in ws_data]
        bs = [[chainer.Variable(b, volatile=volatile) for b in bs]
              for bs in bs_data]
        hy, cy, ys = functions.n_step_lstm(
            self.n_layers, self.dropout, h, c, ws, bs, xs,
            use_cudnn=self.use_cudnn)

        e_hy = self.hx.copy()
        e_cy = self.cx.copy()
        for ind in range(self.length):
            x = self.xs[ind]
            batch = x.shape[0]
            for layer in range(self.n_layers):
                w = self.ws[layer]
                b = self.bs[layer]
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

            testing.assert_allclose(
                ys[ind].data, x, rtol=1e-4, atol=1e-4)

        testing.assert_allclose(hy.data, e_hy, rtol=1e-4, atol=1e-4)
        testing.assert_allclose(cy.data, e_cy, rtol=1e-4, atol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.hx, self.cx, self.xs, self.ws, self.bs, False)

    def test_forward_cpu_volatile(self):
        self.check_forward(self.hx, self.cx, self.xs, self.ws, self.bs, True)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.hx),
                           cuda.to_gpu(self.cx),
                           [cuda.to_gpu(x) for x in self.xs],
                           [[cuda.to_gpu(w) for w in ws] for ws in self.ws],
                           [[cuda.to_gpu(b) for b in bs] for bs in self.bs],
                           False)

    @attr.gpu
    def test_forward_gpu_volatile(self):
        self.check_forward(cuda.to_gpu(self.hx),
                           cuda.to_gpu(self.cx),
                           [cuda.to_gpu(x) for x in self.xs],
                           [[cuda.to_gpu(w) for w in ws] for ws in self.ws],
                           [[cuda.to_gpu(b) for b in bs] for bs in self.bs],
                           True)

    def check_backward(self, h_data, c_data, xs_data, ws_data, bs_data,
                       dhy_data, dcy_data, dys_data):
        args = tuple([h_data, c_data] + sum(ws_data, []) + sum(bs_data, []) +
                     xs_data)
        grads = tuple([dhy_data, dcy_data] + dys_data)

        def f(*inputs):
            (hx, cx), inputs = _split(inputs, 2)
            ws = []
            for i in range(self.n_layers):
                weights, inputs = _split(inputs, 8)
                ws.append(weights)
            bs = []
            for i in range(self.n_layers):
                biases, inputs = _split(inputs, 8)
                bs.append(biases)
            xs = inputs
            hy, cy, ys = functions.n_step_lstm(
                self.n_layers, self.dropout, hx, cx, ws, bs, xs)
            return (hy, cy) + ys

        gradient_check.check_backward(
            f, args, grads, eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.hx, self.cx, self.xs, self.ws, self.bs,
                            self.dhy, self.dcy, self.dys)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.hx),
                            cuda.to_gpu(self.cx),
                            [cuda.to_gpu(x) for x in self.xs],
                            [[cuda.to_gpu(w) for w in ws] for ws in self.ws],
                            [[cuda.to_gpu(b) for b in bs] for bs in self.bs],
                            cuda.to_gpu(self.dhy),
                            cuda.to_gpu(self.dcy),
                            [cuda.to_gpu(dy) for dy in self.dys])


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
}))
class TestNStepBiLSTM(unittest.TestCase):

    batches = [3, 2, 1]
    length = len(batches)
    in_size = 3
    out_size = 2
    n_layers = 3
    dropout = 0.0

    def setUp(self):
        self.xs = [numpy.random.uniform(-1, 1, (b, self.in_size)).astype('f')
                   for b in self.batches]
        h_shape = (self.n_layers * 2, self.batches[0], self.out_size)
        self.cx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.hx = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

        self.ws = []
        self.bs = []
        for i in range(self.n_layers):
            for di in [0, 1]:
                weights = []
                biases = []
                for j in range(8):
                    if i == 0 and j < 4:
                        w_in = self.in_size
                    elif i > 0 and j < 4:
                        w_in = self.out_size * 2
                    else:
                        w_in = self.out_size

                    weights.append(numpy.random.uniform(
                        -1, 1, (self.out_size, w_in)).astype('f'))
                    biases.append(numpy.random.uniform(
                        -1, 1, (self.out_size,)).astype('f'))
                self.ws.append(weights)
                self.bs.append(biases)

        self.dys = [numpy.random.uniform(-1, 1, (b, self.out_size * 2))
                    .astype('f') for b in self.batches]
        self.dcy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)
        self.dhy = numpy.random.uniform(-1, 1, h_shape).astype(numpy.float32)

    def check_forward(
            self, h_data, c_data, xs_data, ws_data, bs_data, volatile):
        h = chainer.Variable(h_data, volatile=volatile)
        c = chainer.Variable(c_data, volatile=volatile)
        xs = [chainer.Variable(x, volatile=volatile) for x in xs_data]
        ws = [[chainer.Variable(w, volatile=volatile) for w in ws]
              for ws in ws_data]
        bs = [[chainer.Variable(b, volatile=volatile) for b in bs]
              for bs in bs_data]
        hy, cy, ys = functions.n_step_bilstm(
            self.n_layers, self.dropout, h, c, ws, bs, xs,
            use_cudnn=self.use_cudnn)

        xs_next = self.xs
        e_hy = self.hx.copy()
        e_cy = self.cx.copy()
        for layer in range(self.n_layers):
            # forward
            di = 0
            xf = []
            layer_idx = layer * 2 + di
            w = self.ws[layer_idx]
            b = self.bs[layer_idx]
            for ind in range(self.length):
                x = xs_next[ind]
                batch = x.shape[0]
                h_prev = e_hy[layer_idx, :batch]
                c_prev = e_cy[layer_idx, :batch]
                i = sigmoid(x.dot(w[0].T) + h_prev.dot(w[4].T) + b[0] + b[4])
                f = sigmoid(x.dot(w[1].T) + h_prev.dot(w[5].T) + b[1] + b[5])
                c_bar = numpy.tanh(
                    x.dot(w[2].T) + h_prev.dot(w[6].T) + b[2] + b[6])
                o = sigmoid(x.dot(w[3].T) + h_prev.dot(w[7].T) + b[3] + b[7])
                e_c = (f * c_prev + i * c_bar)
                e_h = o * numpy.tanh(e_c)
                e_hy[layer_idx, :batch] = e_h
                e_cy[layer_idx, :batch] = e_c

                xf.append(e_h)

            # backward
            di = 1
            xb = []
            layer_idx = layer * 2 + di
            w = self.ws[layer_idx]
            b = self.bs[layer_idx]
            for ind in reversed(range(self.length)):
                x = xs_next[ind]
                batch = x.shape[0]
                h_prev = e_hy[layer_idx, :batch]
                c_prev = e_cy[layer_idx, :batch]
                i = sigmoid(x.dot(w[0].T) + h_prev.dot(w[4].T) + b[0] + b[4])
                f = sigmoid(x.dot(w[1].T) + h_prev.dot(w[5].T) + b[1] + b[5])
                c_bar = numpy.tanh(
                    x.dot(w[2].T) + h_prev.dot(w[6].T) + b[2] + b[6])
                o = sigmoid(x.dot(w[3].T) + h_prev.dot(w[7].T) + b[3] + b[7])
                e_c = (f * c_prev + i * c_bar)
                e_h = o * numpy.tanh(e_c)
                e_hy[layer_idx, :batch] = e_h
                e_cy[layer_idx, :batch] = e_c

                xb.append(e_h)

            xb.reverse()
            xs_next = [numpy.concatenate([hfi, hbi], axis=1) for (hfi, hbi) in
                       zip(xf, xb)]

        for k, (ysi, xsi) in enumerate(zip(ys, xs_next)):
            testing.assert_allclose(ysi.data, xsi, rtol=1e-4, atol=1e-4)

        testing.assert_allclose(hy.data, e_hy, rtol=1e-4, atol=1e-4)
        testing.assert_allclose(cy.data, e_cy, rtol=1e-4, atol=1e-4)

    def test_forward_cpu(self):
        self.check_forward(self.hx, self.cx, self.xs, self.ws, self.bs, False)

    def test_forward_cpu_volatile(self):
        self.check_forward(self.hx, self.cx, self.xs, self.ws, self.bs, True)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.hx),
                           cuda.to_gpu(self.cx),
                           [cuda.to_gpu(x) for x in self.xs],
                           [[cuda.to_gpu(w) for w in ws] for ws in self.ws],
                           [[cuda.to_gpu(b) for b in bs] for bs in self.bs],
                           False)

    @attr.gpu
    def test_forward_gpu_volatile(self):
        self.check_forward(cuda.to_gpu(self.hx),
                           cuda.to_gpu(self.cx),
                           [cuda.to_gpu(x) for x in self.xs],
                           [[cuda.to_gpu(w) for w in ws] for ws in self.ws],
                           [[cuda.to_gpu(b) for b in bs] for bs in self.bs],
                           True)

    def check_backward(self, h_data, c_data, xs_data, ws_data, bs_data,
                       dhy_data, dcy_data, dys_data):
        args = tuple([h_data, c_data] + sum(ws_data, []) + sum(bs_data, []) +
                     xs_data)
        grads = tuple([dhy_data, dcy_data] + dys_data)

        def f(*inputs):
            (hx, cx), inputs = _split(inputs, 2)
            ws = []
            for i in range(self.n_layers * 2):
                weights, inputs = _split(inputs, 8)
                ws.append(weights)
            bs = []
            for i in range(self.n_layers * 2):
                biases, inputs = _split(inputs, 8)
                bs.append(biases)
            xs = inputs
            hy, cy, ys = functions.n_step_bilstm(
                self.n_layers, self.dropout, hx, cx, ws, bs, xs,
                use_cudnn=self.use_cudnn)
            return (hy, cy) + ys

        gradient_check.check_backward(
            f, args, grads, eps=1e-2, rtol=1e-3, atol=1e-3)

    def test_backward_cpu(self):
        self.check_backward(self.hx, self.cx, self.xs, self.ws, self.bs,
                            self.dhy, self.dcy, self.dys)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.hx),
                            cuda.to_gpu(self.cx),
                            [cuda.to_gpu(x) for x in self.xs],
                            [[cuda.to_gpu(w) for w in ws] for ws in self.ws],
                            [[cuda.to_gpu(b) for b in bs] for bs in self.bs],
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
            weights = []
            biases = []
            for j in range(8):
                if i == 0 and j < 4:
                    w_in = self.in_size
                else:
                    w_in = self.out_size

                weights.append(cuda.cupy.random.uniform(
                    -1, 1, (self.out_size, w_in)).astype('f'))
                biases.append(cuda.cupy.random.uniform(
                    -1, 1, (self.out_size,)).astype('f'))

            self.ws.append(weights)
            self.bs.append(biases)

        self.dys = [cuda.cupy.random.uniform(
            -1, 1, (b, self.out_size)).astype('f')
            for b in self.batches]
        self.dcy = cuda.cupy.random.uniform(-1, 1, h_shape).astype('f')
        self.dhy = cuda.cupy.random.uniform(-1, 1, h_shape).astype('f')
        self.expect = self.use_cudnn and (
            cuda.cudnn.cudnn.getVersion() >= 5000)

    def forward(self, train):
        volatile = not train
        h = chainer.Variable(self.hx, volatile=volatile)
        c = chainer.Variable(self.cx, volatile=volatile)
        xs = [chainer.Variable(x, volatile=volatile) for x in self.xs]
        ws = [[chainer.Variable(w, volatile=volatile) for w in ws]
              for ws in self.ws]
        bs = [[chainer.Variable(b, volatile=volatile) for b in bs]
              for bs in self.bs]
        return functions.n_step_lstm(
            self.n_layers, self.dropout, h, c, ws, bs, xs,
            train=train, use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward_training(self):
        with mock.patch('cupy.cuda.cudnn.RNNForwardTraining') as func:
            self.forward(True)
            self.assertEqual(func.called, self.expect)

    def test_call_cudnn_forward_inference(self):
        with mock.patch('cupy.cuda.cudnn.RNNForwardInference') as func:
            self.forward(False)
            self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        hy, cy, ys = self.forward(True)
        hy.grad = self.dhy
        with mock.patch('cupy.cuda.cudnn.RNNBackwardWeights') as func:
            hy.backward()
            self.assertEqual(func.called, self.expect)


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
}))
@attr.cudnn
class TestNStepBiLSTMCudnnCall(unittest.TestCase):

    batches = [4, 3, 2, 1]
    length = len(batches)
    in_size = 3
    out_size = 4
    n_layers = 2
    dropout = 0.0

    def setUp(self):
        self.xs = [cuda.cupy.random.uniform(
            -1, 1, (b, self.in_size)).astype('f')
            for b in self.batches]
        h_shape = (self.n_layers * 2, self.batches[0], self.out_size)
        self.cx = cuda.cupy.random.uniform(-1, 1, h_shape).astype('f')
        self.hx = cuda.cupy.random.uniform(-1, 1, h_shape).astype('f')

        self.ws = []
        self.bs = []
        for i in range(self.n_layers):
            for di in [0, 1]:
                weights = []
                biases = []
                for j in range(8):
                    if i == 0 and j < 4:
                        w_in = self.in_size
                    elif i > 0 and j < 4:
                        w_in = self.out_size * 2
                    else:
                        w_in = self.out_size

                    weights.append(cuda.cupy.random.uniform(
                        -1, 1, (self.out_size, w_in)).astype('f'))
                    biases.append(cuda.cupy.random.uniform(
                        -1, 1, (self.out_size,)).astype('f'))

                self.ws.append(weights)
                self.bs.append(biases)

        self.dys = [cuda.cupy.random.uniform(
            -1, 1, (b, self.out_size * 2)).astype('f')
            for b in self.batches]
        self.dcy = cuda.cupy.random.uniform(-1, 1, h_shape).astype('f')
        self.dhy = cuda.cupy.random.uniform(-1, 1, h_shape).astype('f')
        self.expect = self.use_cudnn and (
            cuda.cudnn.cudnn.getVersion() >= 5000)

    def forward(self, train):
        volatile = not train
        h = chainer.Variable(self.hx, volatile=volatile)
        c = chainer.Variable(self.cx, volatile=volatile)
        xs = [chainer.Variable(x, volatile=volatile) for x in self.xs]
        ws = [[chainer.Variable(w, volatile=volatile) for w in ws]
              for ws in self.ws]
        bs = [[chainer.Variable(b, volatile=volatile) for b in bs]
              for bs in self.bs]
        return functions.n_step_bilstm(
            self.n_layers, self.dropout, h, c, ws, bs, xs,
            train=train, use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward_training(self):
        with mock.patch('cupy.cuda.cudnn.RNNForwardTraining') as func:
            self.forward(True)
            self.assertEqual(func.called, self.expect)

    def test_call_cudnn_forward_inference(self):
        with mock.patch('cupy.cuda.cudnn.RNNForwardInference') as func:
            self.forward(False)
            self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        hy, cy, ys = self.forward(True)
        hy.grad = self.dhy
        with mock.patch('cupy.cuda.cudnn.RNNBackwardWeights') as func:
            hy.backward()
            self.assertEqual(func.called, self.expect)


def _stack_weight(ws):
    # TODO(unno): Input of the current LSTM implementaiton is shuffled
    w = functions.stack(ws, axis=1)
    shape = w.shape
    return functions.reshape(w, (shape[0] * shape[1],) + shape[2:])


def count_close(x, y, atol=1e-4):
    assert x.shape == y.shape
    return int(sum(abs(x - y) / abs(x) < atol))


def lstm_without_dropout(n_layer, dropout, hx, cx, ws, bs, xs):
    xws = [_stack_weight([w[2], w[0], w[1], w[3]]) for w in ws]
    hws = [_stack_weight([w[6], w[4], w[5], w[7]]) for w in ws]
    xbs = [_stack_weight([b[2], b[0], b[1], b[3]]) for b in bs]
    hbs = [_stack_weight([b[6], b[4], b[5], b[7]]) for b in bs]
    xs = [xs[i] for i in range(3)]
    ys = []
    for x in xs:
        cx_next = []
        hx_next = []
        for layer in range(n_layer):
            c = cx[layer]
            h = hx[layer]

            if layer != 0:
                # Only multiply ratio
                x = x * (1 / (1.0 - dropout))
            lstm_in = functions.linear(x, xws[layer], xbs[layer]) + \
                functions.linear(h, hws[layer], hbs[layer])
            c_new, h_new = functions.lstm(c, lstm_in)
            cx_next.append(c_new)
            hx_next.append(h_new)
            x = h_new
        cx = cx_next
        hx = hx_next
        ys.append(x)
    cy = functions.stack(cx)
    hy = functions.stack(hx)
    return hy, cy, ys


def rand_vector(shape):
    # return cuda.cupy.random.randint(-2, 2, shape).astype('f')
    return cuda.cupy.random.uniform(-1, 1, shape).astype('f')
    # return cuda.cupy.ones(shape).astype('f')


@testing.parameterize(*testing.product({
    'use_cudnn': [True, False],
}))
@attr.cudnn
class TestNStepLSTMDropout(unittest.TestCase):

    batch = 20
    length = 3
    in_size = 1
    out_size = 1
    n_layers = 2
    dropout = 0.3
    n_tests = 100

    def setUp(self):
        self.xs = [rand_vector((self.batch, self.in_size))
                   for _ in range(self.length)]
        h_shape = (self.n_layers, self.batch, self.out_size)
        self.cx = rand_vector(h_shape)
        self.hx = rand_vector(h_shape)
        self.ws = []
        self.bs = []
        for i in range(self.n_layers):
            weights = []
            biases = []
            for j in range(8):
                if i == 0 and j < 4:
                    w_in = self.in_size
                else:
                    w_in = self.out_size

                weights.append(rand_vector((self.out_size, w_in)))
                biases.append(rand_vector((self.out_size,)))

            self.ws.append(weights)
            self.bs.append(biases)

    def assert_count(self, actual, expect):
        self.assertTrue(expect * 0.8 < actual < expect * 1.2)

    def test_forward_dropout_count(self):
        y_counts = [0] * self.length
        h_counts = [0] * self.n_layers
        c_counts = [0] * self.n_layers

        for _ in range(self.n_tests):
            hy1, cy1, ys1 = lstm_without_dropout(
                self.n_layers, self.dropout, self.hx, self.cx, self.ws,
                self.bs, self.xs)
            hy2, cy2, ys2 = functions.n_step_lstm(
                self.n_layers, self.dropout, self.hx, self.cx, self.ws,
                self.bs, self.xs, train=True, use_cudnn=self.use_cudnn)

            for i in range(self.length):
                y_counts[i] += count_close(ys1[i].data, ys2[i].data)

            for i in range(self.n_layers):
                h_counts[i] += count_close(hy1[i].data, hy2[i].data)
                c_counts[i] += count_close(cy1[i].data, cy2[i].data)

        total = self.batch * self.n_tests
        for i in range(self.length):
            self.assert_count(
                y_counts[i],
                total * (1 - self.dropout) ** ((self.n_layers - 1) * (i + 1)))
        for i in range(self.n_layers):
            self.assert_count(
                h_counts[i], total * (1 - self.dropout) ** (self.length * i))
            self.assert_count(
                c_counts[i], total * (1 - self.dropout) ** (self.length * i))

testing.run_module(__name__, __file__)
