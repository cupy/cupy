import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer import links
from chainer import variable


def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


class TestPeephole(unittest.TestCase):

    def setUp(self):
        self.link = links.Peephole(6, 8) 
        peep_i = self.link.peep_i.W.data
        peep_i[...] = numpy.random.uniform(-1, 1, peep_i.shape)
        peep_f = self.link.peep_f.W.data
        peep_f[...] = numpy.random.uniform(-1, 1, peep_f.shape)
        peep_o = self.link.peep_o.W.data
        peep_o[...] = numpy.random.uniform(-1, 1, peep_o.shape)
        self.link.zerograds()

        self.peep_i = peep_i.copy()
        self.peep_f = peep_f.copy()
        self.peep_o = peep_o.copy()

        self.c_prev = numpy.random.uniform(-1,
                                           1, (3, 2, 4)).astype(numpy.float32)
        self.x = numpy.random.uniform(-1, 1, (3, 8, 4)).astype(numpy.float32)

        self.gc = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)
        self.gh = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(numpy.float32)

    def flat(self):
        self.c_prev = self.c_prev[:, :, 0].copy()
        self.x = self.x[:, :, 0].copy()
        self.gc = self.gc[:, :, 0].copy()
        self.gh = self.gh[:, :, 0].copy()

    def check_forward(self, c_prev_data, x_data):
        c_prev = chainer.Variable(c_prev_data)
        x = chainer.Variable(x_data)
        c, h = functions.peephole(c_prev, x, self.link.peep_i, self.link.peep_f, self.link.peep_o)
        self.assertEqual(c.data.dtype, numpy.float32)
        self.assertEqual(h.data.dtype, numpy.float32)
        
        # Compute expected out
        a_in = self.x[:, [0, 4]]
        i_in = self.x[:, [1, 5]]
        f_in = self.x[:, [2, 6]]
        o_in = self.x[:, [3, 7]]

        peep_in_i = self.link.peep_i(c_prev)
        peep_in_i.data = numpy.reshape(peep_in_i.data, i_in.shape) 
        peep_in_f = self.link.peep_f(c_prev)
        peep_in_f.data = numpy.reshape(peep_in_f.data, f_in.shape) 
        c_expect = _sigmoid(i_in + peep_in_i.data) * numpy.tanh(a_in) + _sigmoid(f_in + peep_in_f.data) * self.c_prev
        peep_in_o = self.link.peep_o(variable.Variable(c_expect))
        peep_in_o.data = numpy.reshape(peep_in_o.data, o_in.shape) 
        h_expect = _sigmoid(o_in + peep_in_o.data) * numpy.tanh(c_expect)

        gradient_check.assert_allclose(c_expect, c.data)
        gradient_check.assert_allclose(h_expect, h.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.c_prev, self.x)

    @condition.retry(3)
    def test_flat_forward_cpu(self):
        self.flat()
        self.test_forward_cpu()

    #@attr.gpu
    #@condition.retry(3)
    #def test_forward_gpu(self):
    #    self.check_forward(cuda.to_gpu(self.c_prev), cuda.to_gpu(self.x))

    #@attr.gpu
    #@condition.retry(3)
    #def test_flat_forward_gpu(self):
    #    self.flat()
    #    self.test_forward_gpu()

    #def check_backward(self, c_prev_data, x_data, c_grad, h_grad):
    #    gradient_check.check_backward(
    #        functions.Peephole(),
    #        (c_prev_data, x_data), (c_grad, h_grad), eps=1e-2)

    #@condition.retry(3)
    #def test_full_backward_cpu(self):
    #    self.check_backward(self.c_prev, self.x, self.gc, self.gh)

    #@condition.retry(3)
    #def test_flat_full_backward_cpu(self):
    #    self.flat()
    #    self.test_full_backward_cpu()

    #@condition.retry(3)
    #def test_no_gc_backward_cpu(self):
    #    self.check_backward(self.c_prev, self.x, None, self.gh)

    #@condition.retry(3)
    #def test_flat_no_gc_backward_cpu(self):
    #    self.flat()
    #    self.test_no_gc_backward_cpu()

    #@condition.retry(3)
    #def test_no_gh_backward_cpu(self):
    #    self.check_backward(self.c_prev, self.x, self.gc, None)

    #@condition.retry(3)
    #def test_flat_no_gh_backward_cpu(self):
    #    self.flat()
    #    self.test_no_gh_backward_cpu()

    #@attr.gpu
    #@condition.retry(3)
    #def test_full_backward_gpu(self):
    #    self.check_backward(
    #        cuda.to_gpu(self.c_prev), cuda.to_gpu(self.x),
    #        cuda.to_gpu(self.gc), cuda.to_gpu(self.gh))

    #@attr.gpu
    #@condition.retry(3)
    #def test_flat_full_backward_gpu(self):
    #    self.flat()
    #    self.test_full_backward_gpu()

    #@attr.gpu
    #@condition.retry(3)
    #def test_no_gc_backward_gpu(self):
    #    self.check_backward(
    #        cuda.to_gpu(self.c_prev), cuda.to_gpu(self.x),
    #        None, cuda.to_gpu(self.gh))

    #@attr.gpu
    #@condition.retry(3)
    #def test_flat_no_gc_backward_gpu(self):
    #    self.flat()
    #    self.test_no_gc_backward_gpu()

    #@attr.gpu
    #@condition.retry(3)
    #def test_no_gh_backward_gpu(self):
    #    self.check_backward(
    #        cuda.to_gpu(self.c_prev), cuda.to_gpu(self.x),
    #        cuda.to_gpu(self.gc), None)

    #@attr.gpu
    #@condition.retry(3)
    #def test_flat_no_gh_backward_gpu(self):
    #    self.flat()
    #    self.test_no_gh_backward_gpu()


testing.run_module(__name__, __file__)
