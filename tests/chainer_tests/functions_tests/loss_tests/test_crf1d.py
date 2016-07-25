import unittest

import numpy

import chainer
from chainer import functions
from chainer import testing


class TestCRF1d(unittest.TestCase):

    batch = 2
    n_label = 3

    def setUp(self):
        self.cost = numpy.random.uniform(
            -1, 1, (self.n_label, self.n_label)).astype(numpy.float32)
        self.xs = [numpy.random.uniform(
            -1, 1, (self.batch, 3)).astype(numpy.float32) for _ in range(3)]
        self.ys = [
            numpy.random.randint(
                0, self.n_label, (self.batch,)).astype(numpy.int32)
            for _ in range(3)]
        self.gy = numpy.random.uniform(
            -1, 1, (self.batch,)).astype(numpy.float32)

    def test_forward(self):
        cost = chainer.Variable(self.cost)
        xs = [chainer.Variable(x) for x in self.xs]
        ys = [chainer.Variable(y) for y in self.ys]
        log_p = functions.crf1d(cost, xs, ys)

        z = numpy.zeros((self.batch,), numpy.float32)
        for y1 in range(self.n_label):
            for y2 in range(self.n_label):
                for y3 in range(self.n_label):
                    z += numpy.exp(self.xs[0][range(self.batch), y1] +
                                   self.xs[1][range(self.batch), y2] +
                                   self.xs[2][range(self.batch), y3] +
                                   self.cost[y1, y2] +
                                   self.cost[y2, y3])

        score = (self.xs[0][range(self.batch), self.ys[0]] +
                 self.xs[1][range(self.batch), self.ys[1]] +
                 self.xs[2][range(self.batch), self.ys[2]] +
                 self.cost[self.ys[0], self.ys[1]] +
                 self.cost[self.ys[1], self.ys[2]])

        expect = numpy.sum(-(score - numpy.log(z))) / self.batch
        testing.assert_allclose(log_p.data, expect)

    def test_backward(self):
        cost = chainer.Variable(self.cost)
        xs = [chainer.Variable(self.xs[i]) for i in range(3)]
        ys = [chainer.Variable(self.ys[i]) for i in range(3)]
        log_p = functions.crf1d(cost, xs, ys)
        log_p.backward()

    def test_argmax(self):
        cost = chainer.Variable(self.cost)
        xs = [chainer.Variable(self.xs[i]) for i in range(3)]
        s, path = functions.loss.crf1d.argmax_crf1d(cost, xs)

        best_paths = [numpy.empty((self.batch,), numpy.int32)
                      for i in range(len(xs))]
        for b in range(self.batch):
            best_path = None
            best_score = 0
            for y1 in range(3):
                for y2 in range(3):
                    for y3 in range(3):
                        score = (self.xs[0][b, y1] +
                                 self.xs[1][b, y2] +
                                 self.xs[2][b, y3] +
                                 self.cost[y1, y2] +
                                 self.cost[y2, y3])
                        if best_path is None or best_score < score:
                            best_path = [y1, y2, y3]
                            best_score = score

            best_paths[0][b] = best_path[0]
            best_paths[1][b] = best_path[1]
            best_paths[2][b] = best_path[2]

            testing.assert_allclose(s.data[b], best_score)

        numpy.testing.assert_array_equal(path, best_paths)


testing.run_module(__name__, __file__)
