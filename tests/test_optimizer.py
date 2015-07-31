import unittest

import mock
import numpy as np

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import optimizer
from chainer import optimizers
from chainer import testing
from chainer.testing import attr


if cuda.available:
    cuda.init()


class TestOptimizerUtility(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-1.0, 1.5, num=6).astype(np.float32).reshape(2, 3)
        self.a = np.array(2.0)

    def test_sqnorm_cpu(self):
        # \Sum_{n=0}^{5} (-1.0+0.5n)**2 = 4.75
        self.assertAlmostEqual(optimizer._sqnorm(self.x), 4.75)

    def test_sqnorm_scalar_cpu(self):
        self.assertAlmostEqual(optimizer._sqnorm(self.a), 4)

    @attr.gpu
    def test_sqnorm_gpu(self):
        x = cuda.to_gpu(self.x)
        self.assertAlmostEqual(optimizer._sqnorm(x), 4.75)

    @attr.gpu
    def test_sqnorm_scalar_gpu(self):
        a = cuda.to_gpu(self.a)
        self.assertAlmostEqual(optimizer._sqnorm(a), 4)


class TestOptimizerWeightDecay(unittest.TestCase):

    def setUp(self):
        self.w = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.g = np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3)

    def check_weight_decay(self, w, g):
        decay = 0.2
        expect = w - g - decay * w

        opt = optimizers.SGD(lr=1)
        opt.setup((w, g))
        opt.weight_decay(decay)
        opt.update()

        gradient_check.assert_allclose(expect, w)

    def test_weight_decay_cpu(self):
        self.check_weight_decay(self.w, self.g)

    @attr.gpu
    def test_weight_decay_gpu(self):
        self.check_weight_decay(cuda.to_gpu(self.w), cuda.to_gpu(self.g))


class TestOptimizer(unittest.TestCase):

    def _suffix(self, gpu):
        if gpu:
            return 'gpu'
        else:
            return 'cpu'

    def _get_method(self, prefix, gpu):
        return getattr(self.optimizer, prefix + '_' + self._suffix(gpu))

    def setUp(self):
        opt = chainer.Optimizer()
        opt.init_state_cpu = mock.MagicMock(return_value=1)
        opt.init_state_gpu = mock.MagicMock(return_value=1)
        opt.update_one_cpu = mock.MagicMock()
        opt.update_one_gpu = mock.MagicMock()
        self.optimizer = opt

        self.params = [np.arange(3).astype(np.float32)]
        self.grads = [np.arange(3).astype(np.float32)]

    def setup_cpu(self):
        self.optimizer.setup((self.params, self.grads))

    def setup_gpu(self):
        self.params = list(map(cuda.to_gpu, self.params))
        self.grads = list(map(cuda.to_gpu, self.grads))
        self.optimizer.setup((self.params, self.grads))

    def check_init_state(self, param, grad, gpu):
        state = self.optimizer.init_state(param, grad)

        self.assertEqual(state, 1)
        self._get_method('init_state', gpu).assert_called_once_with(
            param, grad)
        self.assertEqual(self._get_method('init_state', not gpu).call_count, 0)

    def test_init_state_cpu(self):
        param = np.arange(3)
        grad = np.arange(3)
        self.check_init_state(param, grad, False)

    @attr.gpu
    def test_init_state_gpu(self):
        param = cuda.to_gpu(np.arange(3))
        grad = cuda.to_gpu(np.arange(3))
        self.check_init_state(param, grad, True)

    def check_update(self, gpu):
        self.assertEqual(self.optimizer.t, 0)

        self.optimizer.update()
        self.assertEqual(self.optimizer.t, 1)

        self._get_method('update_one', gpu).assert_called_once_with(
            self.params[0], self.grads[0], 1)
        self.assertEqual(self._get_method('update_one', not gpu).call_count, 0)

        self.optimizer.zero_grads()
        self.assertTrue((cuda.to_cpu(self.grads[0]) == 0).all())

    def test_update_cpu(self):
        self.setup_cpu()
        self.check_update(False)

    @attr.gpu
    def test_update_gpu(self):
        self.setup_gpu()
        self.check_update(True)

    def check_accumulate_grads(self):
        self.optimizer.accumulate_grads([np.arange(3)])
        self.assertTrue((cuda.to_cpu(self.grads[0]) == np.arange(3) * 2).all())

    def test_accumulate_grads_cpu(self):
        self.setup_cpu()
        self.check_accumulate_grads()

    @attr.gpu
    def test_accumulate_grads_gpu(self):
        self.setup_gpu()
        self.check_accumulate_grads()

    def check_compute_grads_norm(self):
        norm = self.optimizer.compute_grads_norm()
        self.assertAlmostEqual(norm, np.sqrt(5))

    def test_compute_grads_norm_cpu(self):
        self.setup_cpu()
        self.check_compute_grads_norm()

    @attr.gpu
    def test_compute_grads_norm_gpu(self):
        self.setup_gpu()
        self.check_compute_grads_norm()

    def check_weight_decay(self):
        self.optimizer.weight_decay(0.1)
        g = cuda.to_cpu(self.grads[0])
        expect = np.array([0.0, 1.1, 2.2], dtype=np.float32)
        gradient_check.assert_allclose(g, expect)

    def test_weight_decay_cpu(self):
        self.setup_cpu()
        self.check_weight_decay()

    @attr.gpu
    def test_weight_decay_gpu(self):
        self.setup_gpu()
        self.check_weight_decay()

    def check_clip_grads(self):
        self.optimizer.clip_grads(1.0)
        g = cuda.to_cpu(self.grads[0])
        sqnorm = g.dot(g)
        self.assertAlmostEqual(sqnorm, 1.0, delta=1.0e-5)

    def test_clip_grads_cpu(self):
        self.setup_cpu()
        self.check_clip_grads()

    @attr.gpu
    def test_clip_grads_gpu(self):
        self.setup_gpu()
        self.check_clip_grads()


testing.run_module(__name__, __file__)
