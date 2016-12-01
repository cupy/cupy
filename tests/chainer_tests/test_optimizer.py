import unittest

import mock
import numpy as np

import chainer
from chainer import cuda
from chainer import optimizer
from chainer import optimizers
from chainer import testing
from chainer.testing import attr


class TestOptimizerUtility(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(-1.0, 1.5, num=6).astype(np.float32).reshape(2, 3)
        self.a = np.array(2.0)

    def test_sqnorm_cpu(self):
        # \Sum_{n=0}^{5} (-1.0+0.5n)**2 = 4.75
        self.assertAlmostEqual(optimizer._sum_sqnorm([self.x]), 4.75)

    def test_sqnorm_scalar_cpu(self):
        self.assertAlmostEqual(optimizer._sum_sqnorm([self.a]), 4)

    @attr.gpu
    def test_sqnorm_gpu(self):
        x = cuda.to_gpu(self.x)
        self.assertAlmostEqual(optimizer._sum_sqnorm([x]), 4.75)

    @attr.gpu
    def test_sqnorm_scalar_gpu(self):
        a = cuda.to_gpu(self.a)
        self.assertAlmostEqual(optimizer._sum_sqnorm([a]), 4)

    @attr.gpu
    def test_sqnorm_array(self):
        x = cuda.to_gpu(self.x)
        a = cuda.to_gpu(self.a)
        self.assertAlmostEqual(optimizer._sum_sqnorm(
            [self.x, self.a, x, a]), 8.75 * 2)

    @attr.multi_gpu(2)
    def test_sqnorm_array_multi_gpu(self):
        x0 = cuda.to_gpu(self.x, device=0)
        x1 = cuda.to_gpu(self.x, device=1)
        a0 = cuda.to_gpu(self.a, device=0)
        a1 = cuda.to_gpu(self.a, device=1)
        self.assertAlmostEqual(optimizer._sum_sqnorm(
            [self.x, self.a, x0, a0, x1, a1]), 8.75 * 3)


class TestOptimizerHook(unittest.TestCase):

    def setUp(self):
        self.optimizer = optimizer.Optimizer()
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

    def test_add_hook(self):
        h1 = mock.MagicMock()
        self.optimizer.setup(self.target)
        self.optimizer.add_hook(h1, 'h1')
        self.optimizer.call_hooks()
        h1.assert_called_with(self.optimizer)

    def test_remove_hook(self):
        h1 = mock.MagicMock()
        self.optimizer.setup(self.target)
        self.optimizer.add_hook(h1, 'h1')
        self.optimizer.remove_hook('h1')
        self.optimizer.call_hooks()
        self.assertFalse(h1.called)

    def test_duplicated_hook(self):
        self.optimizer.setup(self.target)
        self.optimizer.add_hook(lambda s: None, 'h1')
        with self.assertRaises(KeyError):
            self.optimizer.add_hook(lambda s: None, 'h1')

    def test_invalid_hook(self):
        with self.assertRaises(TypeError):
            self.optimizer.add_hook(1)

    def test_add_hook_before_setup(self):
        with self.assertRaises(RuntimeError):
            self.optimizer.add_hook(lambda s: None, 'h1')


class SimpleLink(chainer.Link):

    def __init__(self, w, g):
        super(SimpleLink, self).__init__(param=w.shape)
        self.param.data = w
        self.param.grad = g


class TestOptimizerWeightDecay(unittest.TestCase):

    def setUp(self):
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

    def check_weight_decay(self):
        w = self.target.param.data
        g = self.target.param.grad

        decay = 0.2
        expect = w - g - decay * w

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(optimizer.WeightDecay(decay))
        opt.update()

        testing.assert_allclose(expect, w)

    def test_weight_decay_cpu(self):
        self.check_weight_decay()

    @attr.gpu
    def test_weight_decay_gpu(self):
        self.target.to_gpu()
        self.check_weight_decay()


class TestOptimizerLasso(unittest.TestCase):

    def setUp(self):
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

    def check_lasso(self):
        w = self.target.param.data
        g = self.target.param.grad
        xp = cuda.get_array_module(w)
        decay = 0.2
        expect = w - g - decay * xp.sign(w)

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(optimizer.Lasso(decay))
        opt.update()

        testing.assert_allclose(expect, w)

    def test_lasso_cpu(self):
        self.check_lasso()

    @attr.gpu
    def test_lasso_gpu(self):
        self.target.to_gpu()
        self.check_lasso()


class TestOptimizerGradientNoise(unittest.TestCase):

    eta = 0.01

    def setUp(self):
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

        self.noise_value = np.random.normal(
            loc=0, scale=np.sqrt(self.eta / np.power(1, 0.55)),
            size=(2, 3)).astype(np.float32)

    def check_gradient_noise(self):
        w = self.target.param.data
        g = self.target.param.grad
        xp = cuda.get_array_module(w)
        noise_value = xp.asarray(self.noise_value)

        expect = w - g - noise_value

        noise = mock.Mock(return_value=noise_value)
        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        hook = optimizer.GradientNoise(self.eta, noise_func=noise)
        opt.add_hook(hook)
        opt.update()

        testing.assert_allclose(expect, w, rtol=0.4)
        noise.assert_called_once_with(xp, (2, 3), np.float32, hook, opt)

    def test_gradient_noise_cpu(self):
        self.check_gradient_noise()

    @attr.gpu
    def test_gradient_noise_gpu(self):
        self.target.to_gpu()
        self.check_gradient_noise()


class TestGradientHardClipping(unittest.TestCase):

    def setUp(self):
        self.target = SimpleLink(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            np.arange(3, -3, -1, dtype=np.float32).reshape(2, 3))

    def check_hardclipping(self):
        w = self.target.param.data
        g = self.target.param.grad
        xp = cuda.get_array_module(w)
        lower_bound = -0.9
        upper_bound = 1.1
        expect = w - xp.clip(g, lower_bound, upper_bound)

        opt = optimizers.SGD(lr=1)
        opt.setup(self.target)
        opt.add_hook(optimizer.GradientHardClipping(lower_bound, upper_bound))
        opt.update()

        testing.assert_allclose(expect, w)

    def test_hardclipping_cpu(self):
        self.check_hardclipping()

    @attr.gpu
    def test_hardclipping_gpu(self):
        self.target.to_gpu()
        self.check_hardclipping()


class TestGradientMethod(unittest.TestCase):

    def _suffix(self, gpu):
        if gpu:
            return 'gpu'
        else:
            return 'cpu'

    def _get_method(self, prefix, gpu):
        return getattr(self.optimizer, prefix + '_' + self._suffix(gpu))

    def setUp(self):
        opt = chainer.GradientMethod()
        opt.init_state_cpu = mock.MagicMock()
        opt.init_state_gpu = mock.MagicMock()
        opt.update_one_cpu = mock.MagicMock()
        opt.update_one_gpu = mock.MagicMock()
        self.optimizer = opt

        self.target = SimpleLink(
            np.arange(3).astype(np.float32),
            np.arange(3).astype(np.float32))

    def setup_cpu(self):
        self.optimizer.setup(self.target)

    def setup_gpu(self, dst_id=None):
        self.target.to_gpu(dst_id)
        self.optimizer.setup(self.target)

    def check_init_state(self, gpu):
        param = chainer.Variable(np.arange(3))
        param.grad = np.arange(3)
        if gpu:
            param.to_gpu()
        state = {}
        self.optimizer.init_state(param, state)

        self._get_method('init_state', gpu).assert_called_once_with(
            param, state)
        self.assertEqual(self._get_method('init_state', not gpu).call_count, 0)

    def test_init_state_cpu(self):
        self.check_init_state(False)

    @attr.gpu
    def test_init_state_gpu(self):
        self.check_init_state(True)

    def check_update(self, gpu):
        self.assertEqual(self.optimizer.t, 0)

        self.optimizer.update()
        self.assertEqual(self.optimizer.t, 1)

        self._get_method('update_one', gpu).assert_called_once_with(
            self.target.param, {})
        self.assertEqual(self._get_method('update_one', not gpu).call_count, 0)

        self.optimizer.zero_grads()
        self.assertTrue((cuda.to_cpu(self.target.param.grad) == 0).all())

    def test_update_cpu(self):
        self.setup_cpu()
        self.check_update(False)

    @attr.gpu
    def test_update_gpu(self):
        self.setup_gpu()
        self.check_update(True)

    def check_accumulate_grads_from_cpu(self):
        self.optimizer.accumulate_grads([np.arange(3)])
        grad = self.target.param.grad
        self.assertTrue((cuda.to_cpu(grad) == np.arange(3) * 2).all())

    @attr.gpu
    def check_accumulate_grads_from_gpu(self, src_id):
        with cuda.Device(src_id):
            self.optimizer.accumulate_grads([cuda.cupy.arange(3)])
        grad = self.target.param.grad
        self.assertTrue((cuda.to_cpu(grad) == np.arange(3) * 2).all())

    def test_accumulate_grads_cpu_to_cpu(self):
        self.setup_cpu()
        self.check_accumulate_grads_from_cpu()

    @attr.gpu
    def test_accumulate_grads_cpu_to_gpu(self):
        self.setup_gpu()
        self.check_accumulate_grads_from_cpu()

    @attr.gpu
    def test_accumulate_grads_gpu_to_cpu(self):
        self.setup_cpu()
        self.check_accumulate_grads_from_gpu(cuda.Device().id)

    @attr.gpu
    def test_accumulate_grads_gpu_to_gpu(self):
        device_id = cuda.Device().id
        self.setup_gpu(device_id)
        self.check_accumulate_grads_from_gpu(device_id)

    @attr.multi_gpu(2)
    def test_accumulate_grads_multigpu(self):
        self.setup_gpu(0)
        self.check_accumulate_grads_from_gpu(1)

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
        g = cuda.to_cpu(self.target.param.grad)
        expect = np.array([0.0, 1.1, 2.2], dtype=np.float32)
        testing.assert_allclose(g, expect)

    def test_weight_decay_cpu(self):
        self.setup_cpu()
        self.check_weight_decay()

    @attr.gpu
    def test_weight_decay_gpu(self):
        self.setup_gpu()
        self.check_weight_decay()

    def check_clip_grads(self):
        self.optimizer.clip_grads(1.0)
        g = cuda.to_cpu(self.target.param.grad)
        sqnorm = g.dot(g)
        self.assertAlmostEqual(sqnorm, 1.0, delta=1.0e-5)

    def test_clip_grads_cpu(self):
        self.setup_cpu()
        self.check_clip_grads()

    @attr.gpu
    def test_clip_grads_gpu(self):
        self.setup_gpu()
        self.check_clip_grads()


class DummyOptimizer(chainer.GradientMethod):

    def __init__(self, test):
        self.test = test

    def update_one(self, param, state):
        # Confirm all grads are not None
        self.test.assertIsNotNone(param.grad)


class DummyHook(object):

    name = 'Dummy'

    def __init__(self, test):
        self.test = test

    def __call__(self, opt):
        for param in opt.target.params():
            # Confirm all grads are not None
            self.test.assertIsNotNone(param.grad)


class TestGradientMethodClearGrads(unittest.TestCase):

    def setUp(self):
        self.optimizer = DummyOptimizer(self)
        self.target = SimpleLink(
            np.arange(3).astype(np.float32),
            np.arange(3).astype(np.float32))
        self.optimizer.setup(self.target)
        self.optimizer.add_hook(DummyHook(self))

    def test_update(self):
        self.target.cleargrads()
        self.optimizer.update()


testing.run_module(__name__, __file__)
