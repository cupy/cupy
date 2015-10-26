import unittest

import numpy as np
import six
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
from chainer import links as L
from chainer import testing
from chainer.testing import attr


class SimpleLink(chainer.Link):

    def __init__(self, shape):
        super(SimpleLink, self).__init__(p=shape)
        self.p.data.fill(0)
        self.p.grad.fill(1)


class TestFunctionSetBase(unittest.TestCase):

    def _check_setter(self, fs, gpu, attribute):
        expect = getattr(fs, attribute)
        setattr(fs, attribute, expect)
        actual = getattr(fs, attribute)
        if gpu:
            expect = tuple(p.get() for p in expect)
            actual = tuple(p.get() for p in actual)

        self.assertEqual(len(expect), len(actual))
        for e, a in zip(expect, actual):
            np.testing.assert_array_equal(e, a)

    def _check_setter_invalid(self, fs, diff, xp, attribute):
        if diff > 0:
            values = getattr(fs, attribute) + (xp.empty((1,)),) * diff
        else:
            values = getattr(fs, attribute)[:diff]

        with self.assertRaises(AssertionError):
            setattr(fs, attribute, values)


class TestNestedFunctionSet(TestFunctionSetBase):

    def setUp(self):
        self.fs1 = chainer.FunctionSet(
            a=SimpleLink((1, 2)))
        self.fs2 = chainer.FunctionSet(
            fs1=self.fs1,
            b=SimpleLink((3, 4)))

    def test_collect_parameters(self):
        params = self.fs2.parameters
        grads = self.fs2.gradients
        self.assertEqual(len(params), 2)
        self.assertEqual(len(grads), 2)

        self.assertIn(params[0].shape, ((1, 2), (3, 4)))
        idx_a = 0 if params[0].shape == (1, 2) else 1
        idx_b = 1 - idx_a
        self.assertIs(params[idx_a], self.fs1.a.p.data)
        self.assertIs(grads[idx_a], self.fs1.a.p.grad)
        self.assertIs(params[idx_b], self.fs2.b.p.data)
        self.assertIs(grads[idx_b], self.fs2.b.p.grad)

    def test_pickle_cpu(self):
        fs2_serialized = pickle.dumps(self.fs2)
        fs2_loaded = pickle.loads(fs2_serialized)
        self.assertTrue((self.fs2.b.p.data == fs2_loaded.b.p.data).all())
        self.assertTrue(
            (self.fs2.fs1.a.p.data == fs2_loaded.fs1.a.p.data).all())

    @attr.gpu
    def test_pickle_gpu(self):
        self.fs2.to_gpu()
        fs2_serialized = pickle.dumps(self.fs2)
        fs2_loaded = pickle.loads(fs2_serialized)
        fs2_loaded.to_cpu()
        self.fs2.to_cpu()

        self.assertTrue((self.fs2.b.p.data == fs2_loaded.b.p.data).all())
        self.assertTrue(
            (self.fs2.fs1.a.p.data == fs2_loaded.fs1.a.p.data).all())

    def check_getter(self, fs, gpu, attribute):
        params = getattr(fs, attribute)
        self.assertEqual(len(params), 2)
        if gpu:
            params = tuple(p.get() for p in params)
        if attribute == 'parameters':
            np.testing.assert_equal(params[0], self.p_b)
            np.testing.assert_equal(params[1], self.p_a)
        elif attribute == 'gradients':
            np.testing.assert_equal(params[0], self.gp_b)
            np.testing.assert_equal(params[1], self.gp_a)
        else:
            raise ValueError(
                'attribute should be parameters or gradients')

    def test_parameters_getter_cpu(self):
        self.check_getter(self.fs2, False, 'parameters')

    @attr.gpu
    def test_parameters_getter_gpu(self):
        self.fs2.to_gpu()
        self.check_getter(self.fs2, True, 'parameters')

    def test_parameters_setter_cpu(self):
        self._check_setter(self.fs2, False, 'parameters')

    @attr.gpu
    def test_parameters_setter_gpu(self):
        self.fs2.to_gpu()
        self._check_setter(self.fs2, True, 'parameters')

    def test_parameters_setter_invalid_cpu(self):
        self._check_setter_invalid(self.fs2, 1, np, 'parameters')

    @attr.gpu
    def test_parameters_setter_invalid_gpu(self):
        self.fs2.to_gpu()
        self._check_setter_invalid(self.fs2, 1, cuda.cupy, 'parameters')

    def test_parameters_setter_invalid_2_cpu(self):
        self._check_setter_invalid(self.fs2, -1, np, 'parameters')

    @attr.gpu
    def test_parameters_setter_invalid_2_gpu(self):
        self.fs2.to_gpu()
        self._check_setter_invalid(self.fs2, -1, cuda.cupy, 'parameters')

    def test_gradients_getter_cpu(self):
        self.check_getter(self.fs2, False, 'gradients')

    @attr.gpu
    def test_gradients_getter_gpu(self):
        self.fs2.to_gpu()
        self.check_getter(self.fs2, True, 'gradients')

    def test_gradients_setter_cpu(self):
        self._check_setter(self.fs2, False, 'gradients')

    @attr.gpu
    def test_gradients_setter_gpu(self):
        self.fs2.to_gpu()
        self._check_setter(self.fs2, True, 'gradients')

    def test_gradients_setter_invalid_cpu(self):
        self._check_setter_invalid(self.fs2, 1, np, 'gradients')

    @attr.gpu
    def test_gradients_setter_invalid_gpu(self):
        self.fs2.to_gpu()
        self._check_setter_invalid(self.fs2, 1, cuda.cupy, 'gradients')

    def test_gradients_setter_invalid_2_cpu(self):
        self._check_setter_invalid(self.fs2, -1, np, 'gradients')

    @attr.gpu
    def test_gradients_setter_invalid_2_gpu(self):
        self.fs2.to_gpu()
        self._check_setter_invalid(self.fs2, -1, cuda.cupy, 'gradients')


class TestFunctionSet(TestFunctionSetBase):

    def setUp(self):
        self.fs = chainer.FunctionSet(
            a=L.Linear(3, 2),
            b=L.Linear(3, 2)
        )
        self.aW = self.fs.a.W
        self.ab = self.fs.a.b
        self.bW = self.fs.b.W
        self.bb = self.fs.b.b

        self.agW = self.fs.a.gW
        self.agb = self.fs.a.gb
        self.bgW = self.fs.b.gW
        self.bgb = self.fs.b.gb

    def check_equal_fs(self, fs1, fs2):
        self.assertTrue((fs1.a.W.data == fs2.a.W.data).all())
        self.assertTrue((fs1.a.b.data == fs2.a.b.data).all())
        self.assertTrue((fs1.b.W.data == fs2.b.W.data).all())
        self.assertTrue((fs1.b.b.data == fs2.b.b.data).all())

    def test_pickle_cpu(self):
        s = pickle.dumps(self.fs)
        fs2 = pickle.loads(s)
        self.check_equal_fs(self.fs, fs2)

    @attr.gpu
    def test_pickle_gpu(self):
        self.fs.to_gpu()
        s = pickle.dumps(self.fs)
        fs2 = pickle.loads(s)

        self.fs.to_cpu()
        fs2.to_cpu()
        self.check_equal_fs(self.fs, fs2)

    def check_copy_parameters_from(self, src_id, dst_id):
        aW = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)
        ab = np.random.uniform(-1, 1, (2,)).astype(np.float32)
        bW = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)
        bb = np.random.uniform(-1, 1, (2,)).astype(np.float32)
        params = [aW.copy(), ab.copy(), bW.copy(), bb.copy()]

        if dst_id >= 0:
            self.fs.to_gpu(dst_id)

        if src_id >= 0:
            params = [cuda.to_gpu(p, src_id) for p in params]

        self.fs.copy_parameters_from(params)
        self.fs.to_cpu()

        self.assertTrue((self.fs.a.W.data == aW).all())
        self.assertTrue((self.fs.a.b.data == ab).all())
        self.assertTrue((self.fs.b.W.data == bW).all())
        self.assertTrue((self.fs.b.b.data == bb).all())

    def test_copy_parameters_from_cpu_to_cpu(self):
        self.check_copy_parameters_from(-1, -1)

    @attr.gpu
    def test_copy_parameters_from_cpu_to_gpu(self):
        self.check_copy_parameters_from(-1, cuda.Device().id)

    @attr.gpu
    def test_copy_parameters_from_gpu_to_cpu(self):
        self.check_copy_parameters_from(cuda.Device().id, -1)

    @attr.gpu
    def test_copy_parameters_from_gpu_to_gpu(self):
        device_id = cuda.Device().id
        self.check_copy_parameters_from(device_id, device_id)

    @attr.multi_gpu(2)
    def test_copy_parameters_from_multigpu(self):
        self.check_copy_parameters_from(0, 1)

    def test_getitem(self):
        self.assertIs(self.fs['a'], self.fs.a)

    def test_getitem_notfoud(self):
        with self.assertRaises(AttributeError):
            self.fs['not_found']

    def check_getter(self, fs, gpu, attribute):
        params = getattr(fs, attribute)
        self.assertEqual(len(params), 4)
        if gpu:
            params = tuple(p.get() for p in params)

        if attribute == 'parameters':
            np.testing.assert_array_equal(params[0], self.aW)
            np.testing.assert_array_equal(params[1], self.ab)
            np.testing.assert_array_equal(params[2], self.bW)
            np.testing.assert_array_equal(params[3], self.bb)
        elif attribute == 'gradients':
            np.testing.assert_array_equal(params[0], self.agW)
            np.testing.assert_array_equal(params[1], self.agb)
            np.testing.assert_array_equal(params[2], self.bgW)
            np.testing.assert_array_equal(params[3], self.bgb)

    def test_parameters_getter_cpu(self):
        self.check_getter(self.fs, False, 'parameters')

    @attr.gpu
    def test_parameters_getter_gpu(self):
        self.fs.to_gpu()
        self.check_getter(self.fs, True, 'parameters')

    def test_parameters_setter_cpu(self):
        self._check_setter(self.fs, False, 'parameters')

    @attr.gpu
    def test_parameters_setter_gpu(self):
        self.fs.to_gpu()
        self._check_setter(self.fs, True, 'parameters')

    def test_parameters_setter_invalid_cpu(self):
        self._check_setter_invalid(self.fs, 1, np, 'parameters')

    @attr.gpu
    def test_parameters_setter_invalid_gpu(self):
        self._check_setter_invalid(self.fs, 1, cuda.cupy, 'parameters')

    def test_parameters_setter_invalid_2_cpu(self):
        self._check_setter_invalid(self.fs, -1, np, 'parameters')

    @attr.gpu
    def test_parameters_setter_invalid_2_gpu(self):
        self._check_setter_invalid(self.fs, -1, cuda.cupy, 'parameters')

    def test_gradients_getter_cpu(self):
        self.check_getter(self.fs, False, 'gradients')

    @attr.gpu
    def test_gradients_getter_gpu(self):
        self.fs.to_gpu()
        self.check_getter(self.fs, True, 'gradients')

    def test_gradients_setter_cpu(self):
        self._check_setter(self.fs, False, 'gradients')

    @attr.gpu
    def test_gradients_setter_gpu(self):
        self.fs.to_gpu()
        self._check_setter(self.fs, True, 'gradients')

    def test_gradients_setter_invalid_cpu(self):
        self._check_setter_invalid(self.fs, 1, np, 'gradients')

    @attr.gpu
    def test_gradients_setter_invalid_gpu(self):
        self._check_setter_invalid(self.fs, 1, cuda.cupy, 'gradients')

    def test_gradients_setter_invalid_2_cpu(self):
        self._check_setter_invalid(self.fs, -1, np, 'gradients')

    @attr.gpu
    def test_gradients_setter_invalid_2_gpu(self):
        self._check_setter_invalid(self.fs, -1, cuda.cupy, 'gradients')

testing.run_module(__name__, __file__)
