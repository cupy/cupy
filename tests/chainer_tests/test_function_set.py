import unittest
import warnings

import numpy as np
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

    def setUp(self):
        # FunctionSet is deprecated. To suppress warnings, we ignore
        # DeprecationWarning.
        self.warn = warnings.catch_warnings()
        self.warn.__enter__()
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    def tearDown(self):
        self.warn.__exit__()

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

    def _assert_all_is(self, a1, a2):
        self.assertEqual(sorted(map(id, a1)), sorted(map(id, a2)))


class TestNestedFunctionSet(TestFunctionSetBase):

    def setUp(self):
        super(TestNestedFunctionSet, self).setUp()
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
        self._assert_all_is(params, [self.fs1.a.p.data, self.fs2.b.p.data])
        self._assert_all_is(grads, [self.fs1.a.p.grad, self.fs2.b.p.grad])

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

    def check_getter(self, fs, attribute):
        params = getattr(fs, attribute)
        self.assertEqual(len(params), 2)
        if attribute == 'parameters':
            self._assert_all_is(params, [self.fs1.a.p.data, self.fs2.b.p.data])
        elif attribute == 'gradients':
            self._assert_all_is(params, [self.fs1.a.p.grad, self.fs2.b.p.grad])
        else:
            raise ValueError(
                'attribute should be parameters or gradients')

    def test_parameters_getter_cpu(self):
        self.check_getter(self.fs2, 'parameters')

    @attr.gpu
    def test_parameters_getter_gpu(self):
        self.fs2.to_gpu()
        self.check_getter(self.fs2, 'parameters')

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
        self.check_getter(self.fs2, 'gradients')

    @attr.gpu
    def test_gradients_getter_gpu(self):
        self.fs2.to_gpu()
        self.check_getter(self.fs2, 'gradients')

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
        super(TestFunctionSet, self).setUp()
        self.fs = chainer.FunctionSet(
            a=L.Linear(3, 2),
            b=L.Linear(3, 2)
        )
        self.aW = self.fs.a.W.data
        self.ab = self.fs.a.b.data
        self.bW = self.fs.b.W.data
        self.bb = self.fs.b.b.data

        self.agW = self.fs.a.W.grad
        self.agb = self.fs.a.b.grad
        self.bgW = self.fs.b.W.grad
        self.bgb = self.fs.b.b.grad

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

        fs = self.fs.copy()
        fs.a.W.data = aW
        fs.a.b.data = ab
        fs.b.W.data = bW
        fs.b.b.data = bb

        if src_id >= 0:
            fs.to_gpu(src_id)
        if dst_id >= 0:
            self.fs.to_gpu(dst_id)

        self.fs.copy_parameters_from(fs.parameters)
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

    def check_getter(self, fs, attribute):
        params = getattr(fs, attribute)
        self.assertEqual(len(params), 4)
        if attribute == 'parameters':
            self._assert_all_is(
                params,
                [self.fs.a.W.data, self.fs.a.b.data,
                 self.fs.b.W.data, self.fs.b.b.data])
        elif attribute == 'gradients':
            self._assert_all_is(
                params,
                [self.fs.a.W.grad, self.fs.a.b.grad,
                 self.fs.b.W.grad, self.fs.b.b.grad])

    def test_parameters_getter_cpu(self):
        self.check_getter(self.fs, 'parameters')

    @attr.gpu
    def test_parameters_getter_gpu(self):
        self.fs.to_gpu()
        self.check_getter(self.fs, 'parameters')

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
        self.check_getter(self.fs, 'gradients')

    @attr.gpu
    def test_gradients_getter_gpu(self):
        self.fs.to_gpu()
        self.check_getter(self.fs, 'gradients')

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
