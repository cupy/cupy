import unittest

import numpy as np
import six
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import testing
from chainer.testing import attr


if cuda.available:
    cuda.init()


class MockFunction(chainer.Function):

    def __init__(self, shape):
        self.p = np.zeros(shape).astype(np.float32)
        self.gp = np.ones(shape).astype(np.float32)

    parameter_names = ('p', )
    gradient_names = ('gp', )


class TestNestedFunctionSet(unittest.TestCase):

    def setUp(self):
        self.fs1 = chainer.FunctionSet(
            a=MockFunction((1, 2)))
        self.fs2 = chainer.FunctionSet(
            fs1=self.fs1,
            b=MockFunction((3, 4)))

    def test_get_sorted_funcs(self):
        six.assertCountEqual(
            self, [k for (k, v) in self.fs2._get_sorted_funcs()], ('b', 'fs1'))

    def test_collect_parameters(self):
        p_b = np.zeros((3, 4)).astype(np.float32)
        p_a = np.zeros((1, 2)).astype(np.float32)
        gp_b = np.ones((3, 4)).astype(np.float32)
        gp_a = np.ones((1, 2)).astype(np.float32)

        actual = (self.fs2.parameters, self.fs2.gradients)
        self.assertTrue(list(map(len, actual)) == [2, 2])
        self.assertTrue((actual[0][0] == p_b).all())
        self.assertTrue((actual[0][1] == p_a).all())
        self.assertTrue((actual[1][0] == gp_b).all())
        self.assertTrue((actual[1][1] == gp_a).all())

    def test_pickle_cpu(self):
        fs2_serialized = pickle.dumps(self.fs2)
        fs2_loaded = pickle.loads(fs2_serialized)
        self.assertTrue((self.fs2.b.p == fs2_loaded.b.p).all())
        self.assertTrue((self.fs2.fs1.a.p == fs2_loaded.fs1.a.p).all())

    @attr.gpu
    def test_pickle_gpu(self):
        self.fs2.to_gpu()
        fs2_serialized = pickle.dumps(self.fs2)
        fs2_loaded = pickle.loads(fs2_serialized)
        fs2_loaded.to_cpu()
        self.fs2.to_cpu()

        self.assertTrue((self.fs2.b.p == fs2_loaded.b.p).all())
        self.assertTrue((self.fs2.fs1.a.p == fs2_loaded.fs1.a.p).all())


class TestFunctionSet(unittest.TestCase):

    def setUp(self):
        self.fs = chainer.FunctionSet(
            a=F.Linear(3, 2),
            b=F.Linear(3, 2)
        )

    def test_get_sorted_funcs(self):
        six.assertCountEqual(
            self, [k for (k, v) in self.fs._get_sorted_funcs()], ('a', 'b'))

    def check_equal_fs(self, fs1, fs2):
        self.assertTrue((fs1.a.W == fs2.a.W).all())
        self.assertTrue((fs1.a.b == fs2.a.b).all())
        self.assertTrue((fs1.b.W == fs2.b.W).all())
        self.assertTrue((fs1.b.b == fs2.b.b).all())

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

    def check_copy_parameters_from(self, src_gpu=False, dst_gpu=False):
        aW = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)
        ab = np.random.uniform(-1, 1, (2,)).astype(np.float32)
        bW = np.random.uniform(-1, 1, (2, 3)).astype(np.float32)
        bb = np.random.uniform(-1, 1, (2,)).astype(np.float32)
        params = [aW.copy(), ab.copy(), bW.copy(), bb.copy()]

        if dst_gpu:
            self.fs.to_gpu()

        if src_gpu:
            params = map(cuda.to_gpu, params)

        self.fs.copy_parameters_from(params)
        self.fs.to_cpu()

        self.assertTrue((self.fs.a.W == aW).all())
        self.assertTrue((self.fs.a.b == ab).all())
        self.assertTrue((self.fs.b.W == bW).all())
        self.assertTrue((self.fs.b.b == bb).all())

    def test_copy_parameters_from_cpu_to_cpu(self):
        self.check_copy_parameters_from(False, False)

    @attr.gpu
    def test_copy_parameters_from_cpu_to_gpu(self):
        self.check_copy_parameters_from(False, True)

    @attr.gpu
    def test_copy_parameters_from_gpu_to_cpu(self):
        self.check_copy_parameters_from(True, False)

    @attr.gpu
    def test_copy_parameters_from_gpu_to_gpu(self):
        self.check_copy_parameters_from(True, True)

    def test_getitem(self):
        self.assertIs(self.fs['a'], self.fs.a)

    def test_getitem_notfoud(self):
        with self.assertRaises(AttributeError):
            self.fs['not_found']


testing.run_module(__name__, __file__)
