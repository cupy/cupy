import unittest

import mock
import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


if cuda.available:
    cuda.init()


class TestFunction(unittest.TestCase):

    def _get_method(self, prefix, gpu):
        suffix = 'gpu' if gpu else 'cpu'
        return getattr(self.f, prefix + '_' + suffix)

    def setUp(self):
        y1 = numpy.arange(4).astype(numpy.float32)
        y2 = numpy.arange(4).astype(numpy.float32) + 1
        gx1 = numpy.arange(3).astype(numpy.float32)
        gx2 = None
        gy1 = numpy.arange(4).astype(numpy.float32)
        gy2 = numpy.arange(4).astype(numpy.float32)

        f = chainer.Function()
        f.check_type_forward = mock.MagicMock()
        f.forward_cpu = mock.MagicMock(return_value=(y1, y2))
        f.forward_gpu = mock.MagicMock()
        f.backward_cpu = mock.MagicMock(return_value=(gx1, gx2))
        f.backward_gpu = mock.MagicMock()
        self.f = f

        self.x1 = numpy.arange(3).astype(numpy.float32)
        self.x2 = numpy.arange(3).astype(numpy.int32)
        self.y1 = y1
        self.y2 = y2
        self.gx1 = gx1
        self.gx2 = gx2
        self.gy1 = gy1
        self.gy2 = gy2

    def tearDown(self):
        # Set None to delete cuda array
        self.f = None
        self.y1 = None
        self.y2 = None
        self.gx1 = None

    def setup_gpu(self):
        self.x1 = cuda.to_gpu(self.x1)
        self.x2 = cuda.to_gpu(self.x2)
        self.y1 = cuda.to_gpu(self.y1)
        self.y2 = cuda.to_gpu(self.y2)
        self.gx1 = cuda.to_gpu(self.gx1)
        self.gx2 = None
        self.gy1 = cuda.to_gpu(self.gy1)
        self.gy2 = cuda.to_gpu(self.gy2)
        self.f.forward_gpu = mock.MagicMock(return_value=(self.y1, self.y2))
        self.f.backward_gpu = mock.MagicMock(return_value=(self.gx1, self.gx2))

    def check_forward(self, gpu):
        y1, y2 = self.f.forward((self.x1, self.x2))
        self.assertEqual(self.f.check_type_forward.call_count, 0)
        self.assertEqual(self._get_method('forward', not gpu).call_count, 0)
        self._get_method('forward', gpu).assert_called_once_with(
            (self.x1, self.x2))
        self.assertTrue((cuda.to_cpu(y1) == cuda.to_cpu(self.y1)).all())
        self.assertTrue((cuda.to_cpu(y2) == cuda.to_cpu(self.y2)).all())

    def test_forward_cpu(self):
        self.check_forward(False)

    @attr.gpu
    def test_forward_gpu(self):
        self.setup_gpu()
        self.check_forward(True)

    def check_backward(self, gpu):
        gx1, gx2 = self.f.backward((self.x1, self.x2), (self.gy1, self.gy2))
        self.assertEqual(self._get_method('backward', not gpu).call_count, 0)
        self._get_method('backward', gpu).assert_called_once_with(
            (self.x1, self.x2), (self.gy1, self.gy2))
        self.assertTrue((cuda.to_cpu(gx1) == cuda.to_cpu(self.gx1)).all())
        self.assertIsNone(gx2)

    def test_backward_cpu(self):
        self.check_backward(False)

    @attr.gpu
    def test_backward_gpu(self):
        self.setup_gpu()
        self.check_backward(True)

    def check_check_type_forward(self):
        self.assertEqual(self.f.check_type_forward.call_count, 1)
        ts = self.f.check_type_forward.call_args[0][0]
        self.assertIsInstance(ts, type_check.TypeInfoTuple)
        self.assertEqual(len(ts), 2)

        self.assertEqual(ts[0].name, 'in_types[0]')
        t1 = ts[0].eval()
        self.assertEqual(t1.shape, (3,))
        self.assertEqual(t1.dtype, numpy.float32)

        self.assertEqual(ts[1].name, 'in_types[1]')
        t2 = ts[1].eval()
        self.assertEqual(t2.shape, (3,))
        self.assertEqual(t2.dtype, numpy.int32)

    def check_call(self):
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        x1.rank = 1
        x2.rank = 3
        ys = self.f(x1, x2)

        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()

        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            # rank is (maximum rank in xs) + 2, since Function call
            # automatically inserts Split function.
            self.assertEqual(y.rank, 5)
            self.assertFalse(y.volatile)
            # __call__ method makes a copy
            self.assertIsNot(y.creator, self.f)

        self.assertIsNone(self.f.outputs)
        self.assertIsInstance(y.creator.outputs, tuple)

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.setup_gpu()
        self.check_call()

    def check_call_volatile(self):
        x1 = chainer.Variable(self.x1, volatile=True)
        x2 = chainer.Variable(self.x2, volatile=True)
        x1.rank = 1
        x2.rank = 3
        ys = self.f(x1, x2)

        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()

        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            self.assertEqual(y.rank, 0)
            self.assertTrue(y.volatile)
            self.assertIsNone(y.creator)

        self.assertIsNone(self.f.outputs)

    def test_call_volatile_cpu(self):
        self.check_call_volatile()

    @attr.gpu
    def test_call_volatile_gpu(self):
        self.setup_gpu()
        self.check_call_volatile()

    def check_call_single_return_value(self, volatile):
        x1 = chainer.Variable(self.x1, volatile=volatile)
        x2 = chainer.Variable(self.x2, volatile=volatile)
        ret = self.f(x1, x2)
        self.assertIsInstance(ret, chainer.Variable)

    def test_call_sigle_return_value_cpu(self):
        self.f.forward_cpu.return_value = (cuda.to_cpu(self.y1),)
        self.check_call_single_return_value(False)

    @attr.gpu
    def test_call_sigle_return_value_gpu(self):
        self.setup_gpu()
        self.f.forward_gpu.return_value = (cuda.to_gpu(self.y1),)
        self.check_call_single_return_value(False)

    def test_call_sigle_return_value_volatile_cpu(self):
        self.f.forward_cpu.return_value = (cuda.to_cpu(self.y1),)
        self.check_call_single_return_value(True)

    @attr.gpu
    def test_call_sigle_return_value_volatile_gpu(self):
        self.setup_gpu()
        self.f.forward_gpu.return_value = (cuda.to_gpu(self.y1),)
        self.check_call_single_return_value(True)

    def check_call_mixed_volatile(self):
        x1 = chainer.Variable(self.x1, volatile=True)
        x2 = chainer.Variable(self.x2, volatile=False)
        with self.assertRaises(AssertionError):
            self.f(x1, x2)

    def test_call_mixed_volatile_cpu(self):
        self.check_call_mixed_volatile()

    @attr.gpu
    def test_call_mixed_volatile_gpu(self):
        self.setup_gpu()
        self.check_call_mixed_volatile()

    def _get_f(self):
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        y1, y2 = self.f(x1, x2)

        f = y1.creator
        # To test weak refernece, return only x1 and y1.
        # x2 and y2 are deleted by the garbage collector
        return f, x1, y1

    def test_unchain(self):
        f, _x1, _y1 = self._get_f()
        f.unchain()

        y1, y2 = f.outputs
        # As _y1 is alive, this weak ref is also alive
        y1_ref = y1()
        self.assertTrue(y1_ref is not None and y1_ref.creator is None)
        # This weak ref is dead by unchain
        y2_ref = y2()
        self.assertTrue(y2_ref is None)

        self.assertIsNone(f.inputs)

    def test_parameters_getter(self):
        self.assertEqual(self.f.parameters, ())

    def test_gradients_getter(self):
        self.assertEqual(self.f.gradients, ())

    def test_label(self):
        self.assertEqual(self.f.label, 'Function')


class TestParameterizedFunction(unittest.TestCase):

    def setUp(self):
        f = chainer.Function()
        f.p1 = numpy.arange(10)
        f.p2 = numpy.arange(5)
        f.g1 = numpy.arange(8)
        f.g2 = numpy.arange(3)
        f.parameter_names = ('p1', 'p2')
        f.gradient_names = ('g1', 'g2')
        self.f = f

    @attr.gpu
    def test_to_gpu(self):
        self.f.to_gpu()
        self.assertIsInstance(self.f.p1, cuda.GPUArray)
        self.assertIsInstance(self.f.p2, cuda.GPUArray)

    @attr.gpu
    def test_to_cpu(self):
        self.f.to_gpu()
        self.f.to_cpu()
        self.assertIsInstance(self.f.p1, numpy.ndarray)
        self.assertIsInstance(self.f.p2, numpy.ndarray)

    def test_parameters_getter(self):
        ps = self.f.parameters
        self.assertIsInstance(ps, tuple)
        self.assertEqual(len(ps), 2)

    def test_parameters_setter(self):
        p1 = numpy.arange(10) + 1
        p2 = numpy.arange(5) + 1
        self.f.parameters = (p1, p2)
        q1, q2 = self.f.parameters
        self.assertIs(p1, q1)
        self.assertIs(p2, q2)

    def test_parameters_setter_invalid_size(self):
        p1 = numpy.arange(10) + 1
        with self.assertRaises(AssertionError):
            self.f.parameters = (p1,)

    def test_gradients_getter(self):
        gs = self.f.gradients
        self.assertIsInstance(gs, tuple)
        self.assertEqual(len(gs), 2)

    def test_gradients_setter(self):
        g1 = numpy.arange(8) + 1
        g2 = numpy.arange(4) + 1
        self.f.gradients = (g1, g2)
        h1, h2 = self.f.gradients
        self.assertIs(g1, h1)
        self.assertIs(g2, h2)

    def test_gradients_setter_invalid_size(self):
        g1 = numpy.arange(8) + 1
        with self.assertRaises(AssertionError):
            self.f.gradients = (g1,)


class TestFunctionBackwardIntegration(unittest.TestCase):

    def test_backward(self):
        x = chainer.Variable(numpy.array([1]))
        y1 = F.identity(x)
        y2 = F.identity(x)
        z = y1 + y2

        z.grad = numpy.array([1])
        z.backward(retain_grad=True)

        self.assertEqual(y1.grad[0], 1)
        self.assertEqual(y2.grad[0], 1)
        self.assertEqual(x.grad[0], 2)


class TestSplit(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.g1 = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.g2 = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)

    def _make_split(self, x):
        v = chainer.Variable(x)
        v.rank = 1
        return chainer.function.Split(v)

    def check_init(self, x):
        split = self._make_split(x)
        self.assertEqual(split.rank, 1)

    def test_init_cpu(self):
        self.check_init(self.x)

    @attr.gpu
    def test_init_gpu(self):
        self.check_init(cuda.to_gpu(self.x))

    def check_add_branch(self, x):
        split = self._make_split(x)

        out = split.add_branch()
        self.assertIsInstance(out, chainer.Variable)
        self.assertIs(out.creator, split)
        self.assertEqual(len(split.outputs), 1)

    def test_add_branch_cpu(self):
        self.check_add_branch(self.x)

    @attr.gpu
    def test_add_branch_gpu(self):
        self.check_add_branch(cuda.to_gpu(self.x))

    def check_backward(self, x, g1, g2):
        split = self._make_split(x)

        grads = (g1, g2, None)
        gx, = split.backward((x,), grads)
        gradient_check.assert_allclose(g1 + g2, gx)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.g1, self.g2)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.g1),
                            cuda.to_gpu(self.g2))

    def check_backward_one(self, x, g1):
        split = self._make_split(x)

        grads = (g1,)
        gx, = split.backward((x,), grads)
        # Note that when only one argument is given, its return value
        # is a grad itself, and not a copy of it.
        self.assertIs(g1, gx)

    def test_backward_one_cpu(self):
        self.check_backward_one(self.x, self.g1)

    @attr.gpu
    def test_backward_one_gpu(self):
        self.check_backward_one(cuda.to_gpu(self.x),
                                cuda.to_cpu(self.g1))


testing.run_module(__name__, __file__)
