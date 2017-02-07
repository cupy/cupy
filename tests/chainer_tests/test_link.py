import unittest

import mock
import numpy

import chainer
from chainer import cuda
import chainer.serializer
from chainer import testing
from chainer.testing import attr


class TestLink(unittest.TestCase):

    def setUp(self):
        x_shape_0 = 2
        x_shape_1 = numpy.int64(3)
        self.link = chainer.Link(x=((x_shape_0, x_shape_1), 'd'), y=2)
        self.p = numpy.array([1, 2, 3], dtype='f')
        self.link.add_persistent('p', self.p)
        self.link.name = 'a'

    def check_param_init(self, name, shape, dtype):
        self.assertTrue(hasattr(self.link, name))
        var = getattr(self.link, name)
        self.assertEqual(var.name, name)
        self.assertIsInstance(var, chainer.Variable)
        self.assertEqual(var.data.shape, shape)
        self.assertEqual(var.data.dtype, dtype)
        self.assertTrue(numpy.all(numpy.isnan(var.data)))
        self.assertEqual(var.grad.shape, shape)
        self.assertEqual(var.grad.dtype, dtype)
        self.assertTrue(numpy.all(numpy.isnan(var.grad)))

    def test_init(self):
        self.check_param_init('x', (2, 3), 'd')
        self.check_param_init('y', (2,), 'f')

    def test_add_param(self):
        self.link.add_param('z', (2, 3))
        self.check_param_init('z', (2, 3), 'f')

        self.link.add_param('w', (2, 3), dtype='d')
        self.check_param_init('w', (2, 3), 'd')

    def test_add_persistent(self):
        self.assertTrue(hasattr(self.link, 'p'))
        self.assertIs(self.link.p, self.p)

        self.link.add_persistent('q', 'abc')
        self.assertTrue(hasattr(self.link, 'q'))
        self.assertEqual(self.link.q, 'abc')

    def test_copy(self):
        link = self.link.copy()
        self.assertTrue(hasattr(link, 'x'))
        self.assertTrue(hasattr(link, 'y'))
        self.assertTrue(hasattr(link, 'p'))
        self.assertIsNot(link.x, self.link.x)
        self.assertIs(link.x.data, self.link.x.data)
        self.assertIsNot(link.y, self.link.y)
        self.assertIs(link.y.data, self.link.y.data)
        self.assertIs(link.p, self.link.p)
        self.assertIs(link.name, None)

    def test_to_cpu_on_cpu(self):
        x = self.link.x.data
        gx = self.link.x.grad
        y = self.link.y.data
        gy = self.link.y.grad
        p = self.link.p
        self.link.to_cpu()
        self.assertIs(self.link.x.data, x)
        self.assertIs(self.link.x.grad, gx)
        self.assertIs(self.link.y.data, y)
        self.assertIs(self.link.y.grad, gy)
        self.assertIs(self.link.p, p)

    @attr.gpu
    def test_to_cpu(self):
        self.link.to_gpu()
        self.link.to_cpu()
        self.assertIs(self.link.xp, numpy)
        self.assertIsInstance(self.link.x.data, numpy.ndarray)
        self.assertIsInstance(self.link.x.grad, numpy.ndarray)
        self.assertIsInstance(self.link.y.data, numpy.ndarray)
        self.assertIsInstance(self.link.y.grad, numpy.ndarray)
        self.assertIsInstance(self.link.p, numpy.ndarray)

    @attr.gpu
    def test_to_gpu(self):
        cupy = cuda.cupy
        self.link.to_gpu()
        self.assertIs(self.link.xp, cupy)
        self.assertIsInstance(self.link.x.data, cupy.ndarray)
        self.assertIsInstance(self.link.x.grad, cupy.ndarray)
        self.assertIsInstance(self.link.y.data, cupy.ndarray)
        self.assertIsInstance(self.link.y.grad, cupy.ndarray)
        self.assertIsInstance(self.link.p, cupy.ndarray)

    def test_params(self):
        params = list(self.link.params())
        self.assertEqual({id(p) for p in params},
                         {id(self.link.x), id(self.link.y)})

    def test_namedparams(self):
        namedparams = list(self.link.namedparams())
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/x', id(self.link.x)), ('/y', id(self.link.y))})

    def test_links(self):
        links = list(self.link.links())
        self.assertIs(links[0], self.link)

    def test_links_skipself(self):
        links = list(self.link.links(skipself=True))
        self.assertFalse(links)  # empty

    def test_namedlinks(self):
        pl = list(self.link.namedlinks())
        self.assertEqual(len(pl), 1)
        self.assertEqual(pl[0][0], '/')
        self.assertIs(pl[0][1], self.link)

    def test_copyparams(self):
        self.link.x.grad.fill(0)
        self.link.y.grad.fill(1)
        gx = self.link.x.grad.copy()
        gy = self.link.y.grad.copy()

        l = chainer.Link(x=(2, 3), y=2)
        l.x.data.fill(2)
        l.x.grad.fill(3)
        l.y.data.fill(4)
        l.y.grad.fill(5)
        self.link.copyparams(l)
        numpy.testing.assert_array_equal(self.link.x.data, l.x.data)
        numpy.testing.assert_array_equal(self.link.x.grad, gx)
        numpy.testing.assert_array_equal(self.link.y.data, l.y.data)
        numpy.testing.assert_array_equal(self.link.y.grad, gy)

    def test_copyparams_uninitialized(self):
        l = chainer.Link(x=(2, 3))
        l.add_uninitialized_param('y')
        self.link.x.data.fill(2)
        self.link.y.data.fill(4)
        l.copyparams(self.link)
        numpy.testing.assert_array_equal(l.x.data, self.link.x.data)
        self.assertTrue(hasattr(l, 'y'))
        numpy.testing.assert_array_equal(l.y.data, self.link.y.data)

    def test_cleargrads(self):
        self.link.cleargrads()
        self.assertIsNone(self.link.x.grad)
        self.assertIsNone(self.link.y.grad)

    def test_zerograds(self):
        gx_expect = numpy.zeros_like(self.link.x.data)
        gy_expect = numpy.zeros_like(self.link.y.data)
        self.link.zerograds()
        numpy.testing.assert_array_equal(self.link.x.grad, gx_expect)
        numpy.testing.assert_array_equal(self.link.y.grad, gy_expect)

    def test_addgrads(self):
        l = chainer.Link(x=(2, 3), y=2)
        l.x.grad.fill(1)
        l.y.grad.fill(2)

        self.link.x.grad.fill(-1)
        self.link.y.grad.fill(-2)

        self.link.addgrads(l)

        gx_expect = numpy.zeros_like(l.x.grad)
        gy_expect = numpy.zeros_like(l.y.grad)
        numpy.testing.assert_array_equal(self.link.x.grad, gx_expect)
        numpy.testing.assert_array_equal(self.link.y.grad, gy_expect)

    def test_serialize(self):
        serializer = mock.MagicMock(return_value=3)
        l = chainer.Link(x=(2, 3), y=2)
        l.add_persistent('z', 1)
        l.serialize(serializer)
        self.assertEqual(serializer.call_count, 3)
        serializer.assert_any_call('x', l.x.data)
        serializer.assert_any_call('y', l.y.data)
        serializer.assert_any_call('z', 1)
        self.assertEqual(l.z, 3)

    def test_serialize_param_shape_placeholder(self):
        serializer = mock.MagicMock(return_value=3)
        l = chainer.Link(y=2)
        l.add_uninitialized_param('x')
        l.add_param('x', (2, 3))
        l.add_persistent('z', 1)
        l.serialize(serializer)
        self.assertEqual(serializer.call_count, 3)
        serializer.assert_any_call('x', l.x.data)
        serializer.assert_any_call('y', l.y.data)
        serializer.assert_any_call('z', 1)
        self.assertEqual(l.z, 3)

    def test_serialize_uninitialized_param(self):
        class SerializerMock(chainer.serializer.Serializer):

            def __getitem__(self, key):
                pass

            def __call__(self, key, value):
                pass

        serializer = SerializerMock()
        l = chainer.Link()
        l.add_uninitialized_param('x')
        with self.assertRaises(ValueError):
            l.serialize(serializer)

    def test_duplicate_uninitialized_param(self):
        l = chainer.Link(y=2)
        l.add_uninitialized_param('x')
        with self.assertRaises(AttributeError):
            l.add_uninitialized_param('x')

    def test_uninitialized_param_already_param(self):
        l = chainer.Link(y=2)
        l.add_param('x', (2, 3))
        with self.assertRaises(AttributeError):
            l.add_uninitialized_param('x')

    def test_has_uninitialized_params(self):
        l = chainer.Link(y=2)
        self.assertFalse(l.has_uninitialized_params)
        l.add_uninitialized_param('x')
        self.assertTrue(l.has_uninitialized_params)
        l.add_param('x', (2, 3))
        self.assertFalse(l.has_uninitialized_params)


class CountVariable(chainer.Variable):

    def __init__(self, v):
        super(CountVariable, self).__init__(v.data, v.volatile, v.name)
        self.grad = v.grad
        self.count_to_cpu = 0
        self.count_to_gpu = 0
        self.count_zerograd = 0

    def to_cpu(self):
        self.count_to_cpu += 1
        super(CountVariable, self).to_cpu()

    def to_gpu(self, device=None):
        self.count_to_gpu += 1
        super(CountVariable, self).to_gpu(device)

    def zerograd(self):
        self.count_zerograd += 1
        super(CountVariable, self).zerograd()


class TestChain(unittest.TestCase):

    def setUp(self):
        self.l1 = chainer.Link(x=(2, 3))
        self.l2 = chainer.Link(x=2)
        self.l3 = chainer.Link(x=3)

        self.c1 = chainer.Chain(l1=self.l1)
        self.c1.add_link('l2', self.l2)
        self.c2 = chainer.Chain(c1=self.c1, l3=self.l3)

    def test_init(self):
        self.assertIs(self.c1.l1, self.l1)
        self.assertIs(self.c1['l1'], self.l1)
        self.assertEqual(self.l1.name, 'l1')

        self.assertIs(self.c2.c1, self.c1)
        self.assertIs(self.c2['c1'], self.c1)
        self.assertEqual(self.c1.name, 'c1')

        self.assertIs(self.c2.l3, self.l3)
        self.assertIs(self.c2['l3'], self.l3)
        self.assertEqual(self.l3.name, 'l3')

    def test_add_link(self):
        self.assertIs(self.c1.l2, self.l2)
        self.assertEqual(self.l2.name, 'l2')

    def test_copy(self):
        c2 = self.c2.copy()
        self.assertIs(c2.name, None)
        self.assertTrue(hasattr(c2, 'c1'))
        self.assertEqual(c2.c1.name, 'c1')
        self.assertIsNot(c2.c1, self.c1)
        self.assertEqual(c2.c1.l1.name, 'l1')
        self.assertIsNot(c2.c1.l1, self.l1)
        self.assertIsNot(c2.c1.l1.x, self.l1.x)
        self.assertIs(c2.c1.l1.x.data, self.l1.x.data)
        self.assertIs(c2.c1.l1.x.grad, None)
        self.assertIs(c2.name, None)

        self.assertTrue(hasattr(c2.c1, 'l2'))
        self.assertEqual(c2.c1.l2.name, 'l2')
        self.assertIsNot(c2.c1.l2, self.l2)
        self.assertIsNot(c2.c1.l2.x, self.l2.x)
        self.assertIs(c2.c1.l2.x.data, self.l2.x.data)
        self.assertIs(c2.c1.l2.x.grad, None)

        self.assertTrue(hasattr(c2, 'l3'))
        self.assertEqual(c2.l3.name, 'l3')
        self.assertIsNot(c2.l3, self.l3)
        self.assertIsNot(c2.l3.x, self.l3.x)
        self.assertIs(c2.l3.x.data, self.l3.x.data)
        self.assertIs(c2.l3.x.grad, None)

    def test_to_cpu_on_cpu(self):
        x1 = self.l1.x.data
        gx1 = self.l1.x.grad
        x2 = self.l2.x.data
        gx2 = self.l2.x.grad
        x3 = self.l3.x.data
        gx3 = self.l3.x.grad

        self.c2.to_cpu()
        self.assertIs(self.l1.x.data, x1)
        self.assertIs(self.l1.x.grad, gx1)
        self.assertIs(self.l2.x.data, x2)
        self.assertIs(self.l2.x.grad, gx2)
        self.assertIs(self.l3.x.data, x3)
        self.assertIs(self.l3.x.grad, gx3)

    def set_count_variables(self):
        self.l1.x = CountVariable(self.l1.x)
        self.l2.x = CountVariable(self.l2.x)
        self.l3.x = CountVariable(self.l3.x)

    @attr.gpu
    def test_to_cpu(self):
        self.set_count_variables()
        self.c2.to_gpu()
        self.c2.to_cpu()
        self.assertIs(self.c2.xp, numpy)
        self.assertIs(self.c1.xp, numpy)
        self.assertIs(self.l1.xp, numpy)
        self.assertIs(self.l2.xp, numpy)
        self.assertIs(self.l3.xp, numpy)
        self.assertIsInstance(self.l1.x.data, numpy.ndarray)
        self.assertIsInstance(self.l1.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l2.x.data, numpy.ndarray)
        self.assertIsInstance(self.l2.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l3.x.data, numpy.ndarray)
        self.assertIsInstance(self.l3.x.grad, numpy.ndarray)
        self.assertEqual(self.l1.x.count_to_cpu, 1)
        self.assertEqual(self.l1.x.count_to_gpu, 1)
        self.assertEqual(self.l2.x.count_to_cpu, 1)
        self.assertEqual(self.l2.x.count_to_gpu, 1)
        self.assertEqual(self.l3.x.count_to_cpu, 1)
        self.assertEqual(self.l3.x.count_to_gpu, 1)

    @attr.gpu
    def test_to_gpu(self):
        self.set_count_variables()
        cupy = cuda.cupy
        self.c2.to_gpu()
        self.assertIs(self.c2.xp, cupy)
        self.assertIs(self.c1.xp, cupy)
        self.assertIs(self.l1.xp, cupy)
        self.assertIs(self.l2.xp, cupy)
        self.assertIs(self.l3.xp, cupy)
        self.assertIsInstance(self.l1.x.data, cupy.ndarray)
        self.assertIsInstance(self.l1.x.grad, cupy.ndarray)
        self.assertIsInstance(self.l2.x.data, cupy.ndarray)
        self.assertIsInstance(self.l2.x.grad, cupy.ndarray)
        self.assertIsInstance(self.l3.x.data, cupy.ndarray)
        self.assertIsInstance(self.l3.x.grad, cupy.ndarray)
        self.assertEqual(self.l1.x.count_to_gpu, 1)
        self.assertEqual(self.l2.x.count_to_gpu, 1)
        self.assertEqual(self.l3.x.count_to_gpu, 1)

    def test_params(self):
        params = list(self.c2.params())
        self.assertEqual({id(p) for p in params},
                         {id(self.l1.x), id(self.l2.x), id(self.l3.x)})

    def test_namedparams(self):
        namedparams = list(self.c2.namedparams())
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/c1/l1/x', id(self.l1.x)),
                          ('/c1/l2/x', id(self.l2.x)),
                          ('/l3/x', id(self.l3.x))})

    def test_links(self):
        links = list(self.c2.links())
        self.assertEqual({id(l) for l in links},
                         {id(l) for l in [self.l1, self.l2, self.l3,
                                          self.c1, self.c2]})

    def test_links_skipself(self):
        links = list(self.c2.links(skipself=True))
        self.assertEqual({id(l) for l in links},
                         {id(l) for l in [self.l1, self.l2, self.l3, self.c1]})

    def test_namedlinks(self):
        namedlinks = list(self.c2.namedlinks())
        self.assertEqual({(name, id(l)) for name, l in namedlinks},
                         {('/', id(self.c2)),
                          ('/c1', id(self.c1)),
                          ('/c1/l1', id(self.l1)),
                          ('/c1/l2', id(self.l2)),
                          ('/l3', id(self.l3))})

    def test_namedlinks_skipself(self):
        namedlinks = list(self.c2.namedlinks(skipself=True))
        self.assertEqual({(name, id(l)) for name, l in namedlinks},
                         {('/c1', id(self.c1)),
                          ('/c1/l1', id(self.l1)),
                          ('/c1/l2', id(self.l2)),
                          ('/l3', id(self.l3))})

    def test_children(self):
        children = list(self.c2.children())
        self.assertEqual({id(c) for c in children}, {id(self.c1), id(self.l3)})

    def test_copyparams(self):
        l1 = chainer.Link(x=(2, 3))
        l2 = chainer.Link(x=2)
        l3 = chainer.Link(x=3)
        c1 = chainer.Chain(l1=l1, l2=l2)
        c2 = chainer.Chain(c1=c1, l3=l3)
        l1.x.data.fill(0)
        l2.x.data.fill(1)
        l3.x.data.fill(2)

        self.c2.copyparams(c2)

        numpy.testing.assert_array_equal(self.l1.x.data, l1.x.data)
        numpy.testing.assert_array_equal(self.l2.x.data, l2.x.data)
        numpy.testing.assert_array_equal(self.l3.x.data, l3.x.data)

    def test_zerograds(self):
        self.set_count_variables()
        self.c2.zerograds()
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.zeros(3))
        numpy.testing.assert_array_equal(self.l1.x.count_zerograd, 1)
        numpy.testing.assert_array_equal(self.l2.x.count_zerograd, 1)
        numpy.testing.assert_array_equal(self.l3.x.count_zerograd, 1)

    def test_addgrads(self):
        l1 = chainer.Link(x=(2, 3))
        l2 = chainer.Link(x=2)
        l3 = chainer.Link(x=3)
        c1 = chainer.Chain(l1=l1, l2=l2)
        c2 = chainer.Chain(c1=c1, l3=l3)
        l1.x.grad.fill(1)
        l2.x.grad.fill(2)
        l3.x.grad.fill(3)

        self.l1.x.grad.fill(-1)
        self.l2.x.grad.fill(-2)
        self.l3.x.grad.fill(-3)

        self.c2.addgrads(c2)
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.zeros(3))

    def test_serialize(self):
        mocks = {'l1': mock.MagicMock(), 'l2': mock.MagicMock()}
        serializer = mock.MagicMock()
        serializer.__getitem__.side_effect = lambda k: mocks[k]
        self.c1.serialize(serializer)

        self.assertEqual(serializer.call_count, 0)
        self.assertEqual(serializer.__getitem__.call_count, 2)
        serializer.__getitem__.assert_any_call('l1')
        serializer.__getitem__.assert_any_call('l2')

        mocks['l1'].assert_called_with('x', self.l1.x.data)
        mocks['l2'].assert_called_with('x', self.l2.x.data)


class TestChainList(unittest.TestCase):

    def setUp(self):
        self.l1 = chainer.Link(x=(2, 3))
        self.l1.add_uninitialized_param('y')
        self.l2 = chainer.Link(x=2)
        self.l3 = chainer.Link(x=3)
        self.c1 = chainer.ChainList(self.l1)
        self.c1.add_link(self.l2)
        self.c2 = chainer.ChainList(self.c1, self.l3)

    def test_init(self):
        self.assertIs(self.c1[0], self.l1)
        self.assertEqual(self.l1.name, '0')
        self.assertIs(self.c2[0], self.c1)
        self.assertEqual(self.c1.name, '0')
        self.assertIs(self.c2[1], self.l3)
        self.assertEqual(self.l3.name, '1')

    def test_add_link(self):
        self.assertIs(self.c1[1], self.l2)
        self.assertEqual(self.l2.name, '1')

    def test_iter(self):
        links = list(self.c2)
        self.assertEqual(2, len(links))
        self.assertIs(links[0], self.c1)
        self.assertIs(links[1], self.l3)

    def test_len(self):
        self.assertEqual(len(self.c1), 2)
        self.assertEqual(len(self.c2), 2)

    def test_copy(self):
        c2 = self.c2.copy()

        self.assertIs(c2.name, None)
        self.assertIsNot(c2[0], self.c1)
        self.assertEqual(c2[0].name, '0')
        self.assertIsNot(c2[0][0], self.l1)
        self.assertEqual(c2[0][0].name, '0')
        self.assertIsNot(c2[0][0].x, self.l1.x)
        self.assertIs(c2[0][0].x.data, self.l1.x.data)
        self.assertIs(c2[0][0].x.grad, None)

        self.assertIsNot(c2[0][1], self.l2)
        self.assertEqual(c2[0][1].name, '1')
        self.assertIsNot(c2[0][1].x, self.l2.x)
        self.assertIs(c2[0][1].x.data, self.l2.x.data)
        self.assertIs(c2[0][1].x.grad, None)

        self.assertIsNot(c2[1], self.l3)
        self.assertEqual(c2[1].name, '1')
        self.assertIsNot(c2[1].x, self.l3.x)
        self.assertIs(c2[1].x.data, self.l3.x.data)
        self.assertIs(c2[1].x.grad, None)

    @attr.gpu
    def test_copy_and_send_to_gpu(self):
        c2 = self.c2.copy()
        self.c2.to_gpu()
        self.assertIsInstance(self.c2[0][0].x.data, cuda.cupy.ndarray)
        self.assertIsInstance(self.c2[0][1].x.data, cuda.cupy.ndarray)
        self.assertIsInstance(c2[0][0].x.data, numpy.ndarray)
        self.assertIsInstance(c2[0][1].x.data, numpy.ndarray)

    @attr.gpu
    def test_copy_and_send_to_gpu_2(self):
        c2 = self.c2.copy()
        c2.to_gpu()
        self.assertIsInstance(self.c2[0][0].x.data, numpy.ndarray)
        self.assertIsInstance(self.c2[0][1].x.data, numpy.ndarray)
        self.assertIsInstance(c2[0][0].x.data, cuda.cupy.ndarray)
        self.assertIsInstance(c2[0][1].x.data, cuda.cupy.ndarray)

    @attr.multi_gpu(2)
    def test_copy_and_send_to_gpu_multi(self):
        c2 = self.c2.copy()
        self.c2.to_gpu(0)
        c2.to_gpu(1)
        self.assertEqual(self.c2[0][0].x.data.device.id, 0)
        self.assertEqual(self.c2[0][1].x.data.device.id, 0)
        self.assertEqual(c2[0][0].x.data.device.id, 1)
        self.assertEqual(c2[0][1].x.data.device.id, 1)

    def test_to_cpu_on_cpu(self):
        x1 = self.l1.x.data
        gx1 = self.l1.x.grad
        x2 = self.l2.x.data
        gx2 = self.l2.x.grad
        x3 = self.l3.x.data
        gx3 = self.l3.x.grad

        self.c2.to_cpu()

        self.assertIs(self.l1.x.data, x1)
        self.assertIs(self.l1.x.grad, gx1)
        self.assertIs(self.l2.x.data, x2)
        self.assertIs(self.l2.x.grad, gx2)
        self.assertIs(self.l3.x.data, x3)
        self.assertIs(self.l3.x.grad, gx3)

    @attr.gpu
    def test_to_cpu(self):
        self.c2.to_gpu()
        self.c2.to_cpu()
        self.assertIs(self.c2.xp, numpy)
        self.assertIs(self.c1.xp, numpy)
        self.assertIs(self.l1.xp, numpy)
        self.assertIs(self.l2.xp, numpy)
        self.assertIs(self.l3.xp, numpy)
        self.assertIsInstance(self.l1.x.data, numpy.ndarray)
        self.assertIsInstance(self.l1.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l2.x.data, numpy.ndarray)
        self.assertIsInstance(self.l2.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l3.x.data, numpy.ndarray)
        self.assertIsInstance(self.l3.x.grad, numpy.ndarray)

    @attr.gpu
    def test_to_gpu(self):
        cupy = cuda.cupy
        self.c2.to_gpu()
        self.assertIs(self.c2.xp, cupy)
        self.assertIs(self.c1.xp, cupy)
        self.assertIs(self.l1.xp, cupy)
        self.assertIs(self.l2.xp, cupy)
        self.assertIs(self.l3.xp, cupy)
        self.assertIsInstance(self.l1.x.data, cupy.ndarray)
        self.assertIsInstance(self.l1.x.grad, cupy.ndarray)
        self.assertIsInstance(self.l2.x.data, cupy.ndarray)
        self.assertIsInstance(self.l2.x.grad, cupy.ndarray)
        self.assertIsInstance(self.l3.x.data, cupy.ndarray)
        self.assertIsInstance(self.l3.x.grad, cupy.ndarray)

    def test_params(self):
        params = list(self.c2.params())
        self.assertEqual({id(p) for p in params},
                         {id(self.l1.x), id(self.l2.x), id(self.l3.x)})

    def test_namedparams(self):
        namedparams = list(self.c2.namedparams())
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/0/0/x', id(self.l1.x)),
                          ('/0/1/x', id(self.l2.x)),
                          ('/1/x', id(self.l3.x))})

    def test_links(self):
        links = list(self.c2.links())
        self.assertEqual({id(l) for l in links},
                         {id(l) for l in [self.l1, self.l2, self.l3,
                                          self.c1, self.c2]})

    def test_links_skipself(self):
        links = list(self.c2.links(skipself=True))
        self.assertEqual({id(l) for l in links},
                         {id(l) for l in [self.l1, self.l2, self.l3, self.c1]})

    def test_namedlinks(self):
        namedlinks = list(self.c2.namedlinks())
        self.assertEqual({(name, id(l)) for name, l in namedlinks},
                         {('/', id(self.c2)),
                          ('/0', id(self.c1)),
                          ('/0/0', id(self.l1)),
                          ('/0/1', id(self.l2)),
                          ('/1', id(self.l3))})

    def test_namedlinks_skipself(self):
        namedlinks = list(self.c2.namedlinks(skipself=True))
        self.assertEqual({(name, id(l)) for name, l in namedlinks},
                         {('/0', id(self.c1)),
                          ('/0/0', id(self.l1)),
                          ('/0/1', id(self.l2)),
                          ('/1', id(self.l3))})

    def test_children(self):
        self.assertEqual(tuple(id(c) for c in self.c2.children()),
                         (id(self.c1), id(self.l3)))

        self.assertEqual(tuple(id(c) for c in self.c1.children()),
                         (id(self.l1), id(self.l2)))

    def test_copyparams(self):
        l1 = chainer.Link(x=(2, 3))
        l2 = chainer.Link(x=2)
        l3 = chainer.Link(x=3)
        c1 = chainer.ChainList(l1, l2)
        c2 = chainer.ChainList(c1, l3)
        l1.x.data.fill(0)
        l2.x.data.fill(1)
        l3.x.data.fill(2)

        self.c2.copyparams(c2)

        numpy.testing.assert_array_equal(self.l1.x.data, l1.x.data)
        numpy.testing.assert_array_equal(self.l2.x.data, l2.x.data)
        numpy.testing.assert_array_equal(self.l3.x.data, l3.x.data)

    def test_zerograds(self):
        self.c2.zerograds()
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.zeros(3))

        self.assertTrue(self.l1._uninitialized_params['y']._zeroed)

    def test_cleargrads(self):
        self.c2.cleargrads()
        self.assertIsNone(self.l1.x.grad)
        self.assertIsNone(self.l2.x.grad)
        self.assertIsNone(self.l3.x.grad)

        self.assertTrue(self.l1._uninitialized_params['y']._cleared)

    def test_addgrads(self):
        l1 = chainer.Link(x=(2, 3))
        l2 = chainer.Link(x=2)
        l3 = chainer.Link(x=3)
        c1 = chainer.ChainList(l1, l2)
        c2 = chainer.ChainList(c1, l3)
        l1.x.grad.fill(1)
        l2.x.grad.fill(2)
        l3.x.grad.fill(3)

        self.l1.x.grad.fill(-1)
        self.l2.x.grad.fill(-2)
        self.l3.x.grad.fill(-3)

        self.c2.addgrads(c2)
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.zeros(3))

    def test_serialize(self):
        self.l1.add_param('y', (1, 1))
        mocks = {'0': mock.MagicMock(), '1': mock.MagicMock()}
        serializer = mock.MagicMock()
        serializer.__getitem__.side_effect = lambda k: mocks[k]
        self.c1.serialize(serializer)

        self.assertEqual(serializer.call_count, 0)
        self.assertEqual(serializer.__getitem__.call_count, 2)
        serializer.__getitem__.assert_any_call('0')
        serializer.__getitem__.assert_any_call('1')

        mocks['0'].assert_called_with('y', self.l1.y.data)
        mocks['1'].assert_called_with('x', self.l2.x.data)


testing.run_module(__name__, __file__)
