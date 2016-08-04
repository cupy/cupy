import unittest

import mock
import numpy

import chainer
from chainer import dataset
from chainer import testing
from chainer import training


class DummyIterator(dataset.Iterator):

    epoch = 1
    is_new_epoch = True

    def __init__(self, next_data):
        self.finalize = mock.MagicMock()
        self.__next__ = mock.MagicMock(return_value=next_data)
        self.serialize = mock.MagicMock()


class DummyOptimizer(chainer.Optimizer):

    def __init__(self):
        self.update = mock.MagicMock()
        self.serialize = mock.MagicMock()


class DummySerializer(chainer.Serializer):

    def __init__(self, path=[]):
        self.path = path
        self.called = []

    def __getitem__(self, key):
        return DummySerializer(self.path + [key])

    def __call__(self, key, value):
        self.called.append((key, value))


class TestUpdater(unittest.TestCase):

    def setUp(self):
        self.target = chainer.Link()
        self.iterator = DummyIterator([(numpy.array(1), numpy.array(2))])
        self.optimizer = DummyOptimizer()
        self.optimizer.setup(self.target)
        self.updater = training.StandardUpdater(self.iterator, self.optimizer)

    def test_init_values(self):
        self.assertIsNone(self.updater.device)
        self.assertIsNone(self.updater.loss_func)
        self.assertEqual(self.updater.iteration, 0)

    def test_epoch(self):
        self.assertEqual(self.updater.epoch, 1)

    def test_new_epoch(self):
        self.assertTrue(self.updater.is_new_epoch)

    def test_get_iterator(self):
        self.assertIs(self.updater.get_iterator('main'), self.iterator)

    def test_get_optimizer(self):
        self.assertIs(self.updater.get_optimizer('main'), self.optimizer)

    def test_get_all_optimizers(self):
        self.assertEqual(self.updater.get_all_optimizers(),
                         {'main': self.optimizer})

    def test_update(self):
        self.updater.update()
        self.assertEqual(self.updater.iteration, 1)
        self.assertEqual(self.iterator.__next__.call_count, 1)

    def test_finalizer(self):
        self.updater.finalize()
        self.assertEqual(self.iterator.finalize.call_count, 1)

    def test_serialize(self):
        serializer = DummySerializer()
        self.updater.serialize(serializer)

        self.assertEqual(self.iterator.serialize.call_count, 1)
        args, kwargs = self.iterator.serialize.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertEqual(args[0].path, ['iterator:main'])

        self.assertEqual(self.optimizer.serialize.call_count, 1)
        args, kwargs = self.optimizer.serialize.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertEqual(args[0].path, ['optimizer:main'])

        self.assertEqual(serializer.called, [('iteration', 0)])


class TestUpdaterUpdateArguments(unittest.TestCase):

    def setUp(self):
        self.target = chainer.Link()
        self.optimizer = DummyOptimizer()
        self.optimizer.setup(self.target)

    def test_update_tuple(self):
        iterator = DummyIterator([(numpy.array(1), numpy.array(2))])
        updater = training.StandardUpdater(iterator, self.optimizer)

        updater.update_core()

        self.assertEqual(self.optimizer.update.call_count, 1)
        args, kwargs = self.optimizer.update.call_args
        self.assertEqual(len(args), 3)
        loss, v1, v2 = args
        self.assertEqual(len(kwargs), 0)

        self.assertIs(loss, self.optimizer.target)
        self.assertIsInstance(v1, chainer.Variable)
        self.assertEqual(v1.data, 1)
        self.assertIsInstance(v2, chainer.Variable)
        self.assertEqual(v2 .data, 2)

        self.assertEqual(iterator.__next__.call_count, 1)

    def test_update_dict(self):
        iterator = DummyIterator([{'x': numpy.array(1), 'y': numpy.array(2)}])
        updater = training.StandardUpdater(iterator, self.optimizer)

        updater.update_core()

        self.assertEqual(self.optimizer.update.call_count, 1)
        args, kwargs = self.optimizer.update.call_args
        self.assertEqual(len(args), 1)
        loss, = args
        self.assertEqual(set(kwargs.keys()), {'x', 'y'})

        v1 = kwargs['x']
        v2 = kwargs['y']
        self.assertIs(loss, self.optimizer.target)
        self.assertIsInstance(v1, chainer.Variable)
        self.assertEqual(v1.data, 1)
        self.assertIsInstance(v2, chainer.Variable)
        self.assertEqual(v2 .data, 2)

        self.assertEqual(iterator.__next__.call_count, 1)

    def test_update_var(self):
        iterator = DummyIterator([numpy.array(1)])
        updater = training.StandardUpdater(iterator, self.optimizer)

        updater.update_core()

        self.assertEqual(self.optimizer.update.call_count, 1)
        args, kwargs = self.optimizer.update.call_args
        self.assertEqual(len(args), 2)
        loss, v1 = args
        self.assertEqual(len(kwargs), 0)

        self.assertIs(loss, self.optimizer.target)
        self.assertIsInstance(v1, chainer.Variable)
        self.assertEqual(v1.data, 1)

        self.assertEqual(iterator.__next__.call_count, 1)


testing.run_module(__name__, __file__)
