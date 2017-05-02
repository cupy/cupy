import numpy
import unittest

import mock

from chainer import serializer
from chainer import testing
from chainer.training import extensions
from chainer.training.util import get_trigger


class DummySerializer(serializer.Serializer):

    def __init__(self, target):
        super(DummySerializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        self.target[key] = value
        return self.target[key]


class DummyDeserializer(serializer.Deserializer):

    def __init__(self, target):
        super(DummyDeserializer, self).__init__()
        self.target = target

    def __getitem__(self, key):
        raise NotImplementedError

    def __call__(self, key, value):
        if value is None:
            value = self.target[key]
        elif isinstance(value, numpy.ndarray):
            numpy.copyto(value, self.target[key])
        else:
            value = type(value)(numpy.asarray(self.target[key]))
        return value


def _get_mocked_trainer(init):
    trainer = mock.Mock()

    def update():
        trainer.updater.iteration += 1
    trainer.updater.iteration = 0
    trainer.updater.update = update

    trainer.updater.optimizer.x = init
    trainer.updater.get_optimizer = lambda _: trainer.updater.optimizer

    return trainer


@testing.parameterize(
    {'init': 2.0, 'rate': 0.5, 'target': None, 'expect': [2.0, 1.0, 0.5]},
    {'init': 2.0, 'rate': 0.5, 'target': 1.2, 'expect': [2.0, 1.2, 1.2]},
    {'init': -2.0, 'rate': 0.5, 'target': -1.2, 'expect': [-2.0, -1.2, -1.2]},
    {'init': 2.0, 'rate': 2.0, 'target': None, 'expect': [2.0, 4.0, 8.0]},
    {'init': 2.0, 'rate': 2.0, 'target': 3.0, 'expect': [2.0, 3.0, 3.0]},
    {'init': -2.0, 'rate': 2.0, 'target': -3.0, 'expect': [-2.0, -3.0, -3.0]},
)
class TestExponentialShift(unittest.TestCase):

    def setUp(self):
        self.trainer = _get_mocked_trainer(self.init)

        self.interval = 4
        self.expect = [e for e in self.expect for _ in range(self.interval)]
        self.trigger = get_trigger((self.interval, 'iteration'))

    def _run_trainer(self, extension, expect, optimizer=None):
        if optimizer is None:
            optimizer = self.trainer.updater.optimizer

        if extension.invoke_before_training:
            extension(self.trainer)

        for e in expect:
            self.trainer.updater.update()
            self.assertEqual(optimizer.x, e)
            if self.trigger(self.trainer):
                extension(self.trainer)

    def test_basic(self):
        extension = extensions.ExponentialShift(
            'x', self.rate, target=self.target)
        self._run_trainer(extension, self.expect)

    def test_with_init(self):
        self.trainer.updater.optimizer.x = 0
        extension = extensions.ExponentialShift(
            'x', self.rate, init=self.init, target=self.target)
        self._run_trainer(extension, self.expect)

    def test_with_optimizer(self):
        optimizer = mock.Mock()
        optimizer.x = self.init
        extension = extensions.ExponentialShift(
            'x', self.rate, target=self.target, optimizer=optimizer)
        self._run_trainer(extension, self.expect, optimizer)

    def test_serialize(self):
        extension = extensions.ExponentialShift(
            'x', self.rate, target=self.target)
        self._run_trainer(extension, self.expect[:len(self.expect) // 2])
        target = dict()
        extension.serialize(DummySerializer(target))

        extension = extensions.ExponentialShift(
            'x', self.rate, target=self.target)
        extension.serialize(DummyDeserializer(target))
        self._run_trainer(extension, self.expect[len(self.expect) // 2:])

    def test_serialize_before_first_interval(self):
        extension = extensions.ExponentialShift(
            'x', self.rate, target=self.target)
        self._run_trainer(extension, self.expect[:self.interval - 1])
        target = dict()
        extension.serialize(DummySerializer(target))

        extension = extensions.ExponentialShift(
            'x', self.rate, target=self.target)
        extension.serialize(DummyDeserializer(target))
        self._run_trainer(extension, self.expect[self.interval - 1:])

    def test_serialize_backward_compat(self):
        extension = extensions.ExponentialShift(
            'x', self.rate, target=self.target)
        self._run_trainer(extension, self.expect[:len(self.expect) // 2])
        target = dict()
        extension.serialize(DummySerializer(target))

        # older version does not have '_init'
        del target['_init']

        extension = extensions.ExponentialShift(
            'x', self.rate, target=self.target)
        extension.serialize(DummyDeserializer(target))
        self._run_trainer(extension, self.expect[len(self.expect) // 2:])


class TestExponentialShiftInvalidArgument(unittest.TestCase):

    def test_negative_rate(self):
        with self.assertRaises(ValueError):
            extensions.ExponentialShift('x', -1.0)


testing.run_module(__name__, __file__)
