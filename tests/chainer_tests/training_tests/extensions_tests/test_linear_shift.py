import numpy
import unittest

import mock

from chainer import serializer
from chainer import testing
from chainer import training
from chainer.training import extensions


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


def _get_mocked_trainer():
    trainer = mock.Mock()

    def update():
        trainer.updater.iteration += 1
    trainer.updater.iteration = 0
    trainer.updater.update = update

    trainer.updater.get_optimizer = lambda _: trainer.updater.optimizer

    return trainer


class TestLinearShift(unittest.TestCase):

    value_range = (2.0, 6.0)
    time_range = (1, 3)
    expect = [2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0]

    def setUp(self):
        self.trainer = _get_mocked_trainer()
        self.optimizer = self.trainer.updater.get_optimizer('main')
        self.interval = 2
        self.trigger = training.get_trigger((self.interval, 'iteration'))

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
        self.trainer.updater.optimizer.x = 0
        extension = extensions.LinearShift(
            'x', self.value_range, self.time_range)
        self._run_trainer(extension, self.expect)

    def test_with_optimizer(self):
        optimizer = mock.Mock()
        optimizer.x = 0
        extension = extensions.LinearShift(
            'x', self.value_range, self.time_range, optimizer)
        self._run_trainer(extension, self.expect, optimizer)

    def test_serialize(self):
        self.trainer.updater.optimizer.x = 0
        extension = extensions.LinearShift(
            'x', self.value_range, self.time_range)
        self._run_trainer(extension, self.expect[:len(self.expect) // 2])
        target = dict()
        extension.serialize(DummySerializer(target))

        self.trainer.updater.optimizer.x = 0
        extension = extensions.LinearShift(
            'x', self.value_range, self.time_range)
        extension.serialize(DummyDeserializer(target))
        self._run_trainer(extension, self.expect[len(self.expect) // 2:])

    def test_serialize_before_first_interval(self):
        self.trainer.updater.optimizer.x = 0
        extension = extensions.LinearShift(
            'x', self.value_range, self.time_range)
        self._run_trainer(extension, self.expect[:self.interval - 1])
        target = dict()
        extension.serialize(DummySerializer(target))

        self.trainer.updater.optimizer.x = 0
        extension = extensions.LinearShift(
            'x', self.value_range, self.time_range)
        extension.serialize(DummyDeserializer(target))
        self._run_trainer(extension, self.expect[self.interval - 1:])


testing.run_module(__name__, __file__)
