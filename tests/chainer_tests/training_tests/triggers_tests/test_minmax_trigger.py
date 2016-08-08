import unittest

import numpy

from chainer import testing
from chainer import training
from chainer.training import triggers


class DummyUpdater(training.Updater):

    def __init__(self):
        self.iteration = 0

    def finalize(self):
        pass

    def get_all_optimizers(self):
        return {}

    def update(self):
        self.iteration += 1

    @property
    def epoch(self):
        return 1

    @property
    def is_new_epoch(self):
        return False


def _test_trigger(self, trigger, key, accuracies, expected):
    updater = DummyUpdater()
    trainer = training.Trainer(updater)
    for accuracy, expected in zip(accuracies, expected):
        updater.update()
        trainer.observation = {key: accuracy}
        self.assertEqual(trigger(trainer), expected)


class TestMaxValueTrigger(unittest.TestCase):

    def test_max_value_trigger(self):
        key = 'main/accuracy'
        trigger = triggers.MaxValueTrigger(key, trigger=(2, 'iteration'))
        accuracies = numpy.asarray([0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
                                   dtype=numpy.float32)
        expected = [False, True, False, False, False, False, False, True]
        _test_trigger(self, trigger, key, accuracies, expected)


class TestMinValueTrigger(unittest.TestCase):

    def test_min_value_trigger(self):
        key = 'main/accuracy'
        trigger = triggers.MinValueTrigger(key, trigger=(2, 'iteration'))
        accuracies = numpy.asarray([0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
                                   dtype=numpy.float32)
        expected = [False, True, False, False, False, True, False, False]
        _test_trigger(self, trigger, key, accuracies, expected)


testing.run_module(__name__, __file__)
