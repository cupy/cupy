from __future__ import division

import unittest

from chainer import testing
from chainer import training
from chainer.training import triggers


class DummyUpdater(training.Updater):

    def __init__(self, iters_per_epoch, initial_iteration=1):
        self.iteration = initial_iteration - 1
        self.iters_per_epoch = iters_per_epoch

    def finalize(self):
        pass

    def get_all_optimizers(self):
        return {}

    def update(self):
        self.iteration += 1

    @property
    def epoch(self):
        return self.iteration // self.iters_per_epoch

    @property
    def epoch_detail(self):
        return self.iteration / self.iters_per_epoch

    @property
    def is_new_epoch(self):
        return 0 <= self.iteration % self.iters_per_epoch < 1


def _test_trigger(self, updater, trigger, expecteds):
    trainer = training.Trainer(updater)
    for expected in expecteds:
        updater.update()
        self.assertEqual(trigger(trainer), expected)


class TestIterationManualScheduleTrigger(unittest.TestCase):

    def test_iteration_manual_single_trigger(self):
        updater = DummyUpdater(iters_per_epoch=3)
        trigger = triggers.ManualScheduleTrigger(2, 'iteration')
        expected = [False, True, False, False, False]
        _test_trigger(self, updater, trigger, expected)

    def test_iteration_manual_multiple_trigger(self):
        updater = DummyUpdater(iters_per_epoch=5)
        trigger = triggers.ManualScheduleTrigger([2, 3], 'iteration')
        expected = [False, True, True, False, False, False, False]
        _test_trigger(self, updater, trigger, expected)


class TestEpochManualScheduleTrigger(unittest.TestCase):

    def test_epoch_manual_single_trigger(self):
        updater = DummyUpdater(iters_per_epoch=3)
        trigger = triggers.ManualScheduleTrigger(1, 'epoch')
        expected = [False, False, True, False, False, False, False]
        _test_trigger(self, updater, trigger, expected)

    def test_epoch_manual_multiple_trigger(self):
        updater = DummyUpdater(iters_per_epoch=3)
        trigger = triggers.ManualScheduleTrigger([1, 2], 'epoch')
        expected = [False, False, True, False, False, True, False]
        _test_trigger(self, updater, trigger, expected)


class TestFractionalEpochManualScheduleTrigger(unittest.TestCase):

    def test_epoch_manual_single_trigger(self):
        updater = DummyUpdater(iters_per_epoch=2)
        trigger = triggers.ManualScheduleTrigger(1.5, 'epoch')
        expected = [False, False, True, False, False, False, False]
        _test_trigger(self, updater, trigger, expected)

    def test_epoch_manual_multiple_trigger(self):
        updater = DummyUpdater(iters_per_epoch=2)
        trigger = triggers.ManualScheduleTrigger([1.5, 2.5], 'epoch')
        expected = [False, False, True, False, True, False, False]
        _test_trigger(self, updater, trigger, expected)


class TestUnalignedEpochManualScheduleTrigger(unittest.TestCase):

    def test_unaligned_epoch_single_manual_trigger(self):
        updater = DummyUpdater(iters_per_epoch=2.5)
        trigger = triggers.ManualScheduleTrigger(1, 'epoch')
        expected = [False, False, True, False, False, False, False]
        _test_trigger(self, updater, trigger, expected)

    def test_unaligned_epoch_multiple_manual_trigger(self):
        updater = DummyUpdater(iters_per_epoch=2.5)
        trigger = triggers.ManualScheduleTrigger([1, 2], 'epoch')
        expected = [False, False, True, False, True, False, False, False]
        _test_trigger(self, updater, trigger, expected)


class TestResumedIterationManualScheduleTrigger(unittest.TestCase):

    def test_resumed_iteration_single_manual_trigger(self):
        updater = DummyUpdater(iters_per_epoch=1, initial_iteration=3)
        trigger = triggers.ManualScheduleTrigger(3, 'iteration')
        expected = [True, False, False, False]
        _test_trigger(self, updater, trigger, expected)

    def test_resumed_iteration_multiple_manual_trigger(self):
        updater = DummyUpdater(iters_per_epoch=1, initial_iteration=3)
        trigger = triggers.ManualScheduleTrigger([1, 3, 5], 'iteration')
        expected = [True, False, True, False, False]
        _test_trigger(self, updater, trigger, expected)


class TestResumedEpochManualScheduleTrigger(unittest.TestCase):

    def test_resumed_epoch_single_manual_trigger(self):
        updater = DummyUpdater(iters_per_epoch=2.5, initial_iteration=3)
        trigger = triggers.ManualScheduleTrigger(3, 'epoch')
        expected = [False, False, False, False, False, True, False]
        _test_trigger(self, updater, trigger, expected)

    def test_resumed_epoch_multiple_manual_trigger(self):
        updater = DummyUpdater(iters_per_epoch=2.5, initial_iteration=3)
        trigger = triggers.ManualScheduleTrigger([1, 3, 5], 'epoch')
        expected = [False, False, False, False, False,
                    True, False, False, False, False, True, False]
        _test_trigger(self, updater, trigger, expected)


testing.run_module(__name__, __file__)
