from __future__ import division

import unittest

from chainer import testing
from chainer import training
from chainer.training import triggers


class DummyUpdater(training.Updater):

    def __init__(self, iters_per_epoch):
        self.iteration = 0
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


class TestIterationIrregularTrigger(unittest.TestCase):

    def test_iteration_irregular_single_trigger(self):
        updater = DummyUpdater(iters_per_epoch=3)
        trigger = triggers.IrregularTrigger(2, 'iteration')
        expected = [False, True, False, False, False]
        _test_trigger(self, updater, trigger, expected)

    def test_iteration_irregular_multiple_trigger(self):
        updater = DummyUpdater(iters_per_epoch=5)
        trigger = triggers.IrregularTrigger([2, 3], 'iteration')
        expected = [False, True, True, False, False, False, False]
        _test_trigger(self, updater, trigger, expected)


class TestEpochIrregularTrigger(unittest.TestCase):

    def test_epoch_irregular_single_trigger(self):
        updater = DummyUpdater(iters_per_epoch=3)
        trigger = triggers.IrregularTrigger(1, 'epoch')
        expected = [False, False, True, False, False, False, False]
        _test_trigger(self, updater, trigger, expected)

    def test_epoch_irregular_multiple_trigger(self):
        updater = DummyUpdater(iters_per_epoch=3)
        trigger = triggers.IrregularTrigger([1, 2], 'epoch')
        expected = [False, False, True, False, False, True, False]
        _test_trigger(self, updater, trigger, expected)


class TestFractionalEpochIrregularTrigger(unittest.TestCase):

    def test_epoch_irregular_single_trigger(self):
        updater = DummyUpdater(iters_per_epoch=2)
        trigger = triggers.IrregularTrigger(1.5, 'epoch')
        expected = [False, False, True, False, False, False, False]
        _test_trigger(self, updater, trigger, expected)

    def test_epoch_irregular_multiple_trigger(self):
        updater = DummyUpdater(iters_per_epoch=2)
        trigger = triggers.IrregularTrigger([1.5, 2.5], 'epoch')
        expected = [False, False, True, False, True, False, False]
        _test_trigger(self, updater, trigger, expected)


class TestUnalignedEpochIrregularTrigger(unittest.TestCase):

    def test_unaligned_epoch_single_irregular_trigger(self):
        updater = DummyUpdater(iters_per_epoch=2.5)
        trigger = triggers.IrregularTrigger(1, 'epoch')
        expected = [False, False, True, False, False, False, False]
        _test_trigger(self, updater, trigger, expected)

    def test_unaligned_epoch_multiple_irregular_trigger(self):
        updater = DummyUpdater(iters_per_epoch=2.5)
        trigger = triggers.IrregularTrigger([1, 2], 'epoch')
        expected = [False, False, True, False, True, False, False, False]
        _test_trigger(self, updater, trigger, expected)


testing.run_module(__name__, __file__)
