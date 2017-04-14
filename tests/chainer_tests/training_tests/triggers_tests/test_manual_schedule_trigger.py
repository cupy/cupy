from __future__ import division

import tempfile
import unittest

from chainer import serializers
from chainer import testing
from chainer import training


class DummyUpdater(training.Updater):

    def __init__(self, iters_per_epoch, initial_iteration=0):
        self.iteration = initial_iteration
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
    def previous_epoch_detail(self):
        if self.iteration == 0:
            return None
        return (self.iteration - 1) / self.iters_per_epoch

    @property
    def is_new_epoch(self):
        return 0 <= self.iteration % self.iters_per_epoch < 1

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)


@testing.parameterize(*testing.product_dict([
    # single iteration
    {
        'iters_per_epoch': 2, 'schedule': (2, 'iteration'), 'resume': 3,
        'expected': [False, True, False, False, False, False, False]},
    # multiple iteration
    {
        'iters_per_epoch': 2, 'schedule': ([2, 4], 'iteration'), 'resume': 3,
        'expected': [False, True, False, True, False, False, False]},
    # single epoch
    {
        'iters_per_epoch': 3, 'schedule': (1, 'epoch'), 'resume': 3,
        'expected': [False, False, True, False, False, False, False]},
    # multiple epoch
    {
        'iters_per_epoch': 3, 'schedule': ([1, 2], 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, False, True, False]},
    # single fractional epoch
    {
        'iters_per_epoch': 2, 'schedule': (1.5, 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, False, False, False]},
    # multiple fractional epoch
    {
        'iters_per_epoch': 2, 'schedule': ([1.5, 2.5], 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, True, False, False]},
    # single unaligned epoch
    {
        'iters_per_epoch': 2.5, 'schedule': (1, 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, False, False, False]},
    # multiple unaligned epoch
    {
        'iters_per_epoch': 2.5, 'schedule': ([1, 2], 'epoch'), 'resume': 4,
        'expected': [False, False, True, False, True, False, False]},
]))
class TestTrigger(unittest.TestCase):

    def test_trigger(self):
        trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
        updater = DummyUpdater(self.iters_per_epoch)
        trainer = training.Trainer(updater)
        for expected in self.expected:
            updater.update()
            self.assertEqual(trigger(trainer), expected)

    def test_resumed_trigger(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
            updater = DummyUpdater(self.iters_per_epoch)
            trainer = training.Trainer(updater)
            for expected in self.expected[:self.resume]:
                updater.update()
                self.assertEqual(trigger(trainer), expected)
            serializers.save_npz(f.name, updater)

            trigger = training.triggers.ManualScheduleTrigger(*self.schedule)
            updater = DummyUpdater(self.iters_per_epoch)
            serializers.load_npz(f.name, updater)
            trainer = training.Trainer(updater)
            for expected in self.expected[self.resume:]:
                updater.update()
                self.assertEqual(trigger(trainer), expected)


testing.run_module(__name__, __file__)
