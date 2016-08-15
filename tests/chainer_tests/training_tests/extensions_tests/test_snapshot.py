import unittest

import mock

from chainer import testing
from chainer.training import extensions
from chainer.training import trigger


@testing.parameterize(
    {'trigger': ('epoch', 2)},
    {'trigger': ('iteration', 10)},
    {'trigger': trigger.IntervalTrigger(5, 'epoch')},
    {'trigger': trigger.IntervalTrigger(20, 'iteration')},
)
class TestSnapshotObject(unittest.TestCase):

    def test_trigger(self):
        target = mock.MagicMock()
        snapshot_object = extensions.snapshot_object(target, 'myfile.dat',
                                                     trigger=self.trigger)
        self.assertEqual(snapshot_object.trigger, self.trigger)


@testing.parameterize(
    {'trigger': ('epoch', 2)},
    {'trigger': ('iteration', 10)},
    {'trigger': trigger.IntervalTrigger(5, 'epoch')},
    {'trigger': trigger.IntervalTrigger(20, 'iteration')},
)
class TestSnapshot(unittest.TestCase):

    def test_trigger(self):
        snapshot = extensions.snapshot(trigger=self.trigger)
        self.assertEqual(snapshot.trigger, self.trigger)


testing.run_module(__name__, __file__)
