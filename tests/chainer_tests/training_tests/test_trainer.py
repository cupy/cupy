import collections
import os
import shutil
import tempfile
import time
import unittest

import mock

from chainer import serializers
from chainer import testing
from chainer import training


class TestTrainerElapsedTime(unittest.TestCase):

    def setUp(self):
        self.trainer = _get_mocked_trainer()

    def test_elapsed_time(self):
        with self.assertRaises(RuntimeError):
            self.trainer.elapsed_time

        self.trainer.run()

        self.assertGreater(self.trainer.elapsed_time, 0)

    def test_elapsed_time_serialization(self):
        self.trainer.run()
        serialized_time = self.trainer.elapsed_time

        tempdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tempdir, 'trainer.npz')
            serializers.save_npz(path, self.trainer)

            trainer = _get_mocked_trainer((20, 'iteration'))
            serializers.load_npz(path, trainer)

            trainer.run()

            self.assertGreater(trainer.elapsed_time, serialized_time)

        finally:
            shutil.rmtree(tempdir)
            


def _get_mocked_trainer(stop_trigger=(10, 'iteration')):
    updater = mock.Mock()
    updater.get_all_optimizers.return_value = {}
    updater.iteration = 0
    def update():
        updater.iteration += 1
    updater.update = update
    return training.Trainer(updater, stop_trigger)


testing.run_module(__name__, __file__)
