import os
import shutil
import tempfile
import time
import unittest

import mock

from chainer import serializers
from chainer import testing
from chainer import training


class DummyExtension(training.extension.Extension):

    def __init__(self):
        self.is_called = False
        self.is_finalized = False

    def __call__(self, trainer):
        self.is_called = True

    def finalize(self):
        self.is_finalized = True


class DummyCallableClass(object):

    def __init__(self):
        self.name = "DummyCallableClass"
        self.is_called = False
        self.is_finalized = False

    def __call__(self, trainer):
        self.is_called = True

    def finalize(self):
        self.is_finalized = True


class DummyClass(object):

    def __init__(self):
        self.is_touched = False

    def touch(self):
        self.is_touched = True


class TestTrainer(unittest.TestCase):

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

    def test_add_inherit_class_extension(self):
        dummy_extension = DummyExtension()
        self.trainer.extend(dummy_extension)
        self.trainer.run()
        self.assertTrue(dummy_extension.is_called)
        self.assertTrue(dummy_extension.is_finalized)

    def test_add_callable_class_extension(self):
        dummy_callable_class = DummyCallableClass()
        self.trainer.extend(dummy_callable_class)
        self.trainer.run()
        self.assertTrue(dummy_callable_class.is_called)
        self.assertTrue(dummy_callable_class.is_finalized)

    def test_add_lambda_extension(self):
        dummy_class = DummyClass()
        self.trainer.extend(lambda x: dummy_class.touch())
        self.trainer.run()
        self.assertTrue(dummy_class.is_touched)

    def test_add_make_extension(self):
        self.is_called = False

        @training.make_extension()
        def dummy_extension(trainer):
            self.is_called = True

        self.trainer.extend(dummy_extension)
        self.trainer.run()
        self.assertTrue(self.is_called)

    def test_add_function_extension(self):
        self.is_called = False

        def dummy_function(trainer):
            self.is_called = True

        self.trainer.extend(dummy_function)
        self.trainer.run()
        self.assertTrue(self.is_called)

    def test_add_two_extensions_default_priority(self):
        self.called_order = []

        @training.make_extension(trigger=(1, 'epoch'))
        def dummy_extension_1(trainer):
            self.called_order.append(1)

        @training.make_extension(trigger=(1, 'epoch'))
        def dummy_extension_2(trainer):
            self.called_order.append(2)

        self.trainer.extend(dummy_extension_1)
        self.trainer.extend(dummy_extension_2)
        self.trainer.run()
        self.assertEqual(self.called_order, [1, 2])

    def test_add_two_extensions_specific_priority(self):
        self.called_order = []

        @training.make_extension(trigger=(1, 'epoch'), priority=50)
        def dummy_extension_1(trainer):
            self.called_order.append(1)

        @training.make_extension(trigger=(1, 'epoch'), priority=100)
        def dummy_extension_2(trainer):
            self.called_order.append(2)

        self.trainer.extend(dummy_extension_1)
        self.trainer.extend(dummy_extension_2)
        self.trainer.run()
        self.assertEqual(self.called_order, [2, 1])


def _get_mocked_trainer(stop_trigger=(10, 'iteration')):
    updater = mock.Mock()
    updater.get_all_optimizers.return_value = {}
    updater.iteration = 0
    updater.epoch_detail = 1

    def update():
        time.sleep(0.001)
        updater.iteration += 1

    updater.update = update
    return training.Trainer(updater, stop_trigger)


testing.run_module(__name__, __file__)
