import mock
import unittest

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


class TestExtension(unittest.TestCase):

    def setUp(self):
        self.trainer = _get_mocked_trainer()

    def test_default_name(self):
        class MyExtension(training.Extension):
            pass

        ext = MyExtension()
        self.assertEqual(ext.default_name, 'MyExtension')

    def test_make_extension(self):
        @training.make_extension(trigger=(2, 'epoch'), default_name='my_ext',
                                 priority=50, invoke_before_training=True)
        def my_extension(trainer):
            pass

        self.assertEqual(my_extension.trigger, (2, 'epoch'))
        self.assertEqual(my_extension.default_name, 'my_ext')
        self.assertEqual(my_extension.priority, 50)
        self.assertTrue(my_extension.invoke_before_training)

    def test_make_extension_default_values(self):
        @training.make_extension()
        def my_extension(trainer):
            pass

        self.assertEqual(my_extension.trigger, (1, 'iteration'))
        self.assertEqual(my_extension.default_name, 'my_extension')
        self.assertEqual(my_extension.priority, training.PRIORITY_READER)
        self.assertFalse(my_extension.invoke_before_training)

    def test_add_class_extension(self):
        dummy_extension = DummyExtension()
        self.trainer.extend(dummy_extension)
        self.trainer.run()
        self.assertTrue(dummy_extension.is_called)
        self.assertTrue(dummy_extension.is_finalized)

    def test_add_make_extension(self):
        self.is_called = False

        @training.make_extension()
        def dummy_extension(trainer):
            self.is_called = True

        self.trainer.extend(dummy_extension)
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
        updater.iteration += 1

    updater.update = update
    return training.Trainer(updater, stop_trigger)


testing.run_module(__name__, __file__)
