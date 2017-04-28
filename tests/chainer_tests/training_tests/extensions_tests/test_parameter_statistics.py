import re
import time
import unittest

import mock
import numpy
import six

import chainer
from chainer import testing
from chainer import training
from chainer.training import extensions


class TestParameterStatisticsBase(object):

    def setUp(self):
        self.trainer = _get_mocked_trainer()

    @property
    def links(self):
        links = []
        for optimizer in six.itervalues(
                self.trainer.updater.get_all_optimizers()):
            for _, link in optimizer.target.namedlinks():
                links.append(link)
        return links


@testing.parameterize(
    {'statistics': {'min': numpy.min}, 'expect': 4},
    {'statistics': {'mean': numpy.mean, 'std': numpy.std}, 'expect': 8},
    {'statistics': extensions.ParameterStatistics.default_statistics,
     'expect': 48},
    {'statistics': {}, 'expect': 0}
)
class TestParameterStatistics(TestParameterStatisticsBase, unittest.TestCase):

    def test_report(self):
        extension = extensions.ParameterStatistics(self.links,
                                                   statistics=self.statistics)
        self.trainer.extend(extension)
        self.trainer.run()

        self.assertEqual(len(self.trainer.observation), self.expect)

    def test_report_late_register(self):
        extension = extensions.ParameterStatistics(self.links, statistics={})
        for name, function in six.iteritems(self.statistics):
            extension.register_statistics(name, function)
        self.trainer.extend(extension)
        self.trainer.run()

        self.assertEqual(len(self.trainer.observation), self.expect)

    def test_report_key_pattern(self):
        extension = extensions.ParameterStatistics(self.links)
        self.trainer.extend(extension)
        self.trainer.run()

        pattern = r'^(.+/){2,}(data|grad)/.+[^/]$'
        for name in six.iterkeys(self.trainer.observation):
            self.assertTrue(re.match(pattern, name))


@testing.parameterize(
    {'statistics': {'one': lambda x: 1.0}, 'expect': 1.0}
)
class TestParameterStatisticsCustomFunction(TestParameterStatisticsBase,
                                            unittest.TestCase):

    def test_custom_function(self):
        extension = extensions.ParameterStatistics(
            self.links, statistics=self.statistics)
        self.trainer.extend(extension)
        self.trainer.run()

        for value in six.itervalues(self.trainer.observation):
            self.assertEqual(value, self.expect)


@testing.parameterize(
    {'statistics': extensions.ParameterStatistics.default_statistics}
)
class TestParameterStatisticsArguments(TestParameterStatisticsBase,
                                       unittest.TestCase):

    def test_skip_params(self):
        extension = extensions.ParameterStatistics(
            self.links, statistics=self.statistics, report_params=False)
        self.trainer.extend(extension)
        self.trainer.run()

        for name in six.iterkeys(self.trainer.observation):
            self.assertIn('grad', name)
            self.assertNotIn('data', name)

    def test_skip_grads(self):
        extension = extensions.ParameterStatistics(
            self.links, statistics=self.statistics, report_grads=False)
        self.trainer.extend(extension)
        self.trainer.run()

        for name in six.iterkeys(self.trainer.observation):
            self.assertIn('data', name)
            self.assertNotIn('grad', name)

    def test_report_key_prefix(self):
        extension = extensions.ParameterStatistics(
            self.links, statistics=self.statistics, prefix='prefix')
        self.trainer.extend(extension)
        self.trainer.run()

        for name in six.iterkeys(self.trainer.observation):
            self.assertTrue(name.startswith('prefix'))


def _get_mocked_trainer(stop_trigger=(10, 'iteration')):
    updater = mock.Mock()
    optimizer = mock.Mock()
    target = mock.Mock()
    target.namedlinks.return_value = ('link_name',
                                      chainer.links.Linear(10, 10)),
    optimizer.target = target
    updater.get_all_optimizers.return_value = {'optimizer_name': optimizer}
    updater.iteration = 0
    updater.epoch = 0
    updater.epoch_detail = 0
    updater.is_new_epoch = True
    iter_per_epoch = 10

    def update():
        time.sleep(0.001)
        updater.iteration += 1
        updater.epoch = updater.iteration // iter_per_epoch
        updater.epoch_detail = updater.iteration / iter_per_epoch
        updater.is_new_epoch = updater.epoch == updater.epoch_detail

    updater.update = update

    return training.Trainer(updater, stop_trigger)


testing.run_module(__name__, __file__)
