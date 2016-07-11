import unittest

import chainer
from chainer import testing


class TestReporter(unittest.TestCase):

    def test_empty_reporter(self):
        reporter = chainer.Reporter()
        self.assertEqual(reporter.observation, {})

    def test_enter_exit(self):
        reporter1 = chainer.Reporter()
        reporter2 = chainer.Reporter()
        with reporter1:
            self.assertIs(chainer.get_current_reporter(), reporter1)
            with reporter2:
                self.assertIs(chainer.get_current_reporter(), reporter2)
            self.assertIs(chainer.get_current_reporter(), reporter1)

    def test_scope(self):
        reporter1 = chainer.Reporter()
        reporter2 = chainer.Reporter()
        with reporter1:
            observation = {}
            with reporter2.scope(observation):
                self.assertIs(chainer.get_current_reporter(), reporter2)
                self.assertIs(reporter2.observation, observation)
            self.assertIs(chainer.get_current_reporter(), reporter1)
            self.assertIsNot(reporter2.observation, observation)

    def test_add_observer(self):
        reporter = chainer.Reporter()
        observer = object()
        reporter.add_observer('o', observer)

        reporter.report({'x': 1}, observer)

        observation = reporter.observation
        self.assertIn('o/x', observation)
        self.assertEqual(observation['o/x'], 1)
        self.assertNotIn('x', observation)

    def test_add_observers(self):
        reporter = chainer.Reporter()
        observer1 = object()
        reporter.add_observer('o1', observer1)
        observer2 = object()
        reporter.add_observer('o2', observer2)

        reporter.report({'x': 1}, observer1)
        reporter.report({'y': 2}, observer2)

        observation = reporter.observation
        self.assertIn('o1/x', observation)
        self.assertEqual(observation['o1/x'], 1)
        self.assertIn('o2/y', observation)
        self.assertEqual(observation['o2/y'], 2)
        self.assertNotIn('x', observation)
        self.assertNotIn('y', observation)
        self.assertNotIn('o1/y', observation)
        self.assertNotIn('o2/x', observation)

    def test_report_without_observer(self):
        reporter = chainer.Reporter()
        reporter.report({'x': 1})

        observation = reporter.observation
        self.assertIn('x', observation)
        self.assertEqual(observation['x'], 1)


class TestReport(unittest.TestCase):

    def test_report_without_reporter(self):
        observer = object()
        chainer.report({'x': 1}, observer)

    def test_report(self):
        reporter = chainer.Reporter()
        with reporter:
            chainer.report({'x': 1})
        observation = reporter.observation
        self.assertIn('x', observation)
        self.assertEqual(observation['x'], 1)

    def test_report_with_observer(self):
        reporter = chainer.Reporter()
        observer = object()
        reporter.add_observer('o', observer)
        with reporter:
            chainer.report({'x': 1}, observer)
        observation = reporter.observation
        self.assertIn('o/x', observation)
        self.assertEqual(observation['o/x'], 1)

    def test_report_with_unregistered_observer(self):
        reporter = chainer.Reporter()
        observer = object()
        with reporter:
            with self.assertRaises(KeyError):
                chainer.report({'x': 1}, observer)

    def test_report_scope(self):
        reporter = chainer.Reporter()
        observation = {}

        with reporter:
            with chainer.report_scope(observation):
                chainer.report({'x': 1})

        self.assertIn('x', observation)
        self.assertEqual(observation['x'], 1)
        self.assertNotIn('x', reporter.observation)


testing.run_module(__name__, __file__)
