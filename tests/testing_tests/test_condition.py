import unittest

from chainer import testing
from chainer.testing import condition


# The test fixtures of this TestCase is used to be decorated by
# decorator in test. So we do not run them alone.
class MockUnitTest(unittest.TestCase):

    failure_case_counter = 0
    success_case_counter = 0
    probabilistic_case_counter = 0
    probabilistic_case_success_counter = 0
    probabilistic_case_failure_counter = 0

    def failure_case(self):
        self.failure_case_counter += 1
        self.fail()

    def success_case(self):
        self.success_case_counter += 1
        self.assertTrue(True)

    def error_case(self):
        raise Exception()

    def probabilistic_case(self):
        self.probabilistic_case_counter += 1
        if self.probabilistic_case_counter % 2 == 0:
            self.probabilistic_case_success_counter += 1
            self.assertTrue(True)
        else:
            self.probabilistic_case_failure_counter += 1
            self.fail()

    def runTest(self):
        pass


def _should_fail(self, f):
    try:
        f(self.unit_test)
        self.fail(
            'AssertionError is expected to be raised, but none is raised')
    except AssertionError as e:
        # check if the detail is included in the error object
        self.assertIn('first error message:', str(e))


def _should_pass(self, f):
    f(self.unit_test)


class TestRepeatWithSuccessAtLeast(unittest.TestCase):

    def _decorate(self, f, times, min_success):
        return condition.repeat_with_success_at_least(
            times, min_success)(f)

    def setUp(self):
        self.unit_test = MockUnitTest()

    def test_all_trials_fail(self):
        f = self._decorate(MockUnitTest.failure_case, 10, 1)
        _should_fail(self, f)
        self.assertEqual(self.unit_test.failure_case_counter, 10)

    def test_all_trials_fail2(self):
        f = self._decorate(MockUnitTest.failure_case, 10, 0)
        _should_pass(self, f)
        self.assertLessEqual(self.unit_test.failure_case_counter, 10)

    def test_all_trials_error(self):
        f = self._decorate(MockUnitTest.error_case, 10, 1)
        _should_fail(self, f)

    def test_all_trials_succeed(self):
        f = self._decorate(MockUnitTest.success_case, 10, 10)
        _should_pass(self, f)
        self.assertEqual(self.unit_test.success_case_counter, 10)

    def test_all_trials_succeed2(self):
        self.assertRaises(AssertionError,
                          condition.repeat_with_success_at_least,
                          10, 11)

    def test_half_of_trials_succeed(self):
        f = self._decorate(MockUnitTest.probabilistic_case, 10, 5)
        _should_pass(self, f)
        self.assertLessEqual(self.unit_test.probabilistic_case_counter, 10)
        self.assertGreaterEqual(
            self.unit_test.probabilistic_case_success_counter, 5)
        self.assertLessEqual(
            self.unit_test.probabilistic_case_failure_counter, 5)

    def test_half_of_trials_succeed2(self):
        f = self._decorate(MockUnitTest.probabilistic_case, 10, 6)
        _should_fail(self, f)
        self.assertLessEqual(self.unit_test.probabilistic_case_counter, 10)
        self.assertLess(
            self.unit_test.probabilistic_case_success_counter, 6)
        self.assertGreaterEqual(
            self.unit_test.probabilistic_case_failure_counter, 5)


class TestRepeat(unittest.TestCase):

    def _decorate(self, f, times):
        return condition.repeat(times)(f)

    def setUp(self):
        self.unit_test = MockUnitTest()

    def test_failure_case(self):
        f = self._decorate(MockUnitTest.failure_case, 10)
        _should_fail(self, f)
        self.assertLessEqual(self.unit_test.failure_case_counter, 10)

    def test_success_case(self):
        f = self._decorate(MockUnitTest.success_case, 10)
        _should_pass(self, f)
        self.assertEqual(self.unit_test.success_case_counter, 10)

    def test_probabilistic_case(self):
        f = self._decorate(MockUnitTest.probabilistic_case, 10)
        _should_fail(self, f)
        self.assertLessEqual(self.unit_test.probabilistic_case_counter, 10)
        self.assertLess(self.unit_test.probabilistic_case_success_counter, 10)
        self.assertGreater(
            self.unit_test.probabilistic_case_failure_counter, 0)


class TestRetry(unittest.TestCase):

    def _decorate(self, f, times):
        return condition.retry(times)(f)

    def setUp(self):
        self.unit_test = MockUnitTest()

    def test_failure_case(self):
        f = self._decorate(MockUnitTest.failure_case, 10)
        _should_fail(self, f)
        self.assertEqual(self.unit_test.failure_case_counter, 10)

    def test_success_case(self):
        f = self._decorate(MockUnitTest.success_case, 10)
        _should_pass(self, f)
        self.assertLessEqual(self.unit_test.success_case_counter, 10)

    def test_probabilistic_case(self):
        f = self._decorate(MockUnitTest.probabilistic_case, 10)
        _should_pass(self, f)
        self.assertLessEqual(
            self.unit_test.probabilistic_case_counter, 10)
        self.assertGreater(
            self.unit_test.probabilistic_case_success_counter, 0)
        self.assertLess(self.unit_test.probabilistic_case_failure_counter, 10)


testing.run_module(__name__, __file__)
