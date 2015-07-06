import functools
import unittest

import six


class QuietTestRunner(object):
    def run(self, suite):
        result = unittest.TestResult()
        suite(result)
        return result


def repeat_with_success_at_least(times, min_success):
    """Decorator for multiple trial of the test case

    Decorated test case is launched multiple times.
    The case is judged as passed at least specified number of trials.
    If the number of successful trials exceeds `min_success`,
    the remaining trials are skipped.

    Args:
        times(int): The number of trials
        min_success(int): Threshold that the decorated test
            case is regarded as passed.

    """

    assert times >= min_success

    def _repeat_with_success_at_least(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            assert len(args) > 0
            cls = args[0]
            assert isinstance(cls, unittest.TestCase)
            success_counter = 0
            failure_counter = 0
            for _ in six.moves.range(times):
                suite = unittest.TestSuite()
                suite.addTest(
                    unittest.FunctionTestCase(
                        lambda: f(*args, **kwargs),
                        setUp=cls.setUp,
                        tearDown=cls.tearDown))

                if QuietTestRunner().run(suite).wasSuccessful():
                    success_counter += 1
                    if success_counter >= min_success:
                        cls.assertTrue(True)
                        return
                else:
                    failure_counter += 1
                    if failure_counter > times - min_success:
                        cls.fail()
            cls.fail()
        return wrapper
    return _repeat_with_success_at_least


def repeat(times):
    """Decorator that imposes the test to be successful in a row.

    Decorated test case is launched multiple times.
    The case is regarded as passed only if it is successful
    specified times in a row.

    Args:
        times(int): The number of trials.
    """
    return repeat_with_success_at_least(times, times)


def retry(times):
    """Decorator that imposes the test to be successful at least once.

    Decorated test case is launched multiple times.
    The case is regarded as passed if it is successful
    at least once.

    Args:
        times(int): The number of trials.
    """
    return repeat_with_success_at_least(times, 1)
