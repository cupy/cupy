import functools
import unittest

import six


class QuietTestRunner(object):
    def run(self, suite):
        result = unittest.TestResult()
        suite(result)
        return result


def repeat_with_success_at_least(times, min_success):
    assert times >= min_success

    def _repeat_with_success_at_least(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            assert len(args) > 0
            cls = args[0]
            assert isinstance(cls, unittest.TestCase)
            success_counter = 0
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
            cls.fail()
        return wrapper
    return _repeat_with_success_at_least


def repeat(times):
    return repeat_with_success_at_least(times, times)


def retry(times):
    return repeat_with_success_at_least(times, 1)
