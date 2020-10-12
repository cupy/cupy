import functools
import os
import unittest


class QuietTestRunner(object):

    def run(self, suite):
        result = unittest.TestResult()
        suite(result)
        return result


def repeat_with_success_at_least(times, min_success):
    """Decorator for multiple trial of the test case.

    The decorated test case is launched multiple times.
    The case is judged as passed at least specified number of trials.
    If the number of successful trials exceeds `min_success`,
    the remaining trials are skipped.

    Args:
        times(int): The number of trials.
        min_success(int): Threshold that the decorated test
            case is regarded as passed.

    """

    assert times >= min_success

    def _repeat_with_success_at_least(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            assert len(args) > 0
            instance = args[0]
            assert isinstance(instance, unittest.TestCase)
            success_counter = 0
            failure_counter = 0
            results = []

            def fail():
                msg = '\nFail: {0}, Success: {1}'.format(
                    failure_counter, success_counter)
                if len(results) > 0:
                    first = results[0]
                    errs = first.failures + first.errors
                    if len(errs) > 0:
                        err_msg = '\n'.join(fail[1] for fail in errs)
                        msg += '\n\nThe first error message:\n' + err_msg
                instance.fail(msg)

            for _ in range(times):
                suite = unittest.TestSuite()
                # Create new instance to call the setup and the teardown only
                # once.
                ins = type(instance)(instance._testMethodName)
                suite.addTest(
                    unittest.FunctionTestCase(
                        lambda: f(ins, *args[1:], **kwargs),
                        setUp=ins.setUp,
                        tearDown=ins.tearDown))

                result = QuietTestRunner().run(suite)
                if len(result.skipped) == 1:
                    # "Skipped" is a special case of "Successful".
                    # When the test has been skipped, immedeately quit the
                    # test regardleess of `times` and `min_success` by raising
                    # SkipTest exception using the original reason.
                    instance.skipTest(result.skipped[0][1])
                elif result.wasSuccessful():
                    success_counter += 1
                else:
                    results.append(result)
                    failure_counter += 1
                if success_counter >= min_success:
                    instance.assertTrue(True)
                    return
                if failure_counter > times - min_success:
                    fail()
                    return
            fail()
        return wrapper
    return _repeat_with_success_at_least


def repeat(times, intensive_times=None):
    """Decorator that imposes the test to be successful in a row.

    Decorated test case is launched multiple times.
    The case is regarded as passed only if it is successful
    specified times in a row.

    .. note::
        In current implementation, this decorator grasps the
        failure information of each trial.

    Args:
        times(int): The number of trials in casual test.
        intensive_times(int or None): The number of trials in more intensive
            test. If ``None``, the same number as `times` is used.
    """
    if intensive_times is None:
        return repeat_with_success_at_least(times, times)

    casual_test = bool(int(os.environ.get('CUPY_TEST_CASUAL', '0')))
    times_ = times if casual_test else intensive_times
    return repeat_with_success_at_least(times_, times_)


def retry(times):
    """Decorator that imposes the test to be successful at least once.

    Decorated test case is launched multiple times.
    The case is regarded as passed if it is successful
    at least once.

    .. note::
        In current implementation, this decorator grasps the
        failure information of each trial.

    Args:
        times(int): The number of trials.
    """
    return repeat_with_success_at_least(times, 1)
