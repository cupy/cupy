import sys
import pytest
import unittest
import contextlib

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

from cupy import testing
from cupyx import fallback_mode
from cupyx._ufunc_config import geterr, seterr, errstate


if sys.version_info < (3, 4):

    class redirect_stdout:

        _stream = "stdout"

        def __init__(self, new_target):
            self._new_target = new_target
            self._old_targets = []

        def __enter__(self):
            self._old_targets.append(getattr(sys, self._stream))
            setattr(sys, self._stream, self._new_target)
            return self._new_target

        def __exit__(self, exctype, excinst, exctb):
            setattr(sys, self._stream, self._old_targets.pop())

    setattr(contextlib, 'redirect_stdout', redirect_stdout)


@testing.gpu
class TestNotifications(unittest.TestCase):

    def test_seterr_geterr(self):

        default = geterr()
        assert default['fallback_mode'] == 'warn'

        old = seterr(fallback_mode='ignore')
        current = geterr()
        assert old['fallback_mode'] == 'warn'
        assert current['fallback_mode'] == 'ignore'
        seterr(**old)

    def test_geterrcall(self):

        def f(func):
            pass

        fallback_mode.seterrcall(f)
        current = fallback_mode.geterrcall()

        assert current == f

    def test_errstate(self):

        old = seterr(fallback_mode='print')
        before = geterr()

        with errstate(fallback_mode='raise'):
            inside = geterr()
            assert inside['fallback_mode'] == 'raise'

        after = geterr()
        assert before['fallback_mode'] == after['fallback_mode']
        seterr(**old)

    def test_errstate_func(self):

        def f(func):
            pass

        old = fallback_mode.seterr('call')
        fallback_mode.seterrcall(f)

        before = fallback_mode.geterr()
        before_func = fallback_mode.geterrcall()

        class L:
            def write(msg):
                pass

        log_obj = L()

        with fallback_mode.errstate('log', log_obj):
            inside = fallback_mode.geterr()
            inside_func = fallback_mode.geterrcall()

            assert inside == 'log'
            assert inside_func is log_obj

        after = fallback_mode.geterr()
        after_func = fallback_mode.geterrcall()

        assert before == after
        assert before_func is after_func
        fallback_mode.seterr(old)


@testing.parameterize(
    {'func': fallback_mode.numpy.array_equal, 'shape': (3, 4)},
    {'func': fallback_mode.numpy.array_equiv, 'shape': (3, 4)},
    {'func': fallback_mode.numpy.polyadd, 'shape': (2, 3)},
    {'func': fallback_mode.numpy.convolve, 'shape': (5,)}
)
@testing.gpu
class TestNotificationModes(unittest.TestCase):

    def test_notification_ignore(self):

        old = seterr(fallback_mode='ignore')
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        seterr(**old)
        output = saved_stdout.getvalue().strip()
        assert output == ""

    def test_notification_print(self):

        old = seterr(fallback_mode='print')
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        seterr(**old)
        nf = self.func._numpy_object
        output = saved_stdout.getvalue().strip()
        msg1 = "'{}' method not in cupy, ".format(nf.__name__)
        msg2 = "falling back to '{}.{}'".format(nf.__module__, nf.__name__)
        assert output == ("Warning: " + msg1 + msg2)

    def test_notification_warn(self):

        seterr(fallback_mode='warn')

        with pytest.warns(fallback_mode.notification.FallbackWarning):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

    def test_notification_raise(self):

        old = seterr(fallback_mode='raise')

        with pytest.raises(AttributeError):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        seterr(**old)

    def test_notification_call(self):

        def custom_callback(func):
            print("'{}' fallbacked".format(func.__name__))

        old = fallback_mode.seterr('call')
        fallback_mode.seterrcall(custom_callback)
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        fallback_mode.seterr(old)
        nf = self.func._numpy_object
        output = saved_stdout.getvalue().strip()
        assert output == "'{}' fallbacked".format(nf.__name__)

    def test_notification_log(self):

        class Log:
            def write(self, msg):
                print("LOG: {}".format(msg))

        log_obj = Log()
        old = fallback_mode.seterr('log')
        fallback_mode.seterrcall(log_obj)
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        fallback_mode.seterr(old)
        nf = self.func._numpy_object
        output = saved_stdout.getvalue().strip()
        msg1 = "'{}' method not in cupy, ".format(nf.__name__)
        msg2 = "falling back to '{}.{}'".format(nf.__module__, nf.__name__)
        assert output == ("LOG: " + msg1 + msg2)
