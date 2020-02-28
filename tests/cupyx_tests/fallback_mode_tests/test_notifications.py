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
from cupyx import _ufunc_config


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

        default = _ufunc_config.geterr()
        assert default['fallback_mode'] == 'warn'

        old = _ufunc_config.seterr(fallback_mode='ignore')
        current = _ufunc_config.geterr()
        assert old['fallback_mode'] == 'warn'
        assert current['fallback_mode'] == 'ignore'
        _ufunc_config.seterr(**old)

    def test_geterrcall(self):

        def f(func):
            pass

        _ufunc_config.seterrcall(f)
        current = _ufunc_config.geterrcall()

        assert current['fallback_mode'] == f

    def test_errstate(self):

        old = _ufunc_config.seterr(fallback_mode='print')
        before = _ufunc_config.geterr()

        with _ufunc_config.errstate(fallback_mode='raise'):
            inside = _ufunc_config.geterr()
            assert inside['fallback_mode'] == 'raise'

        after = _ufunc_config.geterr()
        assert before['fallback_mode'] == after['fallback_mode']
        _ufunc_config.seterr(**old)

    def test_errstate_func(self):

        def f(func):
            pass

        old = _ufunc_config.seterr(fallback_mode='call')
        _ufunc_config.seterrcall(fallback_mode=f)

        before = _ufunc_config.geterr()
        before_func = _ufunc_config.geterrcall()

        class L:
            def write(msg):
                pass

        log_obj = L()

        with _ufunc_config.errstate(fallback_mode='log', fallback_mode_callback=log_obj):
            inside = _ufunc_config.geterr()
            inside_func = _ufunc_config.geterrcall()

            assert inside['fallback_mode'] == 'log'
            assert inside_func['fallback_mode'] is log_obj

        after = _ufunc_config.geterr()
        after_func = _ufunc_config.geterrcall()

        assert before['fallback_mode'] == after['fallback_mode']
        assert before_func['fallback_mode'] is after_func['fallback_mode']
        _ufunc_config.seterr(**old)


@testing.parameterize(
    {'func': fallback_mode.numpy.array_equal, 'shape': (3, 4)},
    {'func': fallback_mode.numpy.array_equiv, 'shape': (3, 4)},
    {'func': fallback_mode.numpy.polyadd, 'shape': (2, 3)},
    {'func': fallback_mode.numpy.convolve, 'shape': (5,)}
)
@testing.gpu
class TestNotificationModes(unittest.TestCase):

    def test_notification_ignore(self):

        old = _ufunc_config.seterr(fallback_mode='ignore')
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        _ufunc_config.seterr(**old)
        output = saved_stdout.getvalue().strip()
        assert output == ""

    def test_notification_print(self):

        old = _ufunc_config.seterr(fallback_mode='print')
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        _ufunc_config.seterr(**old)
        nf = self.func._numpy_object
        output = saved_stdout.getvalue().strip()
        msg1 = "'{}' method not in cupy, ".format(nf.__name__)
        msg2 = "falling back to '{}.{}'".format(nf.__module__, nf.__name__)
        assert output == ("Warning: " + msg1 + msg2)

    def test_notification_warn(self):

        _ufunc_config.seterr(fallback_mode='warn')

        with pytest.warns(fallback_mode.notification.FallbackWarning):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

    def test_notification_raise(self):

        old = _ufunc_config.seterr(fallback_mode='raise')

        with pytest.raises(AttributeError):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        _ufunc_config.seterr(**old)

    def test_notification_call(self):

        def custom_callback(func):
            print("'{}' fallbacked".format(func.__name__))

        old = _ufunc_config.seterr(fallback_mode='call')
        _ufunc_config.seterrcall(fallback_mode=custom_callback)
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        _ufunc_config.seterr(**old)
        nf = self.func._numpy_object
        output = saved_stdout.getvalue().strip()
        assert output == "'{}' fallbacked".format(nf.__name__)

    def test_notification_log(self):

        class Log:
            def write(self, msg):
                print("LOG: {}".format(msg))

        log_obj = Log()
        old = _ufunc_config.seterr(fallback_mode='log')
        _ufunc_config.seterrcall(fallback_mode=log_obj)
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        _ufunc_config.seterr(**old)
        nf = self.func._numpy_object
        output = saved_stdout.getvalue().strip()
        msg1 = "'{}' method not in cupy, ".format(nf.__name__)
        msg2 = "falling back to '{}.{}'".format(nf.__module__, nf.__name__)
        assert output == ("LOG: " + msg1 + msg2)
