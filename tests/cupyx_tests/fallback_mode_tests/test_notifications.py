import contextlib
import io
import pytest
import unittest

from cupy import testing
from cupyx import fallback_mode
from cupyx import _ufunc_config


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

    def test_errstate(self):

        old = _ufunc_config.seterr(fallback_mode='print')
        before = _ufunc_config.geterr()

        with _ufunc_config.errstate(fallback_mode='raise'):
            inside = _ufunc_config.geterr()
            assert inside['fallback_mode'] == 'raise'

        after = _ufunc_config.geterr()
        assert before['fallback_mode'] == after['fallback_mode']
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
        saved_stdout = io.StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        _ufunc_config.seterr(**old)
        output = saved_stdout.getvalue().strip()
        assert output == ""

    def test_notification_print(self):

        old = _ufunc_config.seterr(fallback_mode='print')
        saved_stdout = io.StringIO()

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
