import contextlib
import io

import pytest

from cupy import testing
from cupyx import fallback_mode
from cupyx import _ufunc_config


class NotificationTestBase:

    @pytest.fixture(autouse=True)
    def setUp(self):
        old_config = _ufunc_config.geterr()
        yield
        _ufunc_config.seterr(**old_config)


class TestNotifications(NotificationTestBase):

    def test_seterr_geterr(self):

        default = _ufunc_config.geterr()
        assert default['fallback_mode'] == 'ignore'

        old = _ufunc_config.seterr(fallback_mode='warn')
        current = _ufunc_config.geterr()
        assert old['fallback_mode'] == 'ignore'
        assert current['fallback_mode'] == 'warn'
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
    {'func_name': 'get_include'},
)
class TestNotificationModes(NotificationTestBase):

    @property
    def func(self):
        if self.func_name == 'get_include':
            return fallback_mode.numpy.get_include
        assert False

    def test_notification_ignore(self):

        old = _ufunc_config.seterr(fallback_mode='ignore')
        saved_stdout = io.StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            self.func()

        _ufunc_config.seterr(**old)
        output = saved_stdout.getvalue().strip()
        assert output == ""

    def test_notification_print(self):

        old = _ufunc_config.seterr(fallback_mode='print')
        saved_stdout = io.StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            self.func()

        _ufunc_config.seterr(**old)
        nf = self.func._numpy_object
        output = saved_stdout.getvalue().strip()
        msg1 = "'{}' method not in cupy, ".format(nf.__name__)
        msg2 = "falling back to '{}.{}'".format(nf.__module__, nf.__name__)
        assert output == ("Warning: " + msg1 + msg2)

    def test_notification_warn(self):

        _ufunc_config.seterr(fallback_mode='warn')

        with pytest.warns(fallback_mode.notification.FallbackWarning):
            self.func()

    def test_notification_raise(self):

        old = _ufunc_config.seterr(fallback_mode='raise')

        with pytest.raises(AttributeError):
            self.func()

        _ufunc_config.seterr(**old)
