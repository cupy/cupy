import contextlib
import io
import pytest
import unittest

import numpy

from cupy import testing
from cupyx import fallback_mode
from cupyx import _ufunc_config
from cupyx_tests.fallback_mode_tests import test_fallback as test_utils


class NotificationTestBase(unittest.TestCase):

    def setUp(self):
        self.old_config = _ufunc_config.geterr()

    def tearDown(self):
        _ufunc_config.seterr(**self.old_config)


@testing.gpu
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
    {'func': fallback_mode.numpy.array_equiv, 'shape': (3, 4)},
)
@testing.gpu
class TestNotificationModes(NotificationTestBase):

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


@testing.gpu
class TestNotificationVectorize(NotificationTestBase):

    @test_utils.enable_slice_copy
    def test_custom_or_builtin_pyfunc(self):

        old = _ufunc_config.seterr(fallback_mode='print')
        saved_stdout = io.StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            def custom_abs(x):
                if x >= 0:
                    return x
                return -x

            a = testing.shaped_random((3, 4), fallback_mode.numpy)
            vec_abs = fallback_mode.numpy.vectorize(custom_abs)
            vec_abs(a)

            # built-in
            vec_abs = fallback_mode.numpy.vectorize(abs)
            vec_abs(a)

        _ufunc_config.seterr(**old)
        output = saved_stdout.getvalue().strip()
        msg = "'vectorize' method not in cupy, "
        msg += "falling back to '"
        msg += numpy.vectorize.__module__ + ".vectorize'"
        assert output == ("Warning: " + msg + "\nWarning: " + msg)

    @test_utils.enable_slice_copy
    def test_cupy_supported_pyfunc(self):

        old = _ufunc_config.seterr(fallback_mode='print')
        saved_stdout = io.StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random((3, 4), fallback_mode.numpy)
            vec_abs = fallback_mode.numpy.vectorize(fallback_mode.numpy.abs)
            vec_abs(a)

        _ufunc_config.seterr(**old)
        output = saved_stdout.getvalue().strip()
        msg1 = "'vectorize' method not in cupy, "
        msg1 += "falling back to '"
        msg1 += numpy.vectorize.__module__ + ".vectorize'"
        msg2 = "'absolute' method is available in cupy but cannot be used, "
        msg2 += "falling back to its numpy implementation"
        assert output == ("Warning: " + msg1 + "\nWarning: " + msg2)

    @test_utils.enable_slice_copy
    def test_numpy_only_pyfunc(self):

        old = _ufunc_config.seterr(fallback_mode='print')
        saved_stdout = io.StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random((3, 4), fallback_mode.numpy)
            vec_abs = fallback_mode.numpy.vectorize(fallback_mode.numpy.fabs)
            vec_abs(a)

        _ufunc_config.seterr(**old)
        output = saved_stdout.getvalue().strip()
        msg1 = "'vectorize' method not in cupy, "
        msg1 += "falling back to '"
        msg1 += numpy.vectorize.__module__ + ".vectorize'"
        msg2 = "'fabs' method not in cupy, "
        msg2 += "falling back to its numpy implementation"
        assert output == ("Warning: " + msg1 + "\nWarning: " + msg2)
