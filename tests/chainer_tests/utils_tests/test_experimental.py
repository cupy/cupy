import unittest
import warnings

import chainer
from chainer import testing
from chainer import utils


def f():
    utils.experimental('test.f')


def g():
    utils.experimental()


def h(x):
    utils.experimental()


class C(object):

    def __init__(self):
        utils.experimental()


class TestExperimental(unittest.TestCase):

    def setUp(self):
        self.original = chainer.disable_experimental_feature_warning
        chainer.disable_experimental_feature_warning = False

    def tearDown(self):
        chainer.disable_experimental_feature_warning = self.original

    def test_experimental_with_api_name(self):
        with warnings.catch_warnings(record=True) as w:
            f()

        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert 'test.f is an experimental API.' in str(w[0].message)

    def test_experimental_with_no_api_name(self):
        with warnings.catch_warnings(record=True) as w:
            g()

        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert 'g is an experimental API.' in str(w[0].message)

    def test_experimental_with_no_api_name_2(self):
        with warnings.catch_warnings(record=True) as w:
            C()

        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert 'C.__init__ is an experimental API.' in str(w[0].message)

    def test_multiple_same_function_calls(self):
        with warnings.catch_warnings(record=True) as w:
            f()
            f()

        assert len(w) == 1

    def test_different_functions(self):
        with warnings.catch_warnings(record=True) as w:
            f()
            g()

        assert len(w) == 2

    def test_multiple_same_class_instantiation(self):
        with warnings.catch_warnings(record=True) as w:
            C()
            C()

        assert len(w) == 1

    def test_multiple_calls_with_different_argument(self):
        with warnings.catch_warnings(record=True) as w:
            h(0)
            h(1)

        assert len(w) == 1


class TestDisableExperimentalWarning(unittest.TestCase):

    def setUp(self):
        self.original = chainer.disablpe_experimental_feature_warning
        chainer.disable_experimental_feature_warning = True

    def tearDown(self):
        chainer.disable_experimental_feature_warning = self.original

    def test_experimental(self):
        with warnings.catch_warnings(record=True) as w:
            f()

        assert len(w) == 0


testing.run_module(__name__, __file__)
