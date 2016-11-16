import unittest
import chainer
from chainer import utils
import warnings


def f():
    utils.experimental('test.f')

def g():
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



class TestDisableExperimentalWarning(unittest.TestCase):

    def setUp(self):
        self.original = chainer.disable_experimental_feature_warning
        chainer.disable_experimental_feature_warning = True

    def tearDown(self):
        chainer.disable_experimental_feature_warning = self.original

    def test_experimental(self):
        with warnings.catch_warnings(record=True) as w:
            f()

        assert len(w) == 0
