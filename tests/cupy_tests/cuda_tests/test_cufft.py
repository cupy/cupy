import pickle
import unittest

from cupy.cuda import cufft


class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = cufft.CuFFTError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)
