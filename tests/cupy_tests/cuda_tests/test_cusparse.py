import pickle
import unittest

import cupy
from cupy.cuda import cusparse


class TestException(unittest.TestCase):

    def test_error_message(self):
        e = cusparse.CuSparseError(1)
        if cupy.cuda.runtime.is_hip:
            assert str(e) == (
                'HIPSPARSE_STATUS_NOT_INITIALIZED: '
                'HIPSPARSE_STATUS_NOT_INITIALIZED'
            )
        else:
            assert str(e) == (
                'CUSPARSE_STATUS_NOT_INITIALIZED: '
                'initialization error'
            )

    def test_pickle(self):
        e1 = cusparse.CuSparseError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)
