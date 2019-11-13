import pickle
import unittest

try:
    from cupy.cuda import cutensor
    cutensor_available = True
except Exception:
    cutensor_available = False


@unittest.skipUnless(cutensor_available, 'cuTensor is unavailable')
class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = cutensor.CuTensorError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)
