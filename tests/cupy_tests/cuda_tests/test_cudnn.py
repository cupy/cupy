import pickle
import unittest

from cupy.cuda import cudnn


cudnn_available = cudnn.available


@unittest.skipUnless(cudnn_available, 'cuDNN is unavailable')
class TestExceptionPicklable(unittest.TestCase):

    def test(self):
        e1 = cudnn.CuDNNError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)
