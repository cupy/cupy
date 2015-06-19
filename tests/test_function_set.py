import cPickle as pickle
from unittest import TestCase
from chainer import cuda, FunctionSet
from chainer.functions import Linear
from chainer.testing import attr

if cuda.available:
    cuda.init()

class TestFunctionSet(TestCase):
    def setUp(self):
        self.fs = FunctionSet(
            a = Linear(3, 2),
            b = Linear(3, 2)
        )

    def check_equal_fs(self, fs1, fs2):
        self.assertTrue((fs1.a.W == fs2.a.W).all())
        self.assertTrue((fs1.a.b == fs2.a.b).all())
        self.assertTrue((fs1.b.W == fs2.b.W).all())
        self.assertTrue((fs1.b.b == fs2.b.b).all())

    def test_pickle_cpu(self):
        s   = pickle.dumps(self.fs)
        fs2 = pickle.loads(s)
        self.check_equal_fs(self.fs, fs2)

    @attr.gpu
    def test_pickle_gpu(self):
        self.fs.to_gpu()
        s   = pickle.dumps(self.fs)
        fs2 = pickle.loads(s)

        self.fs.to_cpu()
        fs2.to_cpu()
        self.check_equal_fs(self.fs, fs2)
