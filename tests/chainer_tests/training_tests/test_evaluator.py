import numpy as np
import unittest

from chainer import iterators
from chainer import link
from chainer import reporter as reporter_module
from chainer import testing

from chainer.functions.evaluation import accuracy
from chainer.training import evaluator as evaluator_module


class TestStandardEvaluator(unittest.TestCase):
    def test_it(self):
        class MyChain(link.Chain):
            def __call__(self, x, t):
                reporter_module.report({
                    'accuracy': accuracy.accuracy(x, t),
                })

        data = [
            (np.asarray([1, 0], dtype=np.float32), np.int32(0)),  # true
            (np.asarray([1, 0], dtype=np.float32), np.int32(0)),  # true
            (np.asarray([1, 0], dtype=np.float32), np.int32(0)),  # true
            (np.asarray([0, 1], dtype=np.float32), np.int32(1)),  # true
            (np.asarray([0, 1], dtype=np.float32), np.int32(1)),  # true
            (np.asarray([0, 1], dtype=np.float32), np.int32(1)),  # true
            (np.asarray([0, 1], dtype=np.float32), np.int32(0)),  # false
            (np.asarray([1, 0], dtype=np.float32), np.int32(1)),  # false
        ]
        iterator = iterators.SerialIterator(data, 4, repeat=False)

        model = MyChain()
        evaluator = evaluator_module.StandardEvaluator(iterator, model)

        result = evaluator.run()

        testing.assert_allclose(6. / 8, result['accuracy'])
