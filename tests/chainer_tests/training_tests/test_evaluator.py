import numpy
import unittest

from chainer import iterators
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import evaluator as evaluator_module


class TestStandardEvaluator(unittest.TestCase):
    def test_it(self):
        class MyChain(link.Chain):
            def __call__(self, x, t):
                reporter_module.report({
                    'accuracy': (x.data == t.data).mean(),
                })

        data = [
            (numpy.int32(0), numpy.int32(0)),  # true
            (numpy.int32(0), numpy.int32(0)),  # true
            (numpy.int32(0), numpy.int32(0)),  # true
            (numpy.int32(1), numpy.int32(1)),  # true
            (numpy.int32(1), numpy.int32(1)),  # true
            (numpy.int32(1), numpy.int32(1)),  # true
            (numpy.int32(0), numpy.int32(1)),  # false
            (numpy.int32(1), numpy.int32(0)),  # false
        ]
        iterator = iterators.SerialIterator(data, 4, repeat=False)

        model = MyChain()
        evaluator = evaluator_module.StandardEvaluator(iterator, model)

        result = evaluator.run()

        expected_result = {
            'accuracy': numpy.float32(6 / 8)
        }

        self.assertDictEqual(expected_result, result)
