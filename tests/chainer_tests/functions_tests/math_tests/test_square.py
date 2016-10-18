import unittest

import chainer.functions as F
from chainer import testing


@testing.unary_math_function_unittest(F.Square())
class TestSquare(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
