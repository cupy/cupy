import unittest

import cupyx
from cupy import testing


@testing.parameterize(*testing.product({
    'divide': [None],
}))
class TestErrState(unittest.TestCase):

    def test_errstate(self):
        with cupyx.errstate(divide=self.divide):
            state = cupyx.geterr()
            assert state['divide'] == self.divide

    def test_seterr(self):
        pass


# TODO(hvy): Implement TestErrStateDivide

# TODO(hvy): Implement TestErrStateOver

# TODO(hvy): Implement TestErrStateUnder

# TODO(hvy): Implement TestErrStateInvalid
