import unittest

import cupyx
from cupy import testing


@testing.parameterize(*testing.product({
    'divide': [None],
}))
class TestErrState(unittest.TestCase):

    def test_errstate(self):
        orig = cupyx.geterr()
        with cupyx.errstate(divide=self.divide):
            state = cupyx.geterr()
            assert state['divide'] == self.divide
            for key in state:
                if key != 'divide':
                    assert state[key] == orig[key]

    def test_seterr(self):
        pass


# TODO(hvy): Implement TestErrStateDivide

# TODO(hvy): Implement TestErrStateOver

# TODO(hvy): Implement TestErrStateUnder

# TODO(hvy): Implement TestErrStateInvalid
