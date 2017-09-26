import numpy

from cupy import testing


class TestBinaryRepr(testing.NumpyAliasBasicTestBase):

    func = 'isscalar'


@testing.parameterize(
    *testing.product({
        'value': [
            0, 0.0, True,
            numpy.int32(1), numpy.array([1, 2], numpy.int32),
            numpy.complex(1), numpy.complex(1j), numpy.complex(1 + 1j),
            None, object(), 'abc', '', int, numpy.int32]}))
class TestBinaryReprValues(testing.NumpyAliasValuesTestBase):

    func = 'isscalar'

    def setUp(self):
        self.args = (self.value,)
