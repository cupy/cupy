from cupy import testing


class TestBinaryRepr(testing.NumpyAliasBasicTestBase):

    func = 'binary_repr'


@testing.parameterize(
    *testing.product({
        'args': [
            (0,), (3,), (-3,),
            (0, 0),
            (3, 5),
            (-3, 5),
            (3, 0),
            #(-3, 0),  # NumPy<1.12 did not support insufficient width
        ]}))
class TestBinaryReprValues(testing.NumpyAliasValuesTestBase):

    func = 'binary_repr'


class TestBaseRepr(testing.NumpyAliasBasicTestBase):

    func = 'base_repr'


@testing.parameterize(
    *testing.product({
        'args': [
            (0,), (5,), (-5,),
            (0, 2), (0, 10), (0, 36),
            (5, 2), (5, 10), (5, 36),
            (-5, 2), (-5, 10), (-5, 36),
            (-5, 2, 0),
            (-5, 2, 2),
            (-5, 2, 10),
            (5, 2, 0),
            (5, 2, 2),
            (5, 2, 10),
        ]}))
class TestBaseReprValues(testing.NumpyAliasValuesTestBase):

    func = 'base_repr'
