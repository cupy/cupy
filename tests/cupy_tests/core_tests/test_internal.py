from __future__ import annotations

import numpy
import pytest

from cupy._core import internal


class TestProd:

    def test_empty(self):
        assert internal.prod([]) == 1

    def test_one(self):
        assert internal.prod([2]) == 2

    def test_two(self):
        assert internal.prod([2, 3]) == 6


class TestProdSequence:

    def test_empty(self):
        assert internal.prod_sequence(()) == 1

    def test_one(self):
        assert internal.prod_sequence((2,)) == 2

    def test_two(self):
        assert internal.prod_sequence((2, 3)) == 6


class TestGetSize:

    def check_collection(self, a):
        assert internal.get_size(a) == tuple(a)

    def test_list(self):
        self.check_collection([1, 2, 3])

    def test_tuple(self):
        self.check_collection((1, 2, 3))

    def test_int(self):
        assert internal.get_size(1) == (1,)

    def test_numpy_int(self):
        assert internal.get_size(numpy.int32(1)) == (1,)

    def test_numpy_zero_dim_ndarray(self):
        assert internal.get_size(numpy.array(1)) == (1,)

    def test_tuple_of_numpy_scalars(self):
        assert internal.get_size((numpy.int32(1), numpy.array(1))) == (1, 1)

    @pytest.mark.parametrize(
        'value', [True, numpy.bool_(True), numpy.array(True, dtype='?')])
    def test_bool(self, value):
        with pytest.raises(TypeError):
            internal.get_size(value)
        with pytest.raises(TypeError):
            internal.get_size((value, value))

    def test_float(self):
        # `internal.get_size` is not responsible to interpret values as
        # integers.
        assert internal.get_size(1.0) == (1.0,)


class TestVectorEqual:

    def test_empty(self):
        assert internal.vector_equal([], []) is True

    def test_not_equal(self):
        assert internal.vector_equal([1, 2, 3], [1, 2, 0]) is False

    def test_equal(self):
        assert internal.vector_equal([-1, 0, 1], [-1, 0, 1]) is True

    def test_different_size(self):
        assert internal.vector_equal([1, 2, 3], [1, 2]) is False


class TestGetCContiguity:

    def test_zero_in_shape(self):
        assert internal.get_c_contiguity((1, 0, 1), (1, 1, 1), 3)

    def test_all_one_shape(self):
        assert internal.get_c_contiguity((1, 1, 1), (1, 1, 1), 3)

    def test_normal1(self):
        assert internal.get_c_contiguity((3, 4, 3), (24, 6, 2), 2)

    def test_normal2(self):
        assert internal.get_c_contiguity((3, 1, 3), (6, 100, 2), 2)

    def test_normal3(self):
        assert internal.get_c_contiguity((3,), (4, ), 4)

    def test_normal4(self):
        assert internal.get_c_contiguity((), (), 4)

    def test_normal5(self):
        assert internal.get_c_contiguity((3, 1), (4, 8), 4)

    def test_no_contiguous1(self):
        assert not internal.get_c_contiguity((3, 4, 3), (30, 6, 2), 2)

    def test_no_contiguous2(self):
        assert not internal.get_c_contiguity((3, 1, 3), (24, 6, 2), 2)

    def test_no_contiguous3(self):
        assert not internal.get_c_contiguity((3, 1, 3), (6, 6, 4), 2)


class TestInferUnknownDimension:

    def test_known_all(self):
        assert internal.infer_unknown_dimension((1, 2, 3), 6) == [1, 2, 3]

    def test_multiple_unknown(self):
        with pytest.raises(ValueError):
            internal.infer_unknown_dimension((-1, 1, -1), 10)

    def test_infer(self):
        assert internal.infer_unknown_dimension((-1, 2, 3), 12) == [2, 2, 3]


@pytest.mark.parametrize(
    ("slice_", "expect"),
    [
        ((2, 8, 1),    (2, 8, 1)),
        ((2, None, 1), (2, 10, 1)),
        ((2, 1, 1),    (2, 2, 1)),
        ((2, -1, 1),   (2, 9, 1)),

        ((None, 8, 1),  (0, 8, 1)),
        ((-3, 8, 1),    (7, 8, 1)),
        ((11, 8, 1),    (10, 10, 1)),
        ((11, 11, 1),   (10, 10, 1)),
        ((-11, 8, 1),   (0, 8, 1)),
        ((-11, -11, 1), (0, 0, 1)),

        ((8, 2, -1),    (8, 2, -1)),
        ((8, None, -1), (8, -1, -1)),
        ((8, 9, -1),    (8, 8, -1)),
        ((8, -3, -1),   (8, 7, -1)),

        ((None, 8, -1), (9, 8, -1)),
        ((-3, 6, -1),   (7, 6, -1)),

        ((10, 10, -1),  (9, 9, -1)),
        ((10, 8, -1),   (9, 8, -1)),
        ((9, 10, -1),   (9, 9, -1)),
        ((9, 9, -1),    (9, 9, -1)),
        ((9, 8, -1),    (9, 8, -1)),
        ((8, 8, -1),    (8, 8, -1)),
        ((-9, -8, -1),  (1, 1, -1)),
        ((-9, -9, -1),  (1, 1, -1)),
        ((-9, -10, -1), (1, 0, -1)),
        ((-9, -11, -1), (1, -1, -1)),
        ((-9, -12, -1), (1, -1, -1)),
        ((-10, -9, -1), (0, 0, -1)),
        ((-10, -10, -1), (0, 0, -1)),
        ((-10, -11, -1), (0, -1, -1)),
        ((-10, -12, -1), (0, -1, -1)),
        ((-11, 8, -1),   (-1, -1, -1)),
        ((-11, -9, -1),  (-1, -1, -1)),
        ((-11, -10, -1), (-1, -1, -1)),
        ((-11, -11, -1), (-1, -1, -1)),
        ((-11, -12, -1), (-1, -1, -1)),
    ],
)
def test_complete_slice(slice_, expect):
    assert internal.complete_slice(
        slice(*slice_), 10) == slice(*expect)


class TestCompleteSliceError:

    def test_invalid_step_value(self):
        with pytest.raises(ValueError):
            internal.complete_slice(slice(1, 1, 0), 1)

    def test_invalid_step_type(self):
        with pytest.raises(TypeError):
            internal.complete_slice(slice(1, 1, (1, 2)), 1)

    def test_invalid_start_type(self):
        with pytest.raises(TypeError):
            internal.complete_slice(slice((1, 2), 1, 1), 1)
        with pytest.raises(TypeError):
            internal.complete_slice(slice((1, 2), 1, -1), 1)

    def test_invalid_stop_type(self):
        with pytest.raises(TypeError):
            internal.complete_slice(slice((1, 2), 1, 1), 1)
        with pytest.raises(TypeError):
            internal.complete_slice(slice((1, 2), 1, -1), 1)


@pytest.mark.parametrize(
    ("x", "expect"),
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 4),
        (2 ** 10,     2 ** 10),
        (2 ** 10 - 1, 2 ** 10),
        (2 ** 10 + 1, 2 ** 11),
        (2 ** 40,     2 ** 40),
        (2 ** 40 - 1, 2 ** 40),
        (2 ** 40 + 1, 2 ** 41),
    ],
)
def test_clp2(x, expect):
    assert internal.clp2(x) == expect
